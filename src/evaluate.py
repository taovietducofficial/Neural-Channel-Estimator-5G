import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import CDL, AntennaArray
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver

from src.baseline import CARRIER_FREQUENCY, ebno_to_no
from src.models import CNNChannelEstimator, TransformerChannelEstimator

MODELS = {"cnn": CNNChannelEstimator, "transformer": TransformerChannelEstimator}


def build_eval_link(device="cpu"):
    pusch_config = PUSCHConfig()
    transmitter = PUSCHTransmitter(pusch_config, device=device)
    array_kwargs = dict(
        num_rows=1, num_cols=1, polarization="single", polarization_type="V",
        antenna_pattern="omni", carrier_frequency=CARRIER_FREQUENCY, device=device,
    )
    cdl = CDL(
        model="C", delay_spread=100e-9, carrier_frequency=CARRIER_FREQUENCY,
        ut_array=AntennaArray(**array_kwargs), bs_array=AntennaArray(**array_kwargs),
        direction="uplink", max_speed=3.0, device=device,
    )
    channel = OFDMChannel(
        channel_model=cdl, resource_grid=transmitter.resource_grid,
        normalize_channel=True, device=device,
    )
    return pusch_config, transmitter, channel


@torch.no_grad()
def bler_curve(receiver, transmitter, channel, snr_db_range, batch_size, num_batches):
    blers = []
    for snr_db in snr_db_range:
        no = ebno_to_no(snr_db)
        errors, total = 0, 0
        for _ in range(num_batches):
            x, b = transmitter(batch_size)
            y = channel(x, no)
            b_hat = receiver(y, no)
            errors += (b_hat != b).any(dim=-1).sum().item()
            total += b.shape[0] * b.shape[1]
        blers.append(errors / total)
    return blers


def main(snr_min=-4.0, snr_max=10.0, snr_step=2.0, batch_size=16, num_batches=8,
         checkpoints_dir="checkpoints", out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    snr_range = np.arange(snr_min, snr_max + 1e-9, snr_step)
    pusch_config, transmitter, channel = build_eval_link()

    curves = {}

    baseline_rx = PUSCHReceiver(transmitter)
    print("Evaluating baseline (LS + LMMSE)...")
    curves["baseline_ls"] = bler_curve(baseline_rx, transmitter, channel, snr_range, batch_size, num_batches)

    for name, cls in MODELS.items():
        ckpt = os.path.join(checkpoints_dir, f"{name}.pt")
        if not os.path.exists(ckpt):
            print(f"skip {name}: no checkpoint at {ckpt}")
            continue
        model = cls(transmitter.resource_grid, pusch_config)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()
        rx = PUSCHReceiver(transmitter, channel_estimator=model)
        print(f"Evaluating neural ({name})...")
        curves[name] = bler_curve(rx, transmitter, channel, snr_range, batch_size, num_batches)

    csv_path = os.path.join(out_dir, "bler_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["snr_db"] + list(curves.keys()))
        for i, snr_db in enumerate(snr_range):
            writer.writerow([snr_db] + [curves[k][i] for k in curves])
    print("saved", csv_path)

    plt.figure(figsize=(6, 4.5))
    for name, bler in curves.items():
        plt.semilogy(snr_range, bler, marker="o", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLER")
    plt.title("BLER vs SNR: classical LS vs neural channel estimators")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "bler_vs_snr.png")
    plt.savefig(plot_path, dpi=150)
    print("saved", plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snr-min", type=float, default=-4.0)
    parser.add_argument("--snr-max", type=float, default=10.0)
    parser.add_argument("--snr-step", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=8)
    args = parser.parse_args()
    main(args.snr_min, args.snr_max, args.snr_step, args.batch_size, args.num_batches)
