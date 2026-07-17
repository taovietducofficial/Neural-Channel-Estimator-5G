import argparse
import json
import os
from datetime import datetime, timezone

import torch
import torch.nn as nn
from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import CDL, AntennaArray
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter

from src.baseline import CARRIER_FREQUENCY, ebno_to_no
from src.models import CNNChannelEstimator, TransformerChannelEstimator

MODELS = {"cnn": CNNChannelEstimator, "transformer": TransformerChannelEstimator}


def build_training_link(device="cpu", num_layers=1):
    pusch_config = PUSCHConfig()
    if num_layers > 1:
        pusch_config.num_layers = num_layers
        pusch_config.num_antenna_ports = num_layers
        pusch_config.tpmi = 0
    transmitter = PUSCHTransmitter(pusch_config, device=device)
    ut_kwargs = dict(
        num_rows=1, num_cols=num_layers, polarization="single", polarization_type="V",
        antenna_pattern="omni", carrier_frequency=CARRIER_FREQUENCY, device=device,
    )
    bs_kwargs = dict(
        num_rows=1, num_cols=num_layers, polarization="single", polarization_type="V",
        antenna_pattern="omni", carrier_frequency=CARRIER_FREQUENCY, device=device,
    )
    cdl = CDL(
        model="C", delay_spread=100e-9, carrier_frequency=CARRIER_FREQUENCY,
        ut_array=AntennaArray(**ut_kwargs), bs_array=AntennaArray(**bs_kwargs),
        direction="uplink", max_speed=3.0, device=device,
    )
    channel = OFDMChannel(
        channel_model=cdl, resource_grid=transmitter.resource_grid,
        normalize_channel=True, return_channel=True, device=device,
    )
    return pusch_config, transmitter, channel


def train(model_name, steps=500, batch_size=32, snr_min=-4.0, snr_max=14.0,
          lr=1e-3, device="cpu", seed=0, num_layers=1):
    torch.manual_seed(seed)
    pusch_config, transmitter, channel = build_training_link(device=device, num_layers=num_layers)
    model = MODELS[model_name](transmitter.resource_grid, pusch_config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    final_loss = None
    for step in range(1, steps + 1):
        x, _ = transmitter(batch_size)
        snr_db = torch.empty(1).uniform_(snr_min, snr_max).item()
        no = ebno_to_no(snr_db)
        y, h_perfect = channel(x, no)

        h_hat, _ = model(y, no)
        target = torch.stack([h_perfect.real, h_perfect.imag], dim=-1)
        pred = torch.stack([h_hat.real, h_hat.imag], dim=-1)
        loss = loss_fn(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = loss.item()

        if step % 50 == 0 or step == 1:
            print(f"step {step:4d}/{steps}  loss={loss.item():.6f}")

    return model, final_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default="cnn")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--snr-min", type=float, default=-4.0)
    parser.add_argument("--snr-max", type=float, default=14.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    trained, final_loss = train(
        args.model, steps=args.steps, batch_size=args.batch_size,
        snr_min=args.snr_min, snr_max=args.snr_max, lr=args.lr, seed=args.seed,
        num_layers=args.num_layers,
    )
    out_path = args.out or f"checkpoints/{args.model}.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(trained.state_dict(), out_path)

    meta = {
        "model": args.model,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "snr_min_db": args.snr_min,
        "snr_max_db": args.snr_max,
        "lr": args.lr,
        "seed": args.seed,
        "num_layers": args.num_layers,
        "final_loss": final_loss,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = os.path.splitext(out_path)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("saved to", out_path)
    print("saved metadata to", meta_path)
