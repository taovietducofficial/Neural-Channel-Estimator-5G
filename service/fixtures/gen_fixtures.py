import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sionna.phy.nr import PUSCHLSChannelEstimator
from src.baseline import ebno_to_no
from src.train import build_training_link

BUCKETS = {"low": -2.0, "mid": 8.0, "high": 14.0}
BATCH = 8


def to_grid(h):
    shape = h.shape
    n = int(np.prod(shape[:-2]))
    flat = h.reshape(n, shape[-2], shape[-1])
    return torch.stack([flat.real, flat.imag], dim=1).numpy()


def main():
    torch.manual_seed(7)
    pusch_config, transmitter, channel = build_training_link()
    ls_estimator = PUSCHLSChannelEstimator(
        transmitter.resource_grid,
        dmrs_length=pusch_config.dmrs.length,
        dmrs_additional_position=pusch_config.dmrs.additional_position,
        num_cdm_groups_without_data=pusch_config.dmrs.num_cdm_groups_without_data,
    )

    out_dir = os.path.dirname(__file__)
    for bucket, snr_db in BUCKETS.items():
        with torch.no_grad():
            x, _ = transmitter(BATCH)
            no = ebno_to_no(snr_db)
            y, h_perfect = channel(x, no)
            h_ls, _ = ls_estimator(y, no)

        noisy = to_grid(h_ls)
        perfect = to_grid(h_perfect)
        out_path = os.path.join(out_dir, f"{bucket}.npz")
        np.savez(out_path, noisy=noisy, perfect=perfect, snr_db=snr_db)
        mse = float(np.mean((noisy - perfect) ** 2))
        print(f"{bucket} (SNR={snr_db}dB): saved {out_path}, raw-LS MSE={mse:.6f}")


if __name__ == "__main__":
    main()
