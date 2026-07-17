import numpy as np
import torch

from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import CDL, Antenna, AntennaArray
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver

CARRIER_FREQUENCY = 3.5e9


def build_link(cdl_model="C", delay_spread=100e-9, device="cpu"):
    pusch_config = PUSCHConfig()
    transmitter = PUSCHTransmitter(pusch_config, device=device)

    ut_array = AntennaArray(
        num_rows=1, num_cols=1, polarization="single",
        polarization_type="V", antenna_pattern="omni",
        carrier_frequency=CARRIER_FREQUENCY, device=device,
    )
    bs_array = AntennaArray(
        num_rows=1, num_cols=1, polarization="single",
        polarization_type="V", antenna_pattern="omni",
        carrier_frequency=CARRIER_FREQUENCY, device=device,
    )
    cdl = CDL(
        model=cdl_model, delay_spread=delay_spread,
        carrier_frequency=CARRIER_FREQUENCY,
        ut_array=ut_array, bs_array=bs_array,
        direction="uplink", min_speed=0.0, max_speed=3.0,
        device=device,
    )
    channel = OFDMChannel(
        channel_model=cdl, resource_grid=transmitter.resource_grid,
        normalize_channel=True, device=device,
    )
    receiver = PUSCHReceiver(transmitter, device=device)
    return transmitter, channel, receiver


def ebno_to_no(ebno_db, num_bits_per_symbol=2, coderate=0.5):
    ebno = 10 ** (ebno_db / 10)
    return 1 / (ebno * num_bits_per_symbol * coderate)


def run_bler(snr_db_range, batch_size=32, num_batches=10, device="cpu", seed=1):
    torch.manual_seed(seed)
    transmitter, channel, receiver = build_link(device=device)

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
        bler = errors / total
        blers.append(bler)
        print(f"SNR={snr_db:5.1f} dB  BLER={bler:.4f}")
    return np.array(blers)


if __name__ == "__main__":
    snr_range = np.arange(-4, 11, 2)
    run_bler(snr_range, batch_size=16, num_batches=5)
