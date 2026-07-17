import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from src.baseline import build_link, ebno_to_no, run_bler
from src.models import CNNChannelEstimator, TransformerChannelEstimator


def test_estimator_shapes():
    pusch_config = PUSCHConfig()
    transmitter = PUSCHTransmitter(pusch_config)
    _, channel, _ = build_link()
    x, _ = transmitter(4)
    no = ebno_to_no(5.0)
    y = channel(x, no)

    for cls in (CNNChannelEstimator, TransformerChannelEstimator):
        model = cls(transmitter.resource_grid, pusch_config)
        h_hat, err_var = model(y, no)
        assert h_hat.shape == (4, 1, 1, 1, 1, 14, 48), (cls.__name__, h_hat.shape)
        assert torch.is_complex(h_hat)
        rx = PUSCHReceiver(transmitter, channel_estimator=model)
        b_hat = rx(y, no)
        assert b_hat.shape[0] == 4
    print("test_estimator_shapes: OK")


def test_bler_decreases_with_snr():
    snr_range = np.array([-4.0, 10.0])
    blers = run_bler(snr_range, batch_size=16, num_batches=4)
    assert blers[0] >= blers[1], f"expected BLER to drop as SNR rises, got {blers}"
    print("test_bler_decreases_with_snr: OK", blers)


if __name__ == "__main__":
    test_estimator_shapes()
    test_bler_decreases_with_snr()
    print("all checks passed")
