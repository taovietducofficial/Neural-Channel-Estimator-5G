import torch
import torch.nn as nn

from sionna.phy.nr import PUSCHLSChannelEstimator


class _BaseNeuralEstimator(nn.Module):
    def __init__(self, resource_grid, pusch_config):
        super().__init__()
        self.ls_estimator = PUSCHLSChannelEstimator(
            resource_grid,
            dmrs_length=pusch_config.dmrs.length,
            dmrs_additional_position=pusch_config.dmrs.additional_position,
            num_cdm_groups_without_data=pusch_config.dmrs.num_cdm_groups_without_data,
        )

    def _to_grid(self, h_ls):
        shape = h_ls.shape
        n = int(torch.prod(torch.tensor(shape[:-2])))
        s, f = shape[-2], shape[-1]
        flat = h_ls.reshape(n, s, f)
        return torch.stack([flat.real, flat.imag], dim=1), shape

    def _from_grid(self, refined, shape):
        h = torch.complex(refined[:, 0], refined[:, 1])
        return h.reshape(shape)

    def forward(self, y, no):
        h_ls, err_var = self.ls_estimator(y, no)
        grid, shape = self._to_grid(h_ls)
        delta = self.refine(grid)
        h_hat = self._from_grid(grid + delta, shape)
        return h_hat, err_var


class CNNChannelEstimator(_BaseNeuralEstimator):
    def __init__(self, resource_grid, pusch_config, hidden=32, num_blocks=3):
        super().__init__(resource_grid, pusch_config)
        layers = [nn.Conv2d(2, hidden, 3, padding=1), nn.ReLU()]
        for _ in range(num_blocks):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU()]
        layers += [nn.Conv2d(hidden, 2, 3, padding=1)]
        self.refine = nn.Sequential(*layers)


class TransformerChannelEstimator(_BaseNeuralEstimator):
    def __init__(self, resource_grid, pusch_config, d_model=32, nhead=4, num_layers=2):
        super().__init__(resource_grid, pusch_config)
        self.in_proj = nn.Linear(2, d_model)
        self.out_proj = nn.Linear(d_model, 2)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def refine(self, grid):
        n, c, s, f = grid.shape
        seq = grid.permute(0, 2, 3, 1).reshape(n * s, f, c)
        x = self.in_proj(seq)
        x = self.encoder(x)
        x = self.out_proj(x)
        return x.reshape(n, s, f, c).permute(0, 3, 1, 2)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
