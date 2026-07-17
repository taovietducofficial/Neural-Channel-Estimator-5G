import argparse
import io
import os
import time

import torch
import torch.nn as nn

from src.models import CNNChannelEstimator, TransformerChannelEstimator
from sionna.phy.nr import PUSCHConfig, PUSCHTransmitter

GRID_SHAPE = (8, 2, 14, 48)


def grid_shape(batch_size):
    return (batch_size, 2, 14, 48)


class TransformerRefiner(nn.Module):
    def __init__(self, est: TransformerChannelEstimator):
        super().__init__()
        self.in_proj = est.in_proj
        self.encoder = est.encoder
        self.out_proj = est.out_proj

    def forward(self, grid):
        n, c, s, f = grid.shape
        seq = grid.permute(0, 2, 3, 1).reshape(n * s, f, c)
        x = self.out_proj(self.encoder(self.in_proj(seq)))
        return x.reshape(n, s, f, c).permute(0, 3, 1, 2)


def state_dict_size_kb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / 1024


def time_forward(model, x, repeats=100, warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        elapsed = time.perf_counter() - start
    return elapsed / repeats * 1000


def load_refiner(model_name, checkpoints_dir="checkpoints"):
    pusch_config = PUSCHConfig()
    transmitter = PUSCHTransmitter(pusch_config)
    cls = {"cnn": CNNChannelEstimator, "transformer": TransformerChannelEstimator}[model_name]
    full = cls(transmitter.resource_grid, pusch_config)
    ckpt = os.path.join(checkpoints_dir, f"{model_name}.pt")
    full.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    full.eval()
    if model_name == "transformer":
        return TransformerRefiner(full)
    return full.refine


def main(checkpoints_dir="checkpoints", batch_size=8):
    x = torch.randn(*grid_shape(batch_size))
    rows = []

    for name in ["cnn", "transformer"]:
        ckpt = os.path.join(checkpoints_dir, f"{name}.pt")
        if not os.path.exists(ckpt):
            print(f"skip {name}: no checkpoint at {ckpt}")
            continue
        model = load_refiner(name, checkpoints_dir)
        fp32_ms = time_forward(model, x)
        fp32_kb = state_dict_size_kb(model)

        if name == "transformer":
            qconfig_spec = {
                "in_proj": torch.quantization.default_dynamic_qconfig,
                "out_proj": torch.quantization.default_dynamic_qconfig,
            }
            q_model = torch.quantization.quantize_dynamic(model, qconfig_spec, dtype=torch.qint8)
            opt_ms = time_forward(q_model, x)
            opt_kb = state_dict_size_kb(q_model)
            technique = "dynamic INT8 quant (proj layers only)"
        else:
            traced = torch.jit.trace(model, x)
            traced = torch.jit.freeze(traced)
            opt_ms = time_forward(traced, x)
            opt_kb = fp32_kb
            technique = "TorchScript trace + freeze (op fusion)"

        rows.append((name, technique, fp32_ms, opt_ms, fp32_kb, opt_kb))

    print(f"{'model':<12}{'technique':<30}{'fp32 ms':>10}{'opt ms':>10}{'speedup':>10}{'fp32 KB':>10}{'opt KB':>10}")
    for name, technique, fp32_ms, opt_ms, fp32_kb, opt_kb in rows:
        speedup = fp32_ms / opt_ms if opt_ms > 0 else float("nan")
        print(f"{name:<12}{technique:<30}{fp32_ms:>10.4f}{opt_ms:>10.4f}{speedup:>9.2f}x{fp32_kb:>10.1f}{opt_kb:>10.1f}")

    os.makedirs("results", exist_ok=True)
    with open("results/benchmark.csv", "w") as f:
        f.write("model,technique,fp32_ms,opt_ms,speedup,fp32_kb,opt_kb\n")
        for name, technique, fp32_ms, opt_ms, fp32_kb, opt_kb in rows:
            speedup = fp32_ms / opt_ms if opt_ms > 0 else float("nan")
            f.write(f"{name},{technique},{fp32_ms:.4f},{opt_ms:.4f},{speedup:.2f},{fp32_kb:.1f},{opt_kb:.1f}\n")
    print("saved results/benchmark.csv")
    print("\nNOTE: benchmarked on CPU (no local GPU detected). Re-run on a CUDA")
    print("machine / Colab for GPU latency numbers -- same script, torch will")
    print("pick up CUDA automatically if you move the model/input to .cuda().")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    main(args.checkpoints_dir, args.batch_size)
