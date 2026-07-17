import argparse
import os
import time

import numpy as np
import torch

from src.benchmark import grid_shape, load_refiner

SLOT_BUDGETS_MS = {
    "15 kHz SCS (1 ms slot)": 1.0,
    "30 kHz SCS (0.5 ms slot)": 0.5,
    "60 kHz SCS (0.25 ms slot)": 0.25,
}


def measure_single_sample_latencies(model, x, repeats=500, warmup=50):
    model.eval()
    latencies_ms = np.empty(repeats)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        for i in range(repeats):
            t0 = time.perf_counter()
            model(x)
            latencies_ms[i] = (time.perf_counter() - t0) * 1000
    return latencies_ms


def main(checkpoints_dir="checkpoints", repeats=500):
    x = torch.randn(*grid_shape(1))
    rows = []

    for name in ["cnn", "transformer"]:
        ckpt = os.path.join(checkpoints_dir, f"{name}.pt")
        if not os.path.exists(ckpt):
            print(f"skip {name}: no checkpoint at {ckpt}")
            continue
        model = load_refiner(name, checkpoints_dir)
        latencies = measure_single_sample_latencies(model, x, repeats=repeats)
        p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
        rows.append((name, p50, p95, p99))

    print(f"single-sample (batch=1) inference latency, CPU, n={repeats} runs\n")
    print(f"{'model':<14}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}")
    for name, p50, p95, p99 in rows:
        print(f"{name:<14}{p50:>10.4f}{p95:>10.4f}{p99:>10.4f}")

    print("\nagainst real per-slot processing budgets (a live inline PHY")
    print("function would need to finish well within one slot):\n")
    print(f"{'model':<14}{'budget':<28}{'p99 ms':>10}{'fits?':>8}")
    for name, p50, p95, p99 in rows:
        for budget_name, budget_ms in SLOT_BUDGETS_MS.items():
            fits = "yes" if p99 < budget_ms else "NO"
            print(f"{name:<14}{budget_name:<28}{p99:>10.4f}{fits:>8}")

    os.makedirs("results", exist_ok=True)
    with open("results/realtime_latency.csv", "w") as f:
        f.write("model,p50_ms,p95_ms,p99_ms\n")
        for name, p50, p95, p99 in rows:
            f.write(f"{name},{p50:.4f},{p95:.4f},{p99:.4f}\n")
    print("\nsaved results/realtime_latency.csv")
    print("NOTE: CPU only, single process, no real-time OS scheduling guarantees.")
    print("A live inline PHY function would also need a hard real-time runtime")
    print("(not a general-purpose Python process) to guarantee these percentiles.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    parser.add_argument("--repeats", type=int, default=500)
    args = parser.parse_args()
    main(args.checkpoints_dir, args.repeats)
