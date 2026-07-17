# Business impact: what the measured numbers actually support

Every number below is read directly from this repo's own output
(`results/bler_results.csv`, `results/benchmark.csv`) — nothing here is a
projected or invented figure. Where a real business claim would need data
this project doesn't have (live network traffic, real HARQ combining, real
opex figures), that boundary is stated explicitly rather than papered over.

## 1. SNR margin gain at a standard operating point

Re-running `src/evaluate.py` with a finer sweep around the waterfall region
(`--snr-step 0.5 --num-batches 32`, 512 codewords per point) gives:

| SNR (dB) | Baseline (LS+LMMSE) | CNN | Transformer |
|---:|---:|---:|---:|
| 7.0 | 1.000 | 0.953 | 0.955 |
| 7.5 | 0.865 | 0.418 | 0.529 |
| 8.0 | 0.338 | **0.094** | 0.229 |
| 8.5 | 0.113 | 0.064 | 0.123 |
| 9.0 | 0.066 | 0.035 | 0.123 |
| 9.5 | 0.033 | 0.023 | 0.074 |
| 10.0 | 0.027 | 0.023 | 0.055 |

At the standard **10% BLER operating point** (a common reference target in
link-level 5G evaluations): the classical estimator reaches it at
**~8.6 dB**, the CNN estimator reaches it at **~8.0 dB** — a **~0.6 dB SNR
margin gain**. That's a modest, defensible number, not a dramatic one; it's
reported as measured rather than rounded up. (The Transformer also beats the
baseline in this region but its curve is noisier at this sample count — no
precise crossing point is claimed for it.)

**What a 0.6 dB margin gain would mean in a real deployment:** in a real
link budget, an SNR margin gain of this size typically translates into
either a modest coverage-radius increase or a modest capacity/density gain
at the same radius — the exact conversion depends on the propagation model
and isn't computed here (no real path-loss/deployment model in this repo).
Stated honestly: **this project measures the SNR-domain gain; it does not
compute a coverage-radius or site-count number**, because that requires
real network geometry this project doesn't have.

## 2. BLER → retransmissions (first-order estimate, explicitly not HARQ-aware)

At the 8 dB operating point specifically: baseline BLER 0.338 vs CNN BLER
0.094. Using the simplified expected-transmissions model
`E[transmissions] ≈ 1 / (1 - BLER)` (independent retries, no combining gain):

- Baseline: 1 / (1 - 0.338) ≈ **1.51** transmissions per delivered block.
- CNN: 1 / (1 - 0.094) ≈ **1.10** transmissions per delivered block.
- **≈ 27% reduction in expected radio-resource consumption** at this
  specific operating point.

**Caveat stated directly:** real HARQ soft-combines retransmissions (each
retry raises effective SNR at the receiver), so real BLER-vs-retransmission
behavior is better than this independent-retry model suggests for *both*
methods — this number is a first-order estimate of the gap between them,
not a validated throughput figure. It's also worth noting that a second,
earlier run of the same evaluation (coarser SNR grid, different random
seed/sample count) produced a similar but not identical ratio (0.359 vs
0.117 at 8 dB) — the ~25-30% range is consistent across reruns, but the
exact number moves with Monte Carlo sample size, which is itself an honest
data point (see `README.md`'s discussion of statistical noise).

## 3. Inference cost (from `results/benchmark.csv` and the live service)

- CPU, batch 256: CNN refiner ≈ 0.43 ms/sample after TorchScript
  fusion (1.07x vs eager fp32); Transformer's INT8-quantized projection
  layers show ~no net change (the untouched encoder dominates runtime).
- Live ONNX-served CNN model (measured through the actual FastAPI/Docker
  service, not a research script): sub-2ms per call at batch size 8 on
  CPU inside the container.
- **What this does *not* show:** a real inline PHY function is judged
  against a **single-slot p99 latency** (0.25–1ms, no batching) — this
  project measures batched throughput, not that number. See
  `docs/oran-integration.md` for the explicit gap.

## 4. What's deliberately not claimed

- No fabricated dollar-value ROI, cost-per-GB, or opex savings figure.
- No real deployed-network traffic, no real HARQ-combining model.
- No validated coverage-radius or site-count reduction number — only the
  SNR-domain gain that would feed into such a calculation if a real
  propagation model were applied.

The value of this document is the same as the value of the project: a
concrete, checkable set of numbers with their limits stated up front, not a
polished story that doesn't survive a follow-up question.
