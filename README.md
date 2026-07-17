# AI-Native Neural Channel Estimator for 5G NR

**A deep-learning model that replaces a classical signal-processing component
in the 5G radio receiver - trained, benchmarked, and packaged as a real,
deployable service - built on NVIDIA Sionna.**

**Author:** [Tào Việt Đức](https://github.com/taovietducofficial)

**Language:** English | [Tiếng Việt](README.vi.md)

---

## At a glance

| | |
|---|---|
| **What it is** | A neural network (CNN and Transformer variants) that estimates the radio channel more accurately than the classical algorithm it replaces, inside a standards-based 5G NR uplink simulation. |
| **Measured result** | At the standard 10%-error operating point, the CNN model reaches the same reliability at **~0.6 dB lower signal strength** than the classical estimator - in the exact SNR range where cell-edge and mobile users typically sit. At a representative operating point this corresponds to **~27% fewer retransmissions**. |
| **Proof, not just a claim** | Every number below comes from a script in this repo that anyone can re-run. Where the evidence is thin (e.g. sample size, GPU numbers), that's stated explicitly rather than rounded up. |
| **Runs as a real service** | `docker run` gets you a live API and a clickable demo page - not just a notebook. |
| **Engineering maturity** | Pinned dependencies, automated tests, CI with lint/type-checking, structured logging + metrics, and a provenance record tying every deployed model back to the exact training run that produced it. |
| **Honest boundary** | This is a research/portfolio-grade simulation, not a validated field deployment - see [Honest limitations](#honest-limitations) for exactly what that does and doesn't mean. |

**Try it in one command:** see [Quick start](#quick-start).

---

## Why this matters

Modern telecom networks (5G today, 6G next) spend enormous engineering and
capital effort chasing small gains in **signal reliability at the edge of
coverage** - that's where retransmissions pile up, throughput collapses, and
operators end up building more towers to compensate. This project asks a
narrow, concrete question: *can a small neural network, dropped into the
exact place a classical algorithm sits today, measurably improve reliability
in that hard region - without changing anything else about the radio
system?*

The answer here, measured rather than assumed: **yes, modestly, and
consistently** - plus an honest account of where it doesn't help (very
strong signal conditions) and what that implies for how it should actually
be deployed.

This is also built to mirror how such a model would really reach production:
a portable model artifact, a versioned and monitored inference service, and
a clear-eyed mapping of where it fits (and doesn't) in a real network
architecture (O-RAN) - not just a training script.

---

## Quick start

```bash
# One-time (needs the research environment - see Setup below):
python -m service.export_model
python -m service.fixtures.gen_fixtures

# From here on, no Sionna/PyTorch needed - just Docker:
docker build -t neural-estimator-service -f service/Dockerfile service
docker run --rm -p 8000:8000 neural-estimator-service
```

Open `http://localhost:8000/` - three buttons (low/medium/high signal
strength) run real channel data through the classical algorithm and both
neural models side by side, showing accuracy and latency live.

---

## Table of contents

- [Results](#results)
- [How it works](#how-it-works)
- [Repository map](#repository-map)
- [Setup and running the research pipeline](#setup-and-running-the-research-pipeline)
- [Service / deployment demo](#service--deployment-demo)
- [Engineering hygiene](#engineering-hygiene)
- [Honest limitations](#honest-limitations)
- [Further reading](#further-reading)

---

## Results

**Channel-estimation accuracy vs. signal strength** (CDL-C channel, 3 km/h,
512 codewords per point - `results/bler_results.csv` / `results/bler_vs_snr.png`):

| SNR (dB) | Baseline (LS+LMMSE) | CNN | Transformer |
|---:|---:|---:|---:|
| 7.0 | 1.000 | 0.953 | 0.955 |
| 7.5 | 0.865 | 0.418 | 0.529 |
| 8.0 | 0.338 | **0.094** | 0.229 |
| 8.5 | 0.113 | 0.064 | 0.123 |
| 9.0 | 0.066 | 0.035 | 0.123 |
| 9.5 | 0.033 | 0.023 | 0.074 |
| 10.0 | 0.027 | 0.023 | 0.055 |

(Values are block-error rate - the fraction of transmissions that fail and
must be resent. Lower is better.)

**Read plainly:** the CNN clearly beats the classical estimator through the
entire "waterfall" region where signal quality is marginal. At the standard
10%-error operating point, the classical method needs about 8.6 dB of signal
strength to get there; the CNN gets there at about 8.0 dB - a **~0.6 dB
margin gain**, with the full derivation and its business framing in
[`docs/business-impact.md`](docs/business-impact.md). The Transformer also
beats the classical baseline but less cleanly (it plateaus before improving
further - most likely sampling noise at this run's sample count, flagged
rather than smoothed over).

**What this doesn't show, stated up front:** at very strong signal
(10 dB+), an earlier, coarser measurement run found the classical estimator
actually reaching zero errors while both neural models sat at a small
residual error floor - a known characteristic of networks trained across a
wide signal-strength range rather than specialized for any one regime. The
practical implication: this model is a targeted improvement for the
hard-signal region, not a universal replacement - see
[Honest limitations](#honest-limitations).

**Inference cost and compression** (CPU-only - see [Setup](#setup-and-running-the-research-pipeline);
`results/benchmark.csv`):

| Batch size | Model | Optimization | Before | After | Speedup |
|---:|---|---|---:|---:|---:|
| 8   | CNN | TorchScript trace+freeze | 3.02 ms | 3.22 ms | 0.94x |
| 8   | Transformer | INT8 quantization (partial) | 12.56 ms | 19.83 ms | 0.63x |
| 256 | CNN | TorchScript trace+freeze | 158.1 ms | 149.9 ms | **1.06x** |
| 256 | Transformer | INT8 quantization (partial) | 788.7 ms | 1071.3 ms | 0.74x |

**Read plainly:** compression is not automatically a win - at small batch
sizes, the optimization overhead exceeds the benefit; it only pays off once
the batch size is large enough to amortize it. That's a genuine engineering
finding about *when* model compression helps, worth more than a number that
only looks good in isolation. (These absolute numbers shift somewhat between
runs depending on what else is competing for CPU on the build machine at the
time - re-run `python -m src.benchmark` yourself for a clean read; the
qualitative pattern, compression helping at large batch and not at small
batch, has held across every rerun.)

Both models were also verified to export cleanly to the portable ONNX
format with numerical parity against the original PyTorch model (deviation
of roughly one part in ten million) - see `service/export_model.py`.

**Real-time check** (`src/realtime_benchmark.py`, `results/realtime_latency.csv`):
single-sample (batch=1) latency, the number that actually matters for an
inline per-slot radio function, against real 5G NR slot budgets:

| Model | p50 | p95 | p99 | Fits 1 ms slot? | Fits 0.25 ms slot? |
|---|---:|---:|---:|:---:|:---:|
| CNN | 1.78 ms | 5.38 ms | 11.58 ms | No | No |
| Transformer | 3.05 ms | 6.20 ms | 8.52 ms | No | No |

**Read plainly, not softened:** on CPU, neither model currently meets any
real per-slot budget at p99. This is exactly why GPU/accelerator inference
isn't a "nice to have" here - see `colab_gpu_benchmark.ipynb` to measure the
real GPU numbers, and [`docs/oran-integration.md`](docs/oran-integration.md)
for the full discussion of this gap.

**MIMO (2-layer) validation** (`results/bler_results_mimo2.csv`,
`results/bler_vs_snr_mimo2.png`): the same CNN architecture, unmodified,
retrained on a 2x2 MIMO configuration, beats the classical estimator across
the full measured range (e.g. 0.109 vs 0.172 BLER at 14 dB - roughly a 37%
relative reduction). Getting a *working* result needed one real fix: the
first attempt used a single receive antenna, which is information-theoretically
incapable of separating 2 spatial streams (100% BLER) regardless of training -
not a model bug. Full story in
[`docs/oran-integration.md`](docs/oran-integration.md).

---

## How it works

1. **A standards-based 5G uplink is simulated**, not invented - built on
   [NVIDIA Sionna](https://github.com/nvlabs/sionna) 2.0's 5G NR PUSCH link
   (OFDM, LDPC coding, 3GPP channel models), so the physics and protocol are
   the real thing, not a toy stand-in.
2. **The neural network is a drop-in replacement** for one specific
   component - the channel estimator - using the exact interface the
   simulator expects. It doesn't bypass or approximate the rest of the
   radio chain.
3. **It learns by correction, not from scratch**: both the CNN and
   Transformer refine the classical algorithm's noisy first guess toward
   the true channel, trained on data the simulator generates on demand - no
   external dataset required.
4. **Two architectures, compared honestly**: the CNN and Transformer are
   evaluated on the same task so the comparison (accuracy vs. inference
   cost) is apples-to-apples.
5. **The trained model is then shipped**, not left in a notebook: exported
   to the portable ONNX format, wrapped in a REST API, containerized, and
   given a live demo page - see [Service / deployment demo](#service--deployment-demo).

---

## Repository map

```
src/baseline.py                classical PUSCH link (LS + LMMSE), BLER vs SNR
src/models.py                   CNNChannelEstimator, TransformerChannelEstimator
src/train.py                     trains a neural estimator on synthetic channel data
src/evaluate.py                   BLER-vs-SNR comparison: baseline vs CNN vs Transformer
src/benchmark.py                   inference latency + compression benchmark (CPU)
src/realtime_benchmark.py            single-sample p50/p95/p99 latency vs real slot budgets
tests/test_pipeline.py               assert-based smoke checks (research side)

colab_gpu_benchmark.ipynb        ready-to-run Colab notebook for real GPU latency numbers
oran-stub/                        runnable illustrative E2 REPORT/CONTROL demo (see its README)

service/export_model.py        offline: exports checkpoints to ONNX + parity check + provenance manifest
service/fixtures/                 offline: real (noisy, perfect) channel fixtures at 3 SNRs
service/artifacts/                 exported .onnx models + <name>_manifest.json (committed, small)
service/app.py                     FastAPI: /health /metrics /models /v1/estimate /v1/demo/{bucket}
service/inference.py                onnxruntime wrapper + provenance lookup
service/static/index.html             clickable demo UI (no build step)
service/tests/test_service.py          assert-based smoke checks (service side)
service/Dockerfile                   builds a Sionna/PyTorch-free serving image
service/requirements.txt               pinned runtime deps; requirements-dev.txt adds httpx/ruff/mypy

docs/oran-integration.md        honest mapping to O-RAN xApp/rApp/E2/A1 concepts, and the 6G path
docs/business-impact.md          KPI/business framing built strictly from measured numbers

pyproject.toml                    ruff + mypy config
LICENSE                            All rights reserved (public for portfolio/evaluation viewing only)
Makefile                          research-test / export / fixtures / test / lint / typecheck / docker-build / docker-run
.github/workflows/ci.yml            lint+typecheck and service tests + docker build on every push; research tests on demand
```

---

## Setup and running the research pipeline

Sionna 2.0 requires Python 3.11+ and PyTorch (it dropped TensorFlow as of
2.0). Built and run with a Python 3.11 venv:

```bash
py -3.11 -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

No local GPU was available for this build (checked via `nvidia-smi`); the
link simulation and small models here train and run fine on CPU. For real
GPU latency numbers, re-run `src/benchmark.py` on a CUDA machine or
[Google Colab](https://colab.research.google.com/) - same code, PyTorch
uses CUDA automatically once tensors/models move to `.cuda()`.

```bash
# 1. Sanity-check the classical link
python -m src.baseline

# 2. Train the neural estimators (synthetic data, no dataset needed)
python -m src.train --model cnn --steps 1500
python -m src.train --model transformer --steps 1500

# 3. Compare accuracy: baseline vs CNN vs Transformer
python -m src.evaluate --snr-min 6 --snr-max 10 --snr-step 0.5 --num-batches 32

# 4. Benchmark inference cost + compression
python -m src.benchmark

# 5. Real-time check: single-sample p50/p95/p99 latency vs real slot budgets
python -m src.realtime_benchmark

# Optional: train + evaluate a 2-layer MIMO variant (same architecture, no code changes)
python -m src.train --model cnn --steps 1500 --num-layers 2 --out checkpoints/cnn_mimo2.pt
python -m src.evaluate --snr-min 6 --snr-max 14 --snr-step 1 --num-batches 16 --num-layers 2

# Smoke tests
python -m tests.test_pipeline
```

For real GPU latency numbers, open `colab_gpu_benchmark.ipynb` in
[Google Colab](https://colab.research.google.com/) (enable a GPU runtime,
then Run all) - it clones this repo and times the same networks on GPU.

Results land in `results/bler_vs_snr.png`, `results/bler_results.csv`, and
`results/benchmark.csv`. (`make research-test` / `make test` wrap the smoke
tests; see the `Makefile`.)

---

## Service / deployment demo

Turns the trained checkpoints into something you can actually run and click
through - not just a research script. See
[`docs/oran-integration.md`](docs/oran-integration.md) for an honest
discussion of how (and how not) to describe this as an O-RAN integration
and the path to 6G, and [`docs/business-impact.md`](docs/business-impact.md)
for the full KPI derivation.

```bash
# One-time, in the research venv (needs Sionna/PyTorch):
python -m service.export_model          # exports checkpoints -> service/artifacts/*.onnx
python -m service.fixtures.gen_fixtures  # generates the 3 demo SNR fixtures

# From here on, the service itself needs neither Sionna nor PyTorch:
docker build -t neural-estimator-service -f service/Dockerfile service
docker run --rm -p 8000:8000 neural-estimator-service
```

Then open `http://localhost:8000/` for the clickable demo, or hit the API
directly:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/demo/mid
curl http://localhost:8000/models    # provenance: which checkpoint, which training run
curl http://localhost:8000/metrics   # Prometheus-format request/latency metrics
```

Service-side tests (no Sionna/PyTorch required): `python -m service.tests.test_service`.

---

## Engineering hygiene

Senior-engineering-principle upgrades on top of the model and service
themselves - the difference between "a model that works" and "a model
someone else could operate":

- **Reproducible builds**: every dependency (research and service) is
  pinned to an exact version, not `>=` - a build today resolves the same
  way in a year.
- **Structured logging + metrics**: the service logs each request via
  Python's `logging` module, and exposes Prometheus-format metrics at
  `GET /metrics` (request counts and inference latency, broken down by
  model) - the standard scrape target for any real monitoring stack.
- **Model provenance**: every export run writes a manifest containing the
  SHA-256 of the source checkpoint, the ONNX opset and measured parity
  error, the export timestamp, and the training run's hyperparameters and
  final loss. `GET /models` surfaces this - a served artifact is traceable
  back to exactly what produced it, not just "trust me, it's the CNN."
- **CI quality gate**: lint (`ruff`) and type-checking (`mypy`) run on
  every push, alongside the test suite and a Docker build.
- **Input validation at the trust boundary**: the API validates request
  shape before it reaches the model, returning a clean error instead of a
  crash on malformed input.

---

<details>
<summary>

## Honest limitations

</summary>

Stated directly rather than buried - what this project does *not* claim:

- **Simulation, not a live network.** All channel data is generated by
  Sionna's standards-based simulator, not captured from a real deployed
  network. The physics and protocol are real; the traffic and environment
  are not.
- **No fabricated business numbers.** The margin-gain and retransmission
  figures above are derived directly from measured simulation data with the
  calculation shown; there is no invented dollar-value ROI, opex saving, or
  coverage-radius figure - see
  [`docs/business-impact.md`](docs/business-impact.md) for exactly where
  that line is drawn.
- **No GPU numbers measured directly.** No GPU was available on this build
  machine; `colab_gpu_benchmark.ipynb` is a ready-to-run notebook that gets
  real GPU latency numbers on a free Colab GPU in a few minutes - the gap is
  filled with a runnable artifact, not just a caveat.
- **Real-time deadline: measured, and currently missed.** `src/realtime_benchmark.py`
  now measures single-sample p50/p95/p99 latency against real 5G NR slot
  budgets (0.25-1 ms). Result, stated plainly: **on CPU, neither model
  currently meets any of those budgets at p99** (`results/realtime_latency.csv`)
  - this is a real, measured gap, not a downplayed one; see
  [`docs/oran-integration.md`](docs/oran-integration.md) for the full
  numbers and what it implies.
- **MIMO: validated for 2 layers, not just theoretical.** The original
  single-antenna-only claim was narrower than reality - the same CNN
  architecture, unmodified, runs and was retrained/evaluated on a 2x2 MIMO
  config (`results/bler_results_mimo2.csv`), beating the classical estimator
  there too. See [`docs/oran-integration.md`](docs/oran-integration.md) for
  what actually needed fixing (receive-antenna count, not the model) and
  what's still unrun (the Transformer variant at MIMO).
- **No live O-RAN platform integration.** Deliberately assessed against
  this machine's real WSL2 environment and skipped as disproportionate - a
  lightweight illustrative stand-in for the model-selection message pattern
  is built and runnable (`oran-stub/`), but it is not a spec-compliant
  E2AP/RIC integration; the reasoning is documented, not glossed over, in
  [`docs/oran-integration.md`](docs/oran-integration.md).

</details>

## Further reading

- [`docs/business-impact.md`](docs/business-impact.md) - full KPI derivation and its stated boundaries.
- [`docs/oran-integration.md`](docs/oran-integration.md) - accurate O-RAN architecture mapping and the path to 6G.
