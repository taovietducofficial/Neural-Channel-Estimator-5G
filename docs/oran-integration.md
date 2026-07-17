# How this maps to a real O-RAN deployment (and where it doesn't)

This project is often easiest to describe in an interview by reaching for O-RAN
vocabulary ("it's like an xApp"). That description is inaccurate, and it's
worth being precise about why, since misusing these terms in front of someone
who works on real RAN is a fast way to lose credibility.

## The relevant O-RAN architecture, briefly

- **Non-RT RIC** (inside the SMO, control loop >1s) hosts **rApps**. It talks
  to the Near-RT RIC over the **A1 interface** — policies, enrichment
  information, ML model metadata/updates. Timescale: minutes to hours.
- **Near-RT RIC** (control loop 10ms–1s) hosts **xApps**. It talks to E2 Nodes
  (CU-CP, CU-UP, DU) over the **E2 interface** (E2AP protocol + Service
  Models — E2SM-KPM for KPI monitoring/reporting, E2SM-RC for RAN control
  actions). Timescale: tens of milliseconds and up.
- **Channel estimation is a PHY-layer (L1) function.** It runs **every slot**
  — sub-millisecond, e.g. every 1ms/0.5ms/0.25ms depending on numerology —
  inside the O-DU's baseband processing. That's **faster than the Near-RT
  RIC's own 10ms floor.** E2 and A1 are built to carry KPI-level metrics and
  control policies, not a raw per-slot channel-estimate tensor.

## Where this project's service actually sits

| Concept in this repo | Closest real O-RAN concept | Why it's not exact |
|---|---|---|
| `POST /v1/estimate` (the neural refiner itself) | An **AI-native inline PHY function inside the O-DU's L1 pipeline** | Not an xApp: it runs at per-slot cadence, far below the E2/Near-RT RIC timescale. This is an emerging O-RAN/3GPP discussion area (AI/ML acceleration at the DU/RU split), not a shipped interface. |
| Periodically re-running `src/evaluate.py`'s BLER sweep and deciding "keep CNN for these cells" / "roll back to classical LS" | An **rApp** (or a coarse xApp) | This decision genuinely operates at a minutes-to-hours cadence — the right timescale for A1-style policy push from the Non-RT RIC. Not built in this repo; see "Future extensions" below. |
| ONNX artifact + FastAPI service | A stand-in for how a **vendor's AI/ML pipeline might package a model for deployment** into O-DU software | It is a clean architecture demo, not integrated with any real E2/A1 stack, RIC platform, or vendor SDK. |

## Why no real RIC platform is installed here

Standing up an actual Near-RT RIC — the O-RAN Software Community (OSC)
`ric-plt` stack (e2mgr, e2term, submgr, rtmgr, appmgr, a1mediator, dbaas, ...)
is a multi-microservice Kubernetes/Helm deployment, normally paired with a
real or simulated DU and E2 agent. Even for people who've done it before,
that's realistically multi-day systems-integration work, and it would be
orthogonal to the AI/PHY skill this project is meant to demonstrate. A
lighter option exists — OpenAirInterface's **FlexRIC** (a research-oriented
Near-RT RIC + E2 agent in C) — but exposing channel-estimation-level data
through it is still nontrivial glue, not a same-day task either.

**Decision: skip real RIC infrastructure, but build the illustrative pattern.**
A real OSC/FlexRIC install was assessed against this machine's actual WSL2
Ubuntu environment and found technically feasible but disproportionate — a
from-source FlexRIC build is realistically 30-60+ minutes of fragile
dependency work (cmake, protobuf, SCTP, SWIG) for a result that would only
prove connectivity, not improve the model. Instead, `oran-stub/` is a small,
genuinely-running two-process demo (`ric_server.py` / `e2_node_client.py`,
real TCP sockets, real measured BLER numbers from `results/bler_results.csv`)
of exactly the pattern described above: periodic REPORT → policy decision →
CONTROL. It uses a simplified JSON protocol, not real E2AP/ASN.1/SCTP — see
`oran-stub/README.md` for the explicit boundary. Wiring this same policy
logic into a real FlexRIC E2 Service Model remains a reasonable next project
for someone who wants to go deeper.

## The "real-time" gap — now actually measured

`src/benchmark.py`'s numbers are **batched CPU throughput** (milliseconds per
batch of 8 or 256 samples) — not the single-slot latency a real inline PHY
function would be judged on. `src/realtime_benchmark.py` closes that
measurement gap directly: it times single-sample (batch=1) inference and
reports p50/p95/p99 latency against the real per-slot budgets 5G NR actually
uses (1 ms / 0.5 ms / 0.25 ms for 15/30/60 kHz subcarrier spacing).

**Measured result (CPU, `results/realtime_latency.csv`):** both models miss
every one of those budgets at p99 (CNN ~11.6 ms, Transformer ~8.5 ms, all
roughly 10-50x over budget). Stated plainly rather than downplayed: **on
CPU, neither model is currently real-time-capable for an inline per-slot
PHY function.** This is the honest reason GPU or dedicated accelerator
inference (see `colab_gpu_benchmark.ipynb`) isn't just a "nice to have" for
this use case — it's the difference between a research artifact and a
deployable one. It also reflects that this is a general-purpose Python
process, not a hard real-time runtime; even with faster hardware, meeting
these percentiles in production would need a real-time-scheduled inference
path, not just a faster forward pass.

## Path to 6G: why this architecture doesn't need to be rebuilt

The service (`service/`) is deliberately decoupled from Sionna/PyTorch: it
loads a portable ONNX artifact and operates on an abstract
`(batch, 2, ofdm_symbols, subcarriers)` tensor, with zero assumptions baked
in about carrier frequency, channel model, or generation. Concretely:

- `src/train.py`'s training loop already takes an arbitrary `channel_model`
  (via `build_training_link`) — it isn't hardcoded to CDL-C. A model trained
  against a 6G-flavored channel (higher `carrier_frequency`, e.g. sub-THz;
  or a RIS-assisted scenario, which Sionna has an active research module for
  via `sionna-rt`) is trained with the *same* script.
- Deploying that new model is: retrain → re-run `service/export_model.py` →
  drop the new `.onnx` file into `service/artifacts/`. The API, Docker image,
  and tests are untouched.
- This is the same reason the "AI-native inline PHY function" framing above
  matters: it's the direction 3GPP/O-RAN discussions on deeper AI/ML
  integration into L1 are already heading, not a label invented for this
  project.

On "7G": as of this writing there is no standardized "7G" — 6G itself is
still being standardized, with commercial deployment not expected before
~2030. The honest claim here isn't "ready for 7G"; it's that this
architecture doesn't couple itself to any specific generation's assumptions
beyond the minimum needed, so whatever comes after 6G, the serving layer
doesn't need a rewrite — only a new model artifact.

## MIMO: validated, not just theoretically possible

The single-antenna limitation noted elsewhere in this repo turned out to be
narrower than first assumed. `_BaseNeuralEstimator`'s grid-flattening
(`src/models.py`) already generalizes to extra tensor dimensions — a
2-layer (2x2) MIMO PUSCH config runs through the exact same CNN/Transformer
code with zero architecture changes. That was verified directly: a first
attempt (UE with 2 TX antennas, gNB with only 1 RX antenna) ran without
crashing but produced 100% BLER, because separating 2 spatial streams needs
at least 2 receive antennas — a MIMO information-theory constraint, not a
bug in the neural model. Fixing the receive-antenna count
(`num_layers`-many `bs_array` antennas in `src/baseline.py`/`train.py`/
`evaluate.py`) and retraining (`python -m src.train --model cnn --num-layers 2`)
produced a real, working 2x2 result
(`results/bler_results_mimo2.csv` / `results/bler_vs_snr_mimo2.png`): the
CNN estimator beats the classical one across the whole measured range (e.g.
0.109 vs 0.172 BLER at 14 dB — roughly a 37% relative reduction), using
the identical architecture trained on the original single-antenna config
elsewhere in this repo. The Transformer variant wasn't retrained for MIMO
in this pass; extending the same `--num-layers` flag to it is a direct
repeat of the same steps, not new engineering.

## Future extensions (not built, explicitly out of scope for this repo)

- Real OSC Near-RT RIC or FlexRIC integration (a lightweight illustrative
  stand-in for the model-selection message pattern is built — see
  `oran-stub/` — but it is not a real E2AP/SCTP/RIC-platform integration).
- Transformer MIMO training/evaluation (same `--num-layers` mechanism as
  CNN, just not run in this pass).
- GPU benchmarking beyond what `colab_gpu_benchmark.ipynb` measures when
  run (no local GPU available to run it here directly; see main README).
