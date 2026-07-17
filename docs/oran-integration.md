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

**Decision: skip both.** This repo ships a clean, honestly-labeled analogy
instead. Wiring the refiner into FlexRIC's E2 Service Model, or building a
toy Non-RT-RIC-style periodic model-selection loop around
`src/evaluate.py`, are both reasonable next projects for someone who wants to
go deeper — listed below as explicit future work, not implied to already
exist.

## The "real-time" gap, stated plainly

`src/benchmark.py`'s numbers are **batched CPU throughput** (milliseconds per
batch of 8 or 256 samples), not the **single-slot p99 latency** a real inline
PHY function would actually be judged on (a hard deadline of 0.25–1ms per
slot, no averaging across a batch). This project does not claim to have met
a real-time deadline — it demonstrates the model, the serving architecture,
and honest instrumentation of where the numbers currently stand.

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

## Future extensions (not built, explicitly out of scope for this repo)

- Real OSC Near-RT RIC or FlexRIC integration.
- A periodic "model-selection rApp" simulation (re-evaluate BLER on a
  schedule, toggle which model a cell uses via an A1-style policy).
- Single-sample (batch=1) p99 latency measurement against an actual
  slot-duration budget.
- GPU benchmarking (no local GPU available; see main README).
