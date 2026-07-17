import logging
import os
import time

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from service import inference
from service.schemas import DemoResponse, EstimateRequest, EstimateResponse, MethodResult, ModelInfo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("neural_estimator_service")

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
EXPECTED_GRID_SHAPE = (2, 14, 48)

REQUEST_COUNT = Counter(
    "estimator_requests_total", "Total requests handled", ["endpoint", "model"]
)
INFERENCE_LATENCY = Histogram(
    "estimator_inference_latency_ms", "Model inference latency in milliseconds", ["model"]
)

app = FastAPI(title="5G Neural Channel Estimator Inference Service")

_fixtures: dict[str, dict] = {}


@app.on_event("startup")
def startup():
    loaded = inference.load_sessions()
    logger.info("loaded models: %s", list(loaded.keys()))
    for bucket in ["low", "mid", "high"]:
        path = os.path.join(FIXTURES_DIR, f"{bucket}.npz")
        if os.path.exists(path):
            data = np.load(path)
            _fixtures[bucket] = {
                "noisy": data["noisy"],
                "perfect": data["perfect"],
                "snr_db": float(data["snr_db"]),
            }
    logger.info("loaded fixtures: %s", list(_fixtures.keys()))


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": inference.available_models()}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models", response_model=list[ModelInfo])
def models():
    return [
        ModelInfo(
            name=name,
            artifact_size_kb=inference.artifact_size_kb(name),
            provenance=inference.provenance(name),
        )
        for name in inference.available_models()
    ]


@app.post(
    "/v1/estimate",
    response_model=EstimateResponse,
    responses={400: {"description": "malformed grid"}, 404: {"description": "model not available"}},
)
def estimate(req: EstimateRequest):
    if req.model not in inference.available_models():
        raise HTTPException(status_code=404, detail=f"model '{req.model}' not available")
    try:
        grid = np.asarray(req.grid, dtype=np.float32)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="grid must be a nested numeric list")
    if grid.ndim != 4 or grid.shape[1:] != EXPECTED_GRID_SHAPE:
        raise HTTPException(
            status_code=400,
            detail=f"grid must have shape (batch, 2, 14, 48), got {grid.shape}",
        )
    REQUEST_COUNT.labels(endpoint="estimate", model=req.model).inc()
    delta, latency_ms = inference.run(req.model, grid)
    INFERENCE_LATENCY.labels(model=req.model).observe(latency_ms)
    refined = grid + delta
    logger.info("estimate model=%s batch=%d latency_ms=%.3f", req.model, grid.shape[0], latency_ms)
    return EstimateResponse(model=req.model, refined=refined.tolist(), latency_ms=latency_ms)


@app.get("/v1/demo/{bucket}", response_model=DemoResponse, responses={404: {"description": "unknown bucket"}})
def demo(bucket: str):
    if bucket not in _fixtures:
        raise HTTPException(status_code=404, detail=f"unknown bucket '{bucket}', expected low/mid/high")
    fixture = _fixtures[bucket]
    noisy, perfect = fixture["noisy"], fixture["perfect"]

    results = {}

    start = time.perf_counter()
    raw_mse = float(np.mean((noisy - perfect) ** 2))
    results["raw_ls"] = MethodResult(mse_vs_perfect=raw_mse, latency_ms=(time.perf_counter() - start) * 1000)

    for name in inference.available_models():
        REQUEST_COUNT.labels(endpoint="demo", model=name).inc()
        delta, latency_ms = inference.run(name, noisy)
        INFERENCE_LATENCY.labels(model=name).observe(latency_ms)
        refined = noisy + delta
        mse = float(np.mean((refined - perfect) ** 2))
        results[name] = MethodResult(mse_vs_perfect=mse, latency_ms=latency_ms)

    logger.info("demo bucket=%s snr_db=%.1f", bucket, fixture["snr_db"])
    return DemoResponse(bucket=bucket, snr_db=fixture["snr_db"], results=results)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
