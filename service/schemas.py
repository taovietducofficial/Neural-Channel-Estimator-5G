from typing import Literal

from pydantic import BaseModel


class EstimateRequest(BaseModel):
    model: Literal["cnn", "transformer"]
    grid: list


class EstimateResponse(BaseModel):
    model: str
    refined: list
    latency_ms: float


class ModelInfo(BaseModel):
    name: str
    artifact_size_kb: float
    provenance: dict | None = None


class MethodResult(BaseModel):
    mse_vs_perfect: float
    latency_ms: float


class DemoResponse(BaseModel):
    bucket: str
    snr_db: float
    results: dict[str, MethodResult]
