import json
import os
import time

import numpy as np
import onnxruntime as ort

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

_sessions: dict[str, ort.InferenceSession] = {}


def load_sessions():
    for name in ["cnn", "transformer"]:
        path = os.path.join(ARTIFACTS_DIR, f"{name}_refiner.onnx")
        if os.path.exists(path):
            _sessions[name] = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _sessions


def available_models():
    return list(_sessions.keys())


def artifact_size_kb(name):
    path = os.path.join(ARTIFACTS_DIR, f"{name}_refiner.onnx")
    data_path = path + ".data"
    total = os.path.getsize(path)
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total / 1024


def provenance(name):
    path = os.path.join(ARTIFACTS_DIR, f"{name}_manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def run(model_name, grid):
    if model_name not in _sessions:
        raise KeyError(f"model {model_name} not loaded")
    sess = _sessions[model_name]
    x = np.asarray(grid, dtype=np.float32)
    start = time.perf_counter()
    out = sess.run(None, {"grid": x})[0]
    latency_ms = (time.perf_counter() - start) * 1000
    return out, latency_ms
