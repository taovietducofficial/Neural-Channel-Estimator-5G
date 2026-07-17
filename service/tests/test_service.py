import numpy as np
from fastapi.testclient import TestClient

from service.app import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "cnn" in body["models_loaded"]
        assert "transformer" in body["models_loaded"]
    print("test_health: OK")


def test_demo_cnn_beats_raw_ls_at_mid_snr():
    with TestClient(app) as client:
        r = client.get("/v1/demo/mid")
        assert r.status_code == 200
        body = r.json()
        assert body["bucket"] == "mid"
        raw_mse = body["results"]["raw_ls"]["mse_vs_perfect"]
        cnn_mse = body["results"]["cnn"]["mse_vs_perfect"]
        assert cnn_mse < raw_mse, f"expected CNN to beat raw LS at mid SNR, got cnn={cnn_mse} raw={raw_mse}"
    print("test_demo_cnn_beats_raw_ls_at_mid_snr: OK", raw_mse, cnn_mse)


def test_demo_unknown_bucket_404():
    with TestClient(app) as client:
        r = client.get("/v1/demo/nope")
        assert r.status_code == 404
    print("test_demo_unknown_bucket_404: OK")


def test_estimate_roundtrip_shape():
    with TestClient(app) as client:
        grid = np.zeros((1, 2, 14, 48), dtype=np.float32).tolist()
        r = client.post("/v1/estimate", json={"model": "cnn", "grid": grid})
        assert r.status_code == 200
        body = r.json()
        refined = body["refined"]
        shape = (len(refined), len(refined[0]), len(refined[0][0]), len(refined[0][0][0]))
        assert shape == (1, 2, 14, 48)
    print("test_estimate_roundtrip_shape: OK")


def test_estimate_malformed_grid_400():
    with TestClient(app) as client:
        r = client.post("/v1/estimate", json={"model": "cnn", "grid": [1, 2, 3]})
        assert r.status_code == 400
    print("test_estimate_malformed_grid_400: OK")


if __name__ == "__main__":
    test_health()
    test_demo_cnn_beats_raw_ls_at_mid_snr()
    test_demo_unknown_bucket_404()
    test_estimate_roundtrip_shape()
    test_estimate_malformed_grid_400()
    print("all service checks passed")
