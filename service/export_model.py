import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.benchmark import load_refiner, grid_shape

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
ONNX_OPSET = 17


def export_onnx(model, dummy_input, out_path):
    torch.onnx.export(
        model, dummy_input, out_path,
        input_names=["grid"], output_names=["refined"],
        dynamic_axes={"grid": {0: "batch"}, "refined": {0: "batch"}},
        opset_version=ONNX_OPSET,
    )


def check_parity(torch_model, onnx_path, x, atol=1e-4):
    import onnxruntime as ort
    with torch.no_grad():
        torch_out = torch_model(x).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"grid": x.numpy()})[0]
    max_diff = np.abs(torch_out - onnx_out).max()
    return max_diff, max_diff < atol


def export_via_torchscript(model, dummy_input, out_path):
    traced = torch.jit.trace(model, dummy_input)
    traced.save(out_path)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(name, ckpt_path, export_format, parity_max_diff):
    checkpoint_sha256 = sha256_of(ckpt_path)
    meta_path = os.path.splitext(ckpt_path)[0] + ".meta.json"
    training_run = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            training_run = json.load(f)

    manifest = {
        "model": name,
        "export_format": export_format,
        "onnx_opset": ONNX_OPSET if export_format == "onnx" else None,
        "checkpoint_file": os.path.basename(ckpt_path),
        "checkpoint_sha256": checkpoint_sha256,
        "parity_max_diff": parity_max_diff,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "training_run": training_run,
    }
    manifest_path = os.path.join(ARTIFACTS_DIR, f"{name}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main(checkpoints_dir="checkpoints"):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    x = torch.randn(*grid_shape(4))

    for name in ["cnn", "transformer"]:
        ckpt = os.path.join(os.path.dirname(__file__), "..", checkpoints_dir, f"{name}.pt")
        if not os.path.exists(ckpt):
            print(f"skip {name}: no checkpoint at {ckpt}")
            continue
        model = load_refiner(name, os.path.join(os.path.dirname(__file__), "..", checkpoints_dir))
        onnx_path = os.path.join(ARTIFACTS_DIR, f"{name}_refiner.onnx")
        try:
            export_onnx(model, x, onnx_path)
            max_diff, ok = check_parity(model, onnx_path, x)
            if not ok:
                raise RuntimeError(f"parity check failed, max_diff={max_diff}")
            manifest_path = write_manifest(name, ckpt, "onnx", float(max_diff))
            print(f"{name}: ONNX export OK, parity max_diff={max_diff:.2e} -> {onnx_path}")
            print(f"{name}: manifest -> {manifest_path}")
        except Exception as e:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            data_path = onnx_path + ".data"
            if os.path.exists(data_path):
                os.remove(data_path)
            ts_path = os.path.join(ARTIFACTS_DIR, f"{name}_refiner.pt.ts")
            export_via_torchscript(model, x, ts_path)
            manifest_path = write_manifest(name, ckpt, "torchscript", None)
            print(f"{name}: ONNX export failed ({e}); fell back to TorchScript -> {ts_path}")
            print(f"{name}: manifest -> {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", default="checkpoints")
    args = parser.parse_args()
    main(args.checkpoints_dir)
