"""
Export a trained checkpoint to ONNX (what the Django backend runs) and check it
matches PyTorch.

    python -m roofml.export_onnx runs/convnext_tiny/best.pt --out runs/convnext_tiny/roof_cls.onnx
"""

import argparse
import json

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .train import build_model


def export(checkpoint, out_path):
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(ckpt["model"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    size = ckpt["image_size"]
    dummy = torch.randn(2, 3, size, size)
    torch.onnx.export(model, dummy, out_path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                      dynamo=False)

    # Store what inference needs inside the file, so the backend can't get it out of sync.
    m = onnx.load(out_path)
    for k, v in {"classes": json.dumps(ckpt["classes"]), "image_size": str(size),
                 "architecture": ckpt["model"]}.items():
        m.metadata_props.add(key=k, value=v)
    onnx.save(m, out_path)

    with torch.no_grad():
        expected = model(dummy).numpy()
    got = ort.InferenceSession(out_path).run(None, {"image": dummy.numpy()})[0]
    diff = float(np.abs(expected - got).max())
    if diff > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {diff}")
    return diff


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    diff = export(args.checkpoint, args.out)
    print(f"wrote {args.out} (max diff vs PyTorch {diff:.2e})")


if __name__ == "__main__":
    main()
