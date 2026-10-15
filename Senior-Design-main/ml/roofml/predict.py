"""
Classify photos as damaged / undamaged / unusable with the exported ONNX model.

Each photo first goes through the quality gate; photos that fail it are
"unusable" without running the model. Only needs numpy, Pillow and
onnxruntime, so the backend can import RoofClassifier directly.

    python -m roofml.predict runs/convnext_tiny/roof_cls.onnx photos/ [--json]
"""

import argparse
import json
from dataclasses import asdict

import numpy as np
import onnxruntime as ort
from PIL import Image

from .quality import QualityConfig, check_quality, iter_images, load_image

MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class RoofClassifier:
    def __init__(self, onnx_path, quality_config=QualityConfig()):
        self.session = ort.InferenceSession(onnx_path)
        meta = self.session.get_modelmeta().custom_metadata_map
        self.classes = json.loads(meta["classes"])
        self.image_size = int(meta["image_size"])
        self.quality_config = quality_config

    def _preprocess(self, image):
        x = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        x = (np.asarray(x, dtype=np.float32) / 255.0 - MEANS) / STDS
        return x.transpose(2, 0, 1)[None]

    def predict(self, image):
        """image: PIL.Image (already upright). Returns a dict with label, scores, quality."""
        quality = check_quality(image, self.quality_config)
        if not quality.usable:
            return {"label": "unusable", "reason": "quality_gate", "scores": None, "quality": asdict(quality)}
        logits = self.session.run(None, {"image": self._preprocess(image)})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        scores = {c: round(float(s), 4) for c, s in zip(self.classes, p)}
        label = max(scores, key=scores.get)
        return {"label": label, "reason": "model", "scores": scores, "quality": asdict(quality)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", action="store_true", help="one JSON object per line")
    args = ap.parse_args(argv)

    clf = RoofClassifier(args.model)
    for path in iter_images(args.paths):
        result = clf.predict(load_image(path))
        if args.json:
            print(json.dumps({"image": str(path), **result}))
        else:
            detail = ", ".join(result["quality"]["reasons"]) if result["reason"] == "quality_gate" \
                else " ".join(f"{k}={v:.2f}" for k, v in result["scores"].items())
            print(f"{result['label']:10s} {path}  ({detail})")


if __name__ == "__main__":
    main()
