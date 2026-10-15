"""Run the detector on a folder of new photos (for example, the sponsor's).

    python -m detect.predict --weights runs/detect/v1/weights/best.pt --images path/to/photos --out preds

Saves each photo with its boxes drawn, plus detections.json with every box
(class, confidence, x1, y1, x2, y2 in pixels) and the photo's overall damage type.
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from detect.build_dataset import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("preds"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    report = {}
    for path in files:
        result = model.predict(path, conf=args.conf, verbose=False)[0]
        dets = [{"class": CLASSES[int(c)], "confidence": round(float(s), 3), "box": [round(float(v), 1) for v in xyxy]}
                for c, s, xyxy in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)]
        dets.sort(key=lambda d: -d["confidence"])
        report[path.name] = {"damage_type": dets[0]["class"] if dets else "none", "detections": dets}
        result.save(filename=str(args.out / path.name))
    json.dump(report, open(args.out / "detections.json", "w"), indent=2)
    print(f"{len(files)} photos -> {args.out}")


if __name__ == "__main__":
    main()
