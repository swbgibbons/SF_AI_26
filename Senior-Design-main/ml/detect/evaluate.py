"""Score a trained detector on a held-out split, two ways.

1. Boxes: mAP@50 and mAP@50-95 per class (does it find and outline the damage?).
2. Photos: the photo's damage type is the class of its most confident box, or "none" when no box
   clears --conf. Compared with the manifest label (undamaged photos should get "none").

    python -m detect.evaluate --weights runs/detect/v1/weights/best.pt --split test --out reports/v1

Writes metrics.json, a photo-level confusion table and annotated predictions to --out.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from detect.build_dataset import CLASSES, label_dir


def photo_level(model, rows, dataset, conf, out):
    pred_dir = out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pairs, details = [], []
    for row in rows:
        result = model.predict(dataset / row["path"], conf=conf, verbose=False)[0]
        boxes = result.boxes
        if len(boxes):
            top = int(boxes.conf.argmax())
            pred, score = CLASSES[int(boxes.cls[top])], float(boxes.conf[top])
        else:
            pred, score = "none", 0.0
        truth = row["damage_type"] if row["label"] == "damaged" else "none"
        pairs.append((truth, pred))
        details.append({"path": row["path"], "truth": truth, "pred": pred, "score": round(score, 3),
                        "boxes": len(boxes)})
        result.save(filename=str(pred_dir / f"{truth}__as__{pred}__{Path(row['path']).name}"))
    labels = CLASSES + ["none"]
    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in pairs:
        confusion[t][p] += 1
    damaged = [(t, p) for t, p in pairs if t != "none"]
    clean = [(t, p) for t, p in pairs if t == "none"]
    summary = {
        "photos": len(pairs),
        "type_accuracy_on_damaged": round(sum(t == p for t, p in damaged) / max(len(damaged), 1), 3),
        "damage_found_recall": round(sum(p != "none" for _, p in damaged) / max(len(damaged), 1), 3),
        "clean_roofs_left_clean": f"{sum(p == 'none' for _, p in clean)}/{len(clean)}",
        "per_type_recall": {c: f"{sum(p == c for t, p in damaged if t == c)}/{sum(t == c for t, _ in damaged)}"
                            for c in CLASSES},
    }
    return summary, confusion, details


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data/det/yolo/data.yaml")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--version", help="frozen label snapshot the model was trained on, e.g. v1")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=Path("reports/latest"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    val = model.val(data=args.data, split=args.split, plots=False, verbose=False, project=str(args.out), name="val")
    boxes = {"mAP50": round(float(val.box.map50), 3), "mAP50_95": round(float(val.box.map), 3), "per_class_AP50": {}}
    for i, c in enumerate(val.box.ap_class_index):
        boxes["per_class_AP50"][CLASSES[int(c)]] = round(float(val.box.ap50[i]), 3)

    rows = [r for r in csv.DictReader(open(label_dir(args.dataset, args.version) / "manifest.csv", newline=""))
            if r["split"] == args.split and r["label"] != "unusable"]
    photos, confusion, details = photo_level(model, rows, args.dataset, args.conf, args.out)

    report = {"weights": args.weights, "dataset_version": args.version or "working", "split": args.split, "conf": args.conf, "boxes": boxes, "photos": photos,
              "confusion_truth_by_pred": confusion}
    json.dump(report, open(args.out / "metrics.json", "w"), indent=2)
    with open(args.out / "photos.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(details[0]))
        w.writeheader()
        w.writerows(details)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
