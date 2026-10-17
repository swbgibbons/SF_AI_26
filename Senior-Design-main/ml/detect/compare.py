"""Score YOLO and RF-DETR models on one split with the same scorer, so the numbers compare fairly.

    python -m detect.compare --version v1 --split test --out reports/compare_v1 \
        --model yolo:runs/detect/v1/weights/best.pt \
        --model rfdetr-nano:runs/rfdetr/rfdetr-n-v1/checkpoint_best_total.pth

Boxes are scored with pycocotools against the COCO export (data/det-v1/coco/<split>); photo-level
results use the same rule as detect.evaluate (type = most confident box above --conf). Also reports
CPU time per photo. Writes metrics.json to --out.
"""

import argparse
import csv
import io
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detect.build_dataset import CLASSES, label_dir

COCO_SPLIT = {"val": "valid", "test": "test"}


def load_model(spec):
    """'yolo:<weights>' or 'rfdetr-<size>:<checkpoint>' -> predict(image) -> (xyxy, cls, conf)."""
    kind, path = spec.split(":", 1)
    if kind == "yolo":
        from ultralytics import YOLO

        model = YOLO(path)

        def predict(image):
            b = model.predict(image, conf=0.001, verbose=False, device="cpu")[0].boxes
            return b.xyxy.numpy(), b.cls.numpy().astype(int), b.conf.numpy()
    else:
        import rfdetr

        from detect.train_rfdetr import SIZES

        model = getattr(rfdetr, SIZES[kind.split("-", 1)[1]])(pretrain_weights=path)

        def predict(image):
            d = model.predict(image, threshold=0.001)
            return d.xyxy, d.class_id.astype(int), d.confidence
    return kind, predict


def box_map(gt, preds):
    """preds: list of COCO result dicts. Returns overall and per-class AP like ultralytics reports."""
    if not preds:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "per_class_AP50": {c: 0.0 for c in CLASSES}}
    with redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(preds), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    p = ev.eval["precision"]  # [iou, recall, class, area, maxdets]
    per_class = {}
    for k, cat in enumerate(ev.params.catIds):
        if not gt.getAnnIds(catIds=[cat]):
            continue
        ap = p[0, :, k, 0, -1]
        per_class[gt.cats[cat]["name"]] = round(float(ap[ap > -1].mean()) if (ap > -1).any() else 0.0, 3)
    present = [v for v in per_class.values()]
    ap_all = p[:, :, [k for k, c in enumerate(ev.params.catIds) if gt.getAnnIds(catIds=[c])], 0, -1]
    return {"mAP50": round(float(np.mean(present)), 3),
            "mAP50_95": round(float(ap_all[ap_all > -1].mean()), 3),
            "per_class_AP50": per_class}


def photo_level(raw, rows, conf):
    pairs = []
    for row in rows:
        xyxy, cls, score = raw[row["path"]]
        keep = score >= conf
        pred = CLASSES[int(cls[keep][score[keep].argmax()])] if keep.any() else "none"
        truth = row["damage_type"] if row["label"] == "damaged" else "none"
        pairs.append((truth, pred))
    damaged = [(t, p) for t, p in pairs if t != "none"]
    clean = [(t, p) for t, p in pairs if t == "none"]
    return {
        "type_accuracy_on_damaged": f"{sum(t == p for t, p in damaged)}/{len(damaged)}",
        "damage_found": f"{sum(p != 'none' for _, p in damaged)}/{len(damaged)}",
        "clean_roofs_left_clean": f"{sum(p == 'none' for _, p in clean)}/{len(clean)}",
        "per_type_recall": {c: f"{sum(p == c for t, p in damaged if t == c)}/{sum(t == c for t, _ in damaged)}"
                            for c in CLASSES},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="yolo:<weights> or rfdetr-<size>:<ckpt>")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--coco", type=Path, default=Path("data/det-v1/coco"))
    parser.add_argument("--version", help="frozen label snapshot, e.g. v1")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--conf", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--out", type=Path, default=Path("reports/compare"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    split_dir = args.coco / COCO_SPLIT[args.split]
    with redirect_stdout(io.StringIO()):
        gt = COCO(str(split_dir / "_annotations.coco.json"))
    ids = {Path(im["file_name"]).name: im["id"] for im in gt.dataset["images"]}
    rows = [r for r in csv.DictReader(open(label_dir(args.dataset, args.version) / "manifest.csv", newline=""))
            if r["split"] == args.split and r["label"] != "unusable"]

    report = {"split": args.split, "dataset_version": args.version or "working", "photos": len(rows), "models": {}}
    for spec in args.model:
        kind, predict = load_model(spec)
        raw, preds, seconds = {}, [], []
        for row in rows:
            image = Image.open(args.dataset / row["path"]).convert("RGB")
            start = time.perf_counter()
            raw[row["path"]] = predict(image)
            seconds.append(time.perf_counter() - start)
            image_id = ids[Path(row["path"]).name]
            for (x1, y1, x2, y2), c, s in zip(*raw[row["path"]]):
                preds.append({"image_id": image_id, "category_id": int(c), "score": float(s),
                              "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
        report["models"][spec] = {
            "boxes": box_map(gt, preds),
            "photos": {f"conf_{c}": photo_level(raw, rows, c) for c in args.conf},
            "cpu_ms_per_photo": round(1000 * float(np.median(seconds[1:])), 1),
        }
        print(spec, json.dumps(report["models"][spec], indent=2))
    json.dump(report, open(args.out / "metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
