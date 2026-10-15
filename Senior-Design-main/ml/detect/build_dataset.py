"""Build detector datasets (YOLO and COCO) from dataset/manifest.csv and dataset/boxes.csv.

boxes.csv holds one row per box, coordinates normalized to 0-1 (x1, y1, x2, y2).
Damaged photos get their boxes; undamaged photos go in as background (no boxes);
unusable photos are left out. Splits come from the manifest, which splits by roof group.

    python -m detect.build_dataset --version v1 --out data/det-v1   # frozen snapshot
    python -m detect.build_dataset --out data/det                   # working labels

--version reads dataset/versions/<v>/ and checks every file against its SHA256SUMS first,
so a model trained on v1 can always be rebuilt from exactly the same photos and labels.
"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

# Order matches the backend's ["wind", "hail"] with the new class appended.
CLASSES = ["wind", "hail", "missing_shingles"]
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def verify(dataset, version):
    """Raise if any file listed in versions/<version>/SHA256SUMS changed or is missing."""
    sums = dataset / "versions" / version / "SHA256SUMS"
    bad = []
    for line in sums.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = dataset / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    if bad:
        raise ValueError(f"dataset {version} changed ({len(bad)} files), e.g. {bad[:3]}. Frozen versions must not be edited.")


def label_dir(dataset, version=None):
    if version is None:
        return dataset
    verify(dataset, version)
    return dataset / "versions" / version


def load(dataset, version=None):
    labels = label_dir(dataset, version)
    rows = list(csv.DictReader(open(labels / "manifest.csv", newline="")))
    boxes = defaultdict(list)
    for b in csv.DictReader(open(labels / "boxes.csv", newline="")):
        boxes[b["path"]].append(b)
    return rows, boxes


def build(dataset, out, version=None):
    rows, boxes = load(dataset, version)
    if out.exists():
        shutil.rmtree(out)
    coco = {s: {"images": [], "annotations": [], "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES)]}
            for s in SPLITS}
    counts = Counter()
    for image_id, row in enumerate(rows):
        if row["label"] == "unusable":
            continue
        split = row["split"]
        name = Path(row["path"]).name
        yolo_img = out / "yolo" / split / "images" / name
        yolo_lbl = out / "yolo" / split / "labels" / (Path(name).stem + ".txt")
        coco_img = out / "coco" / SPLITS[split] / name
        for p in (yolo_img, yolo_lbl, coco_img):
            p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(dataset / row["path"], yolo_img)
        shutil.copy(dataset / row["path"], coco_img)
        w, h = Image.open(yolo_img).size
        coco[split]["images"].append({"id": image_id, "file_name": name, "width": w, "height": h})
        lines = []
        if row["label"] == "damaged":
            if not boxes[row["path"]]:
                raise ValueError(f"damaged photo has no boxes: {row['path']}")
            for b in boxes[row["path"]]:
                c = CLASSES.index(b["class"])
                x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
                lines.append(f"{c} {(x1 + x2) / 2:.5f} {(y1 + y2) / 2:.5f} {x2 - x1:.5f} {y2 - y1:.5f}")
                coco[split]["annotations"].append({
                    "id": len(coco[split]["annotations"]), "image_id": image_id, "category_id": c,
                    "bbox": [round(x1 * w, 1), round(y1 * h, 1), round((x2 - x1) * w, 1), round((y2 - y1) * h, 1)],
                    "area": round((x2 - x1) * w * (y2 - y1) * h, 1), "iscrowd": 0})
                counts[(split, b["class"])] += 1
        else:
            counts[(split, "background")] += 1
        yolo_lbl.write_text("\n".join(lines))
    for split, data in coco.items():
        json.dump(data, open(out / "coco" / SPLITS[split] / "_annotations.coco.json", "w"))
    yaml = [f"path: {(out / 'yolo').resolve()}", "train: train/images", "val: val/images", "test: test/images",
            "names:"] + [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    (out / "yolo" / "data.yaml").write_text("\n".join(yaml) + "\n")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--version", help="frozen label snapshot in dataset/versions/, e.g. v1")
    parser.add_argument("--out", type=Path, default=Path("data/det"))
    args = parser.parse_args()
    counts = build(args.dataset, args.out, args.version)
    for split in SPLITS:
        print(split, {k: v for (s, k), v in sorted(counts.items()) if s == split})
    print(f"YOLO: {args.out / 'yolo' / 'data.yaml'}   COCO: {args.out / 'coco'}")


if __name__ == "__main__":
    main()
