"""Add the RF_Clean Roboflow batch (hail, wind, missing shingles) to the working labels.

    python -m detect.import_rf_clean --src path/to/RF_Clean_Annotated

RF_Clean_Annotated is the 1003-photo Roboflow batch already converted to this project's classes
(wind=0, hail=1, missing_shingles=2), split by roof, with a manifest.csv describing every photo.
This copies each photo to dataset/images/rf-<name> and appends one row per photo to
dataset/manifest.csv and one row per box to dataset/boxes.csv. Existing rows (v1) are untouched.
Refuses to run twice. Freeze afterwards with `python -m detect.freeze v2`.
"""

import argparse
import csv
import re
import shutil
from pathlib import Path

from detect.build_dataset import CLASSES

# How sure the source labels are, by the RF_Clean folder the photo came from.
CONFIDENCE = {
    "hail_damage": "high",      # hail hits that passed a visual check on the full photo
    "poten_hail": "low",        # hail boxes on roofs that look undamaged at full-photo view
    "wind_damage": "medium",    # Roboflow set1 broken/loose/missing shingle polygons
    "no_damage": "medium",
}
HINT = {"wind": "wind", "hail": "hail", "missing_shingles": "missing_broken"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="RF_Clean_Annotated folder")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    root = args.dataset
    rows = list(csv.DictReader(open(root / "manifest.csv", newline="")))
    fields = list(rows[0])
    if any(Path(r["path"]).name.startswith("rf-") for r in rows):
        raise SystemExit("RF_Clean photos are already in manifest.csv.")
    box_fields = list(csv.DictReader(open(root / "boxes.csv", newline="")).fieldnames)
    idx = max(int(r["idx"]) for r in rows if r["idx"]) + 1
    new_rows, new_boxes = [], []
    (root / "images").mkdir(exist_ok=True)
    for src in csv.DictReader(open(args.src / "manifest.csv", newline="")):
        name, split = src["file"], src["split"]
        path = f"images/rf-{name}"
        if (root / path).exists():
            raise SystemExit(f"{path} already exists; photos are never overwritten.")
        shutil.copy2(args.src / "images" / split / name, root / path)
        boxes = []
        for line in open(args.src / "labels" / split / (name[:-4] + ".txt")):
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            cx, cy, w, h = map(float, (cx, cy, w, h))
            boxes.append((CLASSES[int(c)], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        kinds = sorted({b[0] for b in boxes})
        assert len(kinds) <= 1, f"{name} has more than one damage type"
        kind = kinds[0] if kinds else ""
        manual = "visual review" in src["notes"]
        conf = "high" if manual else CONFIDENCE[src["rf_clean_folder"]]
        notes = [f"RF_Clean {src['rf_clean_folder']}"]
        if src["set1_original_classes"]:
            notes.append(f"Roboflow set1 classes: {src['set1_original_classes']}")
        notes += [n for n in src["notes"].split("; ") if n]
        new_rows.append({
            **{f: "" for f in fields},
            "path": path, "label": "damaged" if boxes else "undamaged",
            "group": "rf-" + re.sub(r"[^A-Za-z0-9_.-]", "_", src["roof_group"]), "split": split,
            "confidence": conf, "hail": int(kind == "hail"), "wind": int(kind == "wind"),
            "missing_shingles": int(kind == "missing_shingles"), "damage_type": kind,
            "types_confidence": conf if kind else "", "damage_hint": HINT.get(kind, "none"),
            "notes": "; ".join(notes), "source": "roboflow_" + name[:2].replace("s", "set"), "idx": idx,
        })
        idx += 1
        status = "manual" if manual else "roboflow"
        for cls, x1, y1, x2, y2 in boxes:
            new_boxes.append({"path": path, "class": cls, "x1": f"{x1:.5f}", "y1": f"{y1:.5f}",
                              "x2": f"{x2:.5f}", "y2": f"{y2:.5f}", "status": status})
    with open(root / "manifest.csv", "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writerows(new_rows)
    with open(root / "boxes.csv", "a", newline="") as f:
        csv.DictWriter(f, fieldnames=box_fields).writerows(new_boxes)
    print(f"added {len(new_rows)} photos and {len(new_boxes)} boxes")


if __name__ == "__main__":
    main()
