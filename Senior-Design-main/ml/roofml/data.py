"""
Manifest handling, train/val/test split and the PyTorch dataset.

The manifest is a CSV with one row per image:

    path,label,group,source,notes
    damaged/roof_012.jpg,damaged,house_012,istock,torn shingles near ridge

- path:   relative to --images-root
- label:  damaged | undamaged | unusable
- group:  the property/roof the photo shows. Photos of the same roof share a
          group so they always land in the same split (otherwise the test set
          contains near-copies of training photos and the score is inflated).
- source, notes: free text, optional.
- split:  optional column (train/val/test); if absent, `assign_splits` fills it.
"""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import CLASSES
from .quality import IMAGE_EXTS, load_image

FIELDS = ["path", "label", "group", "source", "notes", "split"]


def read_manifest(path):
    with open(path, newline="") as f:
        rows = [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    bad = [r for r in rows if r["label"] not in CLASSES]
    if bad:
        raise ValueError(f"{len(bad)} rows have a label outside {CLASSES}, e.g. {bad[0]}")
    for r in rows:
        r["group"] = r.get("group") or Path(r["path"]).stem
    return rows


def write_manifest(rows, path):
    # Keep any extra columns (e.g. damage-type tags) after the standard ones.
    extra = [k for r in rows for k in r if k not in FIELDS]
    fields = FIELDS + list(dict.fromkeys(extra))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def manifest_from_folders(images_root):
    """Build rows from images_root/<label>/*.jpg folders (one group per file)."""
    root = Path(images_root)
    rows = []
    for label in CLASSES:
        for f in sorted((root / label).rglob("*")):
            if f.suffix.lower() in IMAGE_EXTS:
                rows.append({"path": str(f.relative_to(root)), "label": label,
                             "group": f.stem, "source": "", "notes": ""})
    return rows


def assign_splits(rows, val=0.15, test=0.15, seed=0):
    """Split by group, stratified by each group's majority label. Keeps existing splits."""
    if all(r.get("split") for r in rows):
        return rows
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    by_label = defaultdict(list)
    for g, members in groups.items():
        by_label[Counter(m["label"] for m in members).most_common(1)[0][0]].append(g)

    rng = random.Random(seed)
    for label in sorted(by_label):
        gs = sorted(by_label[label])
        rng.shuffle(gs)
        n_test = round(len(gs) * test)
        n_val = round(len(gs) * val)
        for i, g in enumerate(gs):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            for r in groups[g]:
                r["split"] = split
    return rows


def summarize(rows):
    table = Counter((r.get("split", "?"), r["label"]) for r in rows)
    splits = sorted({s for s, _ in table})
    lines = ["split".ljust(8) + "".join(c.rjust(11) for c in CLASSES)]
    for s in splits:
        lines.append(s.ljust(8) + "".join(str(table[(s, c)]).rjust(11) for c in CLASSES))
    return "\n".join(lines)


class RoofDataset:
    """torch Dataset over manifest rows. Import torch lazily so the quality gate stays light."""

    def __init__(self, rows, images_root, transform):
        self.rows = rows
        self.root = Path(images_root)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        image = load_image(self.root / r["path"])
        return self.transform(image), CLASSES.index(r["label"])


def build_transforms(image_size, train):
    from torchvision import transforms as T

    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if not train:
        return T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), norm])
    return T.Compose([
        # Mild crop only: a heavy crop can cut the damage out of a "damaged" photo.
        T.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.3, 0.3, 0.2, 0.02),
        T.RandomApply([T.GaussianBlur(5)], p=0.2),
        T.ToTensor(),
        norm,
    ])
