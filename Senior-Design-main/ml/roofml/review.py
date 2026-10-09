"""
Review labels by eye, and apply corrections made by moving files between folders.

    # one folder per label, each photo stamped with its label, photo id and confidence
    python -m roofml.review export --manifest dataset/manifest.csv --images-root dataset --out review

    # after moving wrongly labeled photos into the right folder:
    python -m roofml.review apply --manifest dataset/manifest.csv --review review

Folders: wind, missing_shingles, hail (damaged photos, one type each), undamaged, unusable.
Files are named <folder>__<photo id>.jpg; only the folder a file sits in matters on apply.
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

DAMAGE_TYPES = ["wind", "missing_shingles", "hail"]
FOLDERS = DAMAGE_TYPES + ["undamaged", "unusable"]
COLORS = {"wind": (25, 118, 210), "missing_shingles": (230, 81, 0), "hail": (123, 31, 162),
          "undamaged": (46, 125, 50), "unusable": (97, 97, 97)}
TITLES = {"wind": "WIND DAMAGE", "missing_shingles": "MISSING / BROKEN SHINGLES",
          "hail": "HAIL DAMAGE", "undamaged": "UNDAMAGED", "unusable": "UNUSABLE"}


def photo_id(path):
    m = re.search(r"(\d{6,})", Path(path).stem)
    return m.group(1) if m else Path(path).stem


def folder_of(row):
    if row["label"] == "damaged":
        return row.get("damage_type") or "wind"
    return row["label"]


def read_rows(manifest):
    with open(manifest, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(rows, manifest):
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def stamp(image, folder, pid, confidence, notes=""):
    """Return a copy of image with a colored banner naming its label."""
    w, h = image.size
    banner = 78
    out = Image.new("RGB", (w, h + banner), COLORS[folder])
    out.paste(image, (0, banner))
    d = ImageDraw.Draw(out)
    d.text((12, 8), TITLES[folder], fill="white", font=ImageFont.load_default(size=30))
    small = ImageFont.load_default(size=18)
    sub = f"photo {pid}  ·  confidence: {confidence or '?'}" + (f"  ·  {notes}" if notes else "")
    while d.textlength(sub, font=small) > w - 24:
        sub = sub[:-2]
    d.text((12, 48), sub, fill="white", font=small)
    if confidence in ("low", "medium"):  # frame the photos most worth checking
        edge = (229, 57, 53) if confidence == "low" else (255, 213, 0)
        d.rectangle([0, banner, w - 1, h + banner - 1], outline=edge, width=6)
    return out


def export(manifest, images_root, out):
    counts = Counter()
    for r in read_rows(manifest):
        folder = folder_of(r)
        conf = r.get("types_confidence") if r["label"] == "damaged" else r.get("confidence")
        with Image.open(Path(images_root) / r["path"]) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        dest = Path(out) / folder
        dest.mkdir(parents=True, exist_ok=True)
        pid = photo_id(r["path"])
        stamp(image, folder, pid, conf, r.get("notes", "")).save(dest / f"{folder}__{pid}.jpg", quality=85)
        counts[folder] += 1
    return counts


def apply(manifest, review):
    """Set each photo's label from the folder its review copy is in. Returns the changes."""
    rows = read_rows(manifest)
    by_id = {photo_id(r["path"]): r for r in rows}
    changes = []
    for folder in FOLDERS:
        for f in sorted((Path(review) / folder).glob("*.jpg")):
            r = by_id.get(photo_id(f))
            if r is None or folder_of(r) == folder:
                continue
            changes.append((photo_id(f), folder_of(r), folder))
            if folder in DAMAGE_TYPES:
                r["label"] = "damaged"
                r["damage_type"] = folder
                for t in DAMAGE_TYPES:
                    r[t] = "1" if t == folder else "0"
                r["types_confidence"] = "high"  # a person checked it
            else:
                r["label"] = folder
                r["damage_type"] = r["types_confidence"] = ""
                for t in DAMAGE_TYPES:
                    r[t] = ""
            r["confidence"] = "high"
    if changes:
        write_rows(rows, manifest)
    return changes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--manifest", required=True)
    e.add_argument("--images-root", required=True)
    e.add_argument("--out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--manifest", required=True)
    a.add_argument("--review", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "export":
        counts = export(args.manifest, args.images_root, args.out)
        print(", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    else:
        changes = apply(args.manifest, args.review)
        for pid, old, new in changes:
            print(f"{pid}: {old} -> {new}")
        print(f"{len(changes)} label(s) changed")


if __name__ == "__main__":
    main()
