"""
Roof damage dataset comparison matrix (v2).

Scores candidate datasets for the roof damage detector (RF-DETR, ONNX in
the backend) against the three target classes: hail, wind, missing_shingles.

Two ways to fill in a dataset:
  * `add`      - type the facts and 1-5 scores by hand (like v1).
  * `profile`  - point it at a downloaded dataset folder; it scans the images
                 and labels, measures quality, counts classes and suggests
                 scores. Saves to the matrix with --save.

Usage:
  python roof_dataset_matrix.py                 # interactive menu
  python roof_dataset_matrix.py profile PATH [--name N] [--save]
  python roof_dataset_matrix.py list | show ID | export out.csv

Needs Pillow (already a backend dependency). numpy enables blur/exposure
checks. pillow-heif, if installed, enables HEIC/HEIF photos from phones.
"""

import argparse
import csv
import hashlib
import json
import sqlite3
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

try:
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = None  # orthomosaics / satellite tiles are huge
except ImportError:  # profiling needs Pillow, manual entry does not
    Image = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_OK = True
except ImportError:
    HEIF_OK = False

BASE_DIR = Path(__file__).resolve().parent
DB_NAME = BASE_DIR / "roof_dataset_matrix.db"

# -----------------------------
# PROJECT SETTINGS
# -----------------------------
TARGET_CLASSES = ["hail", "wind", "missing_shingles"]

# Raw label names seen in public datasets -> our classes. Anything not listed
# is reported as "unmapped" so you can extend this table.
CLASS_ALIASES = {
    "hail": "hail", "hail_damage": "hail", "hail damage": "hail", "hail_hit": "hail",
    "hailstrike": "hail", "bruise": "hail", "dent": "hail",
    "wind": "wind", "wind_damage": "wind", "wind damage": "wind",
    "creased_shingle": "wind", "lifted_shingle": "wind", "crease": "wind",
    "missing_shingle": "missing_shingles", "missing_shingles": "missing_shingles",
    "missing shingle": "missing_shingles", "missing": "missing_shingles",
    "missing-shingle": "missing_shingles",
}

# Model input / tile size used by the backend (core/services.py: tile_size=560;
# "normal" mode squashes the whole photo to the model input).
MODEL_INPUT = 560
MIN_BOX_PX_AT_INPUT = 8      # boxes smaller than this after resizing are near-undetectable
MIN_SIDE_PX = 480            # images smaller than this on the short side are "low res"
BLUR_THRESHOLD = 100.0       # variance of Laplacian below this = likely blurry

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}

WEIGHTS = {
    "scope_match": 0.30,
    "viewpoint_match": 0.25,
    "annotation_quality": 0.20,
    "image_quality": 0.15,
    "class_balance": 0.10,
}

RUBRIC = {
    "scope_match": "5 = all 3 classes labelled separately; 3 = 1-2 of them; 1 = generic 'damage' only",
    "viewpoint_match": "5 = same viewpoint as deployment photos (see --target-viewpoint); 1 = very different",
    "annotation_quality": "5 = tight boxes/polygons, consistent, verified; 1 = image-level tags or sloppy boxes",
    "image_quality": "5 = sharp, well exposed, short side >= 1024px; 1 = blurry/tiny/compressed",
    "class_balance": "5 = every class >= 500 instances and has undamaged roofs; 1 = one class dominates",
}

# Annotation types RF-DETR can train on (polygons/masks convert to boxes).
DETECTION_USABLE = {"bbox", "polygon", "mask"}

COLUMNS = [
    # name, sql type, label
    ("name", "TEXT NOT NULL", "Name"),
    ("source", "TEXT", "Source"),
    ("url", "TEXT", "URL"),
    ("local_path", "TEXT", "Local Path"),
    ("image_count", "INTEGER", "Image Count"),
    ("viewpoint", "TEXT", "Viewpoint"),
    ("damage_types", "TEXT", "Damage Types (raw)"),
    ("roof_materials", "TEXT", "Roof Materials"),
    ("hail_count", "INTEGER", "Hail Instances"),
    ("wind_count", "INTEGER", "Wind Instances"),
    ("missing_count", "INTEGER", "Missing Shingle Instances"),
    ("negative_images", "INTEGER", "Undamaged Images"),
    ("annotation_type", "TEXT", "Annotation Type"),
    ("annotation_format", "TEXT", "Annotation Format"),
    ("resolution", "TEXT", "Resolution"),
    ("file_formats", "TEXT", "File Formats"),
    ("quality_report", "TEXT", "Quality Report (json)"),
    ("license", "TEXT", "License"),
    ("license_ok", "INTEGER", "License OK"),
    ("notes", "TEXT", "Notes"),
    ("scope_match", "INTEGER", "Scope Match"),
    ("viewpoint_match", "INTEGER", "Viewpoint Match"),
    ("annotation_quality", "INTEGER", "Annotation Quality"),
    ("image_quality", "INTEGER", "Image Quality"),
    ("class_balance", "INTEGER", "Class Balance"),
    ("total_score", "REAL", "Total Score"),
]
SCORE_FIELDS = list(WEIGHTS)


# -----------------------------
# DATABASE
# -----------------------------
def connect():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cols = ",\n".join(f"{n} {t}" for n, t, _ in COLUMNS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS datasets (id INTEGER PRIMARY KEY AUTOINCREMENT,\n{cols})")
    # Migrate a v1 database in place: add any missing columns.
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(datasets)")}
    if "drone_similarity" in existing and "viewpoint_match" not in existing:
        conn.execute("ALTER TABLE datasets RENAME COLUMN drone_similarity TO viewpoint_match")
        existing.add("viewpoint_match")
    for n, t, _ in COLUMNS:
        if n not in existing:
            conn.execute(f"ALTER TABLE datasets ADD COLUMN {n} {t.replace(' NOT NULL', '')}")
    conn.commit()
    return conn


def calculate_score(scores, license_ok=1, annotation_type=""):
    """Weighted score out of 5. Hard gates zero out unusable datasets."""
    if any(scores.get(f) is None for f in SCORE_FIELDS):
        return None
    if license_ok == 0:
        return 0.0
    if annotation_type and annotation_type not in DETECTION_USABLE:
        return 0.0  # image-level labels can't train a detector
    return round(sum(scores[f] * w for f, w in WEIGHTS.items()), 2)


def save_dataset(conn, record, dataset_id=None):
    scores = {f: record.get(f) for f in SCORE_FIELDS}
    record["total_score"] = calculate_score(
        scores, record.get("license_ok", 1), (record.get("annotation_type") or "").lower())
    names = [n for n, _, _ in COLUMNS if n in record]
    if dataset_id is None:
        cur = conn.execute(
            f"INSERT INTO datasets ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
            [record[n] for n in names])
        dataset_id = cur.lastrowid
    else:
        conn.execute(
            f"UPDATE datasets SET {', '.join(n + ' = ?' for n in names)} WHERE id = ?",
            [record[n] for n in names] + [dataset_id])
    conn.commit()
    return dataset_id, record["total_score"]


def get_row(conn, dataset_id):
    try:
        dataset_id = int(dataset_id)
    except (TypeError, ValueError):
        return None
    return conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()


# -----------------------------
# PROFILER: read the actual files
# -----------------------------
def _avg_hash(img):
    small = img.convert("L").resize((8, 8))
    px = list(small.tobytes())
    avg = sum(px) / 64
    return sum(1 << i for i, p in enumerate(px) if p > avg)


def _blur_and_exposure(img):
    if np is None:
        return None, None
    g = np.asarray(img.convert("L").resize((512, 512)), dtype=np.float32)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return float(lap.var()), float(g.mean())


def _viewpoint_hint(exif):
    """Guess capture platform from EXIF. Returns 'drone', 'ground' or None."""
    make = str(exif.get(0x010F, "")).lower()   # Make
    model = str(exif.get(0x0110, "")).lower()  # Model
    if any(k in make + model for k in ("dji", "parrot", "skydio", "autel", "yuneec")):
        return "drone"
    if any(k in make for k in ("apple", "samsung", "google", "canon", "nikon", "sony")):
        return "ground"
    return None


def find_images(root):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_annotations(root, images):
    """Detect COCO / YOLO / Pascal VOC labels.

    Returns (format, type, {image_stem: [(raw_label, w_frac, h_frac), ...]}).
    """
    boxes = {}
    # COCO JSON
    for jf in root.rglob("*.json"):
        try:
            data = json.loads(jf.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict) or not {"images", "annotations", "categories"} <= data.keys():
            continue
        cats = {c["id"]: c["name"] for c in data["categories"]}
        imgs = {i["id"]: i for i in data["images"]}
        ann_type = "bbox"
        for i in imgs.values():
            boxes.setdefault(Path(i["file_name"]).stem, [])
        for a in data["annotations"]:
            im = imgs.get(a["image_id"])
            if not im:
                continue
            if a.get("segmentation"):
                ann_type = "polygon"
            w, h = (a.get("bbox") or [0, 0, 0, 0])[2:4]
            boxes[Path(im["file_name"]).stem].append(
                (cats.get(a["category_id"], "?"), w / max(im.get("width", 1), 1), h / max(im.get("height", 1), 1)))
        if boxes:
            return "coco", ann_type, boxes

    # Pascal VOC XML
    for xf in root.rglob("*.xml"):
        try:
            tree = ET.parse(xf).getroot()
        except ET.ParseError:
            continue
        if tree.tag != "annotation":
            continue
        size = tree.find("size")
        W = float(size.findtext("width", "1")) if size is not None else 1
        H = float(size.findtext("height", "1")) if size is not None else 1
        objs = []
        for o in tree.iter("object"):
            bb = o.find("bndbox")
            if bb is None:
                continue
            w = float(bb.findtext("xmax", "0")) - float(bb.findtext("xmin", "0"))
            h = float(bb.findtext("ymax", "0")) - float(bb.findtext("ymin", "0"))
            objs.append((o.findtext("name", "?"), w / max(W, 1), h / max(H, 1)))
        boxes[Path(tree.findtext("filename") or xf.name).stem] = objs
    if boxes:
        return "voc", "bbox", boxes

    # YOLO txt (+ optional data.yaml / classes.txt for names)
    names = []
    for nf in list(root.rglob("data.yaml")) + list(root.rglob("classes.txt")):
        text = nf.read_text(errors="ignore")
        if nf.suffix == ".txt":
            names = [l.strip() for l in text.splitlines() if l.strip()]
        else:  # tiny parser for `names: [a, b]` or a `names:` list
            for line in text.splitlines():
                if line.strip().startswith("names:") and "[" in line:
                    names = [n.strip(" '\"") for n in line.split("[", 1)[1].rstrip("]").split(",")]
                elif line.strip().startswith(("- ", "0:", "1:", "2:", "3:", "4:")) and "names" in text:
                    names.append(line.split(":", 1)[-1].lstrip("- ").strip(" '\""))
        if names:
            break
    stems = {p.stem for p in images}
    ann_type = "bbox"
    found = False
    for tf in root.rglob("*.txt"):
        if tf.stem not in stems:
            continue
        found = True
        objs = []
        for line in tf.read_text(errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            label = names[cid] if cid < len(names) else f"class_{cid}"
            if len(parts) > 5:  # YOLO-seg polygon
                ann_type = "polygon"
                xs, ys = [float(v) for v in parts[1::2]], [float(v) for v in parts[2::2]]
                objs.append((label, max(xs) - min(xs), max(ys) - min(ys)))
            else:
                objs.append((label, float(parts[3]), float(parts[4])))
        boxes[tf.stem] = objs
    if found:
        return "yolo", ann_type, boxes

    # Classification layout: images sorted into one folder per class.
    parents = {p.parent.name for p in images}
    if 1 < len(parents) <= 20:
        return "folders", "image-level", {str(p.relative_to(root)): [(p.parent.name, 1.0, 1.0)] for p in images}
    return "none", "none", {}


def profile_dataset(root, sample_limit=2000):
    root = Path(root)
    images = find_images(root)
    report = {"path": str(root), "image_files": len(images)}
    if Image is None:
        raise SystemExit("Profiling needs Pillow: pip install pillow")

    formats, modes, viewpoints = Counter(), Counter(), Counter()
    sizes, blurs, brightness = [], [], []
    corrupt, unreadable_heic, rotated, dup = [], 0, 0, 0
    hashes = {}
    long_side = {}  # image stem -> long side px, for box-size checks
    file_hashes = set()
    exact_dupes = 0
    step = max(1, len(images) // sample_limit)
    sampled = images[::step]

    for p in sampled:
        ext = p.suffix.lower()
        if ext in (".heic", ".heif") and not HEIF_OK:
            unreadable_heic += 1
            continue
        try:
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            if digest in file_hashes:
                exact_dupes += 1
            file_hashes.add(digest)
            with Image.open(p) as img:
                img.verify()
            with Image.open(p) as img:
                exif = img.getexif()
                if exif.get(0x0112, 1) not in (1, None):
                    rotated += 1
                img = ImageOps.exif_transpose(img)
                formats[img.format or ext.lstrip(".").upper()] += 1
                modes[img.mode] += 1
                sizes.append(img.size)
                long_side[p.stem] = max(img.size)
                vp = _viewpoint_hint(exif)
                if vp:
                    viewpoints[vp] += 1
                rgb = img.convert("RGB")
                h = _avg_hash(rgb)
                if h in hashes:
                    dup += 1
                hashes[h] = p
                b, e = _blur_and_exposure(rgb)
                if b is not None:
                    blurs.append(b)
                    brightness.append(e)
        except Exception as exc:  # truncated, wrong extension, etc.
            corrupt.append(f"{p.name}: {type(exc).__name__}")

    short = [min(s) for s in sizes]
    report.update({
        "sampled": len(sampled),
        "formats": dict(formats),
        "color_modes": dict(modes),
        "corrupt": len(corrupt),
        "corrupt_examples": corrupt[:5],
        "heic_skipped_no_pillow_heif": unreadable_heic,
        "exif_rotated": rotated,
        "exact_duplicates": exact_dupes,
        "near_duplicates": max(0, dup - exact_dupes),
        "viewpoint_hints": dict(viewpoints),
    })
    if sizes:
        report["resolution"] = {
            "min": f"{min(sizes, key=lambda s: s[0] * s[1])}",
            "median_short_side": int(statistics.median(short)),
            "max": f"{max(sizes, key=lambda s: s[0] * s[1])}",
            "distinct_sizes": len(set(sizes)),
            "pct_low_res": round(100 * sum(s < MIN_SIDE_PX for s in short) / len(short), 1),
        }
    if blurs:
        report["pct_blurry"] = round(100 * sum(b < BLUR_THRESHOLD for b in blurs) / len(blurs), 1)
        report["pct_bad_exposure"] = round(100 * sum(e < 40 or e > 215 for e in brightness) / len(brightness), 1)

    # Labels
    fmt, ann_type, boxes = load_annotations(root, images)
    raw_counts, mapped = Counter(), Counter()
    tiny_full, tiny_tiled = 0, 0
    median_long = statistics.median([max(s) for s in sizes]) if sizes else MODEL_INPUT
    for stem, objs in boxes.items():
        img_long = long_side.get(stem, median_long)
        for label, wf, hf in objs:
            raw_counts[label] += 1
            mapped[CLASS_ALIASES.get(label.strip().lower(), "unmapped")] += 1
            if ann_type == "image-level":
                continue
            # Normal mode: whole photo resized to MODEL_INPUT. Tiled mode: native pixels.
            if min(wf, hf) * MODEL_INPUT < MIN_BOX_PX_AT_INPUT:
                tiny_full += 1
            if min(wf, hf) * max(img_long, MODEL_INPUT) < MIN_BOX_PX_AT_INPUT:
                tiny_tiled += 1
    total_objs = sum(raw_counts.values())
    labelled_stems = {s for s, o in boxes.items() if o}
    report.update({
        "annotation_format": fmt,
        "annotation_type": ann_type,
        "raw_label_counts": dict(raw_counts.most_common()),
        "class_counts": {c: mapped.get(c, 0) for c in TARGET_CLASSES + ["unmapped"]},
        "images_without_labels": sum(1 for p in images if p.stem not in labelled_stems
                                     and str(p.relative_to(root)) not in labelled_stems),
        "pct_tiny_boxes": round(100 * tiny_full / total_objs, 1) if total_objs else None,
        "pct_tiny_boxes_if_tiled": round(100 * tiny_tiled / total_objs, 1) if total_objs else None,
    })
    report["suggested_scores"] = suggest_scores(report)
    return report


def suggest_scores(r):
    """Turn measurements into 1-5 suggestions. Viewpoint is left to a human."""
    s = {}
    counts = [r["class_counts"][c] for c in TARGET_CLASSES]
    s["scope_match"] = {0: 1, 1: 2, 2: 4, 3: 5}[sum(c > 0 for c in counts)]

    t = r["annotation_type"]
    aq = {"polygon": 5, "mask": 5, "bbox": 4, "image-level": 1, "none": 1}.get(t, 2)
    if (r.get("pct_tiny_boxes_if_tiled") or 0) > 25:
        aq -= 1
    if r["class_counts"]["unmapped"] > sum(counts):
        aq -= 1  # mostly labels we can't use as-is
    if r["image_files"] and r["images_without_labels"] / r["image_files"] > 0.5 and t != "image-level":
        aq -= 1
    s["annotation_quality"] = max(1, aq)

    iq = 5
    res = r.get("resolution", {})
    if res.get("median_short_side", 0) < 1024:
        iq -= 1
    if res.get("pct_low_res", 0) > 20:
        iq -= 1
    if r.get("pct_blurry", 0) > 15:
        iq -= 1
    if r.get("pct_bad_exposure", 0) > 15:
        iq -= 1
    if r["sampled"] and (r["corrupt"] + r["near_duplicates"] + r["exact_duplicates"]) / r["sampled"] > 0.05:
        iq -= 1
    s["image_quality"] = max(1, iq)

    present = [c for c in counts if c > 0]
    if len(present) < 2:
        cb = 1
    else:
        ratio = max(present) / min(present)
        cb = 5 if ratio <= 2 else 4 if ratio <= 4 else 3 if ratio <= 8 else 2
        if r["images_without_labels"] == 0:
            cb = max(1, cb - 1)  # no undamaged roofs -> false positives
    s["class_balance"] = cb
    return s


def print_report(r):
    print(f"\n--- Profile: {r['path']} ---")
    for k in ("image_files", "sampled", "formats", "color_modes", "resolution", "pct_blurry",
              "pct_bad_exposure", "corrupt", "corrupt_examples", "heic_skipped_no_pillow_heif",
              "exif_rotated", "exact_duplicates", "near_duplicates", "viewpoint_hints",
              "annotation_format", "annotation_type", "raw_label_counts", "class_counts",
              "images_without_labels", "pct_tiny_boxes", "pct_tiny_boxes_if_tiled", "suggested_scores"):
        if k in r and r[k] not in (None, {}, [], 0) or k in ("corrupt", "class_counts"):
            print(f"{k:<28}{r.get(k)}")
    warnings = []
    if r["class_counts"]["unmapped"]:
        warnings.append("Some labels are unmapped: add them to CLASS_ALIASES or they won't count.")
    if r["annotation_type"] == "image-level":
        warnings.append("Image-level labels only: cannot train RF-DETR without re-labelling boxes.")
    if (r.get("pct_tiny_boxes") or 0) > 25:
        warnings.append(f"{r['pct_tiny_boxes']}% of boxes shrink below {MIN_BOX_PX_AT_INPUT}px when the whole "
                        "photo is resized to the model input: use the backend's tiled mode for this kind of data.")
    if r.get("exif_rotated"):
        warnings.append("Some photos rely on EXIF rotation: apply ImageOps.exif_transpose before "
                        "training and in the backend's open_image, or boxes won't line up.")
    if r.get("heic_skipped_no_pillow_heif"):
        warnings.append("HEIC files skipped: pip install pillow-heif, or convert them to JPEG.")
    for w in warnings:
        print("WARNING:", w)


# -----------------------------
# INTERACTIVE INPUT
# -----------------------------
def ask(prompt, default=None):
    suffix = f" [{default}]" if default not in (None, "") else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (default if default is not None else "")


def ask_int(prompt, default=None):
    while True:
        val = ask(prompt, default)
        if val in ("", None):
            return None
        try:
            return int(val)
        except ValueError:
            print("Enter a whole number (or leave blank if unknown).")


def get_score(field, default=None):
    print(f"  {RUBRIC[field]}")
    while True:
        val = ask(f"{field.replace('_', ' ').title()} (1-5)", default)
        try:
            score = int(val)
            if 1 <= score <= 5:
                return score
        except (TypeError, ValueError):
            pass
        print("Enter a number from 1 to 5.")


def ask_license():
    lic = ask("License (e.g. CC BY 4.0, CC BY-NC, unknown)")
    ok = ask("Can the team legally train on it for this project? (y/n/unknown)", "unknown").lower()
    return lic, {"y": 1, "yes": 1, "n": 0, "no": 0}.get(ok)


def enter_record(prefill=None):
    p = prefill or {}
    rec = {
        "name": ask("Dataset name", p.get("name")),
        "source": ask("Source (Kaggle, Roboflow, Hugging Face, own photos, ...)", p.get("source")),
        "url": ask("URL", p.get("url")),
        "image_count": ask_int("Number of images", p.get("image_count")),
        "viewpoint": ask("Viewpoint (drone / satellite / ground / mixed)", p.get("viewpoint")),
        "damage_types": ask("Damage types as labelled", p.get("damage_types")),
        "roof_materials": ask("Roof materials", p.get("roof_materials")),
        "hail_count": ask_int("Hail instances", p.get("hail_count")),
        "wind_count": ask_int("Wind instances", p.get("wind_count")),
        "missing_count": ask_int("Missing-shingle instances", p.get("missing_count")),
        "negative_images": ask_int("Undamaged images", p.get("negative_images")),
        "annotation_type": ask("Annotation type (bbox / polygon / mask / image-level)",
                               p.get("annotation_type")).lower(),
        "annotation_format": ask("Annotation format (coco / yolo / voc / csv / other)", p.get("annotation_format")),
        "resolution": ask("Resolution", p.get("resolution")),
        "file_formats": ask("File formats", p.get("file_formats")),
        "notes": ask("Notes", p.get("notes")),
    }
    rec["license"], rec["license_ok"] = ask_license()
    if rec["annotation_type"] and rec["annotation_type"] not in DETECTION_USABLE:
        print("Note: image-level labels can't train a detector; total score will be 0.")
    print("\nScore each category from 1 (poor) to 5 (excellent)")
    for f in SCORE_FIELDS:
        rec[f] = get_score(f, p.get(f))
    for k in ("local_path", "quality_report"):
        if k in p:
            rec[k] = p[k]
    return rec


# -----------------------------
# COMMANDS
# -----------------------------
def cmd_add(conn, prefill=None):
    print("\n--- Add New Dataset ---")
    rec = enter_record(prefill)
    dataset_id, total = save_dataset(conn, rec)
    print(f"\nSaved as ID {dataset_id}. Total score: {total}/5")


def cmd_profile(conn, path, name=None, save=False):
    r = profile_dataset(path)
    print_report(r)
    if not save:
        return
    res = r.get("resolution", {})
    prefill = {
        "name": name or Path(path).name,
        "local_path": str(Path(path).resolve()),
        "image_count": r["image_files"],
        "viewpoint": max(r["viewpoint_hints"], key=r["viewpoint_hints"].get) if r["viewpoint_hints"] else "",
        "damage_types": ", ".join(r["raw_label_counts"]),
        "hail_count": r["class_counts"]["hail"],
        "wind_count": r["class_counts"]["wind"],
        "missing_count": r["class_counts"]["missing_shingles"],
        "negative_images": r["images_without_labels"],
        "annotation_type": r["annotation_type"],
        "annotation_format": r["annotation_format"],
        "resolution": f"{res.get('min')} to {res.get('max')}, median short side {res.get('median_short_side')}",
        "file_formats": ", ".join(r["formats"]),
        "quality_report": json.dumps(r),
        **r["suggested_scores"],
    }
    print("\nReview the pre-filled values (Enter keeps them). Viewpoint match needs your judgement.")
    cmd_add(conn, prefill)


def cmd_list(conn):
    rows = conn.execute("SELECT * FROM datasets ORDER BY total_score IS NULL, total_score DESC, id").fetchall()
    if not rows:
        print("\nNo datasets saved yet.")
        return
    print("\n--- Dataset Comparison Matrix ---\n")
    hdr = f"{'Rank':<5}{'ID':<4}{'Dataset':<26}{'View':<10}{'Images':>7}{'Hail':>7}{'Wind':>7}{'Miss':>7}{'Neg':>7}  {'Ann':<12}{'Score':>6}  Flags"
    print(hdr)
    print("-" * len(hdr))
    for rank, r in enumerate(rows, start=1):
        flags = []
        if r["license_ok"] == 0:
            flags.append("LICENSE")
        elif r["license_ok"] is None:
            flags.append("license?")
        if (r["annotation_type"] or "") and r["annotation_type"] not in DETECTION_USABLE:
            flags.append("NO BOXES")
        fmt = lambda v: "-" if v is None else str(v)
        print(f"{rank:<5}{r['id']:<4}{(r['name'] or '')[:24]:<26}{(r['viewpoint'] or '-')[:9]:<10}"
              f"{fmt(r['image_count']):>7}{fmt(r['hail_count']):>7}{fmt(r['wind_count']):>7}"
              f"{fmt(r['missing_count']):>7}{fmt(r['negative_images']):>7}  "
              f"{(r['annotation_type'] or '-')[:11]:<12}{fmt(r['total_score']):>6}  {' '.join(flags)}")
    totals = conn.execute("SELECT SUM(hail_count), SUM(wind_count), SUM(missing_count) FROM datasets "
                          "WHERE total_score > 0").fetchone()
    print(f"\nInstances across usable datasets: hail={totals[0]}, wind={totals[1]}, missing_shingles={totals[2]}")


def cmd_show(conn, dataset_id):
    row = get_row(conn, dataset_id)
    if not row:
        print("Dataset not found.")
        return
    print("\n--- Dataset Details ---\n")
    print(f"{'ID':<28}{row['id']}")
    for n, _, label in COLUMNS:
        val = row[n]
        if n == "quality_report" and val:
            print(f"{label}:")
            for k, v in json.loads(val).items():
                print(f"    {k:<28}{v}")
        else:
            print(f"{label:<28}{val}")


def cmd_edit(conn, dataset_id):
    row = get_row(conn, dataset_id)
    if not row:
        print("Dataset not found.")
        return
    rec = enter_record(dict(row))
    _, total = save_dataset(conn, rec, row["id"])
    print(f"Updated. New score: {total}/5")


def cmd_delete(conn, dataset_id):
    row = get_row(conn, dataset_id)
    if not row:
        print("Dataset not found.")
        return
    if ask(f"Delete '{row['name']}'? (y/n)", "n").lower() != "y":
        print("Cancelled.")
        return
    conn.execute("DELETE FROM datasets WHERE id = ?", (row["id"],))
    conn.commit()
    print("Dataset deleted.")


def cmd_rescore(conn):
    """Recompute every total after changing WEIGHTS or gates."""
    for row in conn.execute("SELECT * FROM datasets").fetchall():
        save_dataset(conn, {f: row[f] for f in SCORE_FIELDS} | {
            "license_ok": row["license_ok"], "annotation_type": row["annotation_type"]}, row["id"])
    print("All totals recomputed.")


def cmd_export(conn, out):
    rows = conn.execute("SELECT * FROM datasets ORDER BY total_score DESC").fetchall()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + [n for n, _, _ in COLUMNS if n != "quality_report"])
        for r in rows:
            w.writerow([r["id"]] + [r[n] for n, _, _ in COLUMNS if n != "quality_report"])
    print(f"Exported {len(rows)} datasets to {out}")


def menu(conn):
    options = {
        "1": ("Add dataset by hand", lambda: cmd_add(conn)),
        "2": ("Profile a dataset folder and add it", lambda: cmd_profile(
            conn, ask("Folder path"), save=True)),
        "3": ("View ranked datasets", lambda: cmd_list(conn)),
        "4": ("View dataset details", lambda: cmd_show(conn, ask("Dataset ID"))),
        "5": ("Edit dataset", lambda: cmd_edit(conn, ask("Dataset ID"))),
        "6": ("Delete dataset", lambda: cmd_delete(conn, ask("Dataset ID"))),
        "7": ("Recompute all scores", lambda: cmd_rescore(conn)),
        "8": ("Export to CSV", lambda: cmd_export(conn, ask("CSV path", "dataset_matrix.csv"))),
    }
    while True:
        print("\n==============================\n Roof Dataset Comparison Tool \n==============================")
        for k, (label, _) in options.items():
            print(f"{k}. {label}")
        print("9. Exit")
        choice = input("\nChoose an option: ").strip()
        if choice == "9":
            print("Goodbye.")
            return
        if choice in options:
            options[choice][1]()
        else:
            print("Invalid option.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("profile", help="scan a dataset folder")
    p.add_argument("path")
    p.add_argument("--name")
    p.add_argument("--save", action="store_true", help="add it to the matrix after scanning")
    p.add_argument("--json", action="store_true", help="print the raw report as JSON")
    sub.add_parser("list")
    s = sub.add_parser("show")
    s.add_argument("id")
    e = sub.add_parser("export")
    e.add_argument("out")
    sub.add_parser("rescore")
    args = ap.parse_args(argv)

    conn = connect()
    try:
        if args.cmd == "profile":
            if args.json:
                print(json.dumps(profile_dataset(args.path), indent=2))
            else:
                cmd_profile(conn, args.path, args.name, args.save)
        elif args.cmd == "list":
            cmd_list(conn)
        elif args.cmd == "show":
            cmd_show(conn, args.id)
        elif args.cmd == "export":
            cmd_export(conn, args.out)
        elif args.cmd == "rescore":
            cmd_rescore(conn)
        else:
            menu(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
