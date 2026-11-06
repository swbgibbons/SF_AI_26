#!/usr/bin/env python3
"""Count the licensing-safe roof photos per damage class and draw a pie chart.

Re-run it every time new photos arrive. It reads any mix of:

  --labels FILE.csv   a label sheet with one row per photo
                      (columns: image_id or file, and label or type; optional
                      labels = "hail;wind" for photos with more than one type,
                      source, license)
  --images DIR        a folder of photos. Three layouts are recognised:
                        * a Roboflow COCO export (_annotations.coco.json)
                        * a Roboflow / YOLO export (data.yaml + labels/*.txt)
                        * one folder per class (hail/, wind/, missing_shingles/,
                          no_damage/, unusable/ ... any depth, train/valid/test ok)

Every photo is checked against the V1 iStock blocklist before it is counted:
  * its SHA-256 is in v1_blocklist_sha256.txt (an exact copy of a V1 photo), or
  * its file name contains "istock" or one of the V1 iStock photo ids
    (catches renamed or resized copies such as hail__2234328050.jpg).
Blocked photos are listed in images.csv with the reason, never counted.
Exact duplicates across sources are counted once.

Outputs (in --out): counts.csv, images.csv, summary.md, pie.png, pie.svg

Examples
  python count_dataset.py --labels labels/rf_clean_parts5-6.csv --out out/
  python count_dataset.py --labels labels/rf_clean_parts5-6.csv \
      --images ~/Downloads/teammate_photos --images roboflow_export/ --out out/
  python count_dataset.py blocklist V1_DIR [V1_DIR ...] >> v1_blocklist_sha256.txt
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BLOCKLIST = HERE / "v1_blocklist_sha256.txt"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}

# Order is fixed so colours never move between runs.
CLASSES = ["hail", "wind", "missing_shingles", "no_damage", "unusable", "unlabeled"]
DISPLAY = {
    "hail": "Hail",
    "wind": "Wind",
    "missing_shingles": "Missing shingles",
    "no_damage": "No damage",
    "unusable": "Unusable",
    "unlabeled": "Not labeled yet",
}
# Pie slices: the classes a model trains on. Unusable / unlabeled are reported, not drawn.
PIE_CLASSES = ["hail", "wind", "missing_shingles", "no_damage"]
COLORS = {  # validated categorical slots 1-4 (light surface)
    "hail": "#2a78d6",
    "wind": "#eb6834",
    "missing_shingles": "#1baf7a",
    "no_damage": "#eda100",
}

ALIASES = {
    "hail": "hail", "hail_damage": "hail", "hailstrike": "hail", "hail_hit": "hail",
    "poten_hail": "hail", "hail_impact": "hail", "bruise": "hail",
    "wind": "wind", "wind_damage": "wind", "creased": "wind", "lifted": "wind",
    "lifted_shingle": "wind", "creased_shingle": "wind", "torn": "wind",
    "missing_shingles": "missing_shingles", "missing_shingle": "missing_shingles",
    "missing": "missing_shingles", "missing_broken": "missing_shingles",
    "broken": "missing_shingles", "broken_shingle": "missing_shingles",
    "missing/broken": "missing_shingles", "cracked": "missing_shingles",
    "none": "no_damage", "no_damage": "no_damage", "undamaged": "no_damage",
    "clean": "no_damage", "normal": "no_damage", "intact": "no_damage",
    "no_visible_damage": "no_damage",
    "unusable": "unusable", "bad": "unusable", "reject": "unusable",
    "unclear/unusable": "unusable", "unclear": "unusable",
}
SPLIT_NAMES = {"train", "valid", "val", "validation", "test", "images", "labels"}


def norm_class(name: str | None) -> str | None:
    if not name:
        return None
    key = re.sub(r"[\s\-]+", "_", name.strip().lower())
    if key in ALIASES:
        return ALIASES[key]
    for prefix in ("poten_hail", "no_damage", "hail", "wind", "missing", "unusable"):
        if key.startswith(prefix):
            return ALIASES.get(prefix, prefix if prefix != "missing" else "missing_shingles")
    return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- blocklist
class Blocklist:
    def __init__(self, path: Path | None):
        self.hashes: set[str] = set()
        self.ids: set[str] = set()
        if path and path.exists():
            for line in path.read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("id:"):
                    self.ids.add(line[3:].strip())
                elif re.fullmatch(r"[0-9a-f]{64}", line):
                    self.hashes.add(line)

    def reason(self, name: str, digest: str | None) -> str | None:
        if digest and digest in self.hashes:
            return "V1 photo (exact copy)"
        low = name.lower()
        if "istock" in low:
            return "V1 photo (iStock file name)"
        for tok in re.findall(r"(?<![0-9a-z])(\d{8,11})(?![0-9a-z])", low):
            if tok in self.ids:
                return f"V1 photo (iStock id {tok})"
        return None


def build_blocklist(dirs: list[str]) -> None:
    """Print sha256 lines and id: lines for every V1 photo under the given folders/zips."""
    hashes, ids = set(), set()
    for d in dirs:
        p = Path(d)
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
        for f in files:
            if f.suffix.lower() == ".zip":
                with zipfile.ZipFile(f) as z:
                    for n in z.namelist():
                        if Path(n).suffix.lower() in IMAGE_EXT:
                            hashes.add(hashlib.sha256(z.read(n)).hexdigest())
                            ids.update(re.findall(r"(?<!\d)(\d{8,11})(?!\d)", Path(n).name))
                continue
            if f.suffix.lower() in IMAGE_EXT or (not f.suffix and _looks_like_image(f)):
                hashes.add(sha256(f))
                ids.update(re.findall(r"istock(?:photo)?-(\d{8,11})", f.name.lower()))
                ids.update(re.findall(r"__(\d{8,11})\.", f.name))
    for h in sorted(hashes):
        print(h)
    for i in sorted(ids):
        print(f"id:{i}")


def _looks_like_image(f: Path) -> bool:
    with open(f, "rb") as fh:
        head = fh.read(12)
    return head.startswith(b"\xff\xd8\xff") or head.startswith(b"\x89PNG")


# ---------------------------------------------------------------- readers
def read_label_csv(path: Path) -> list[dict]:
    """A label sheet. Also reads the RF_Clean `image_labels.csv` that ships inside a
    teammate's export (columns image, damage_type, split, <type>_boxes_*): photos found
    next to it are hashed, so V1 copies and duplicates are caught there too."""
    rows = []
    on_disk = {f.name: f for f in _images_under(path.parent)}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
            name = r.get("file") or r.get("filename") or r.get("path") or r.get("image") or r.get("image_id") or ""
            multi = [norm_class(x) for x in re.split(r"[;|]", r.get("labels", "")) if x.strip()]
            primary = (norm_class(r.get("label") or r.get("type") or r.get("class") or r.get("damage_type"))
                       or (multi[0] if multi else None))
            boxes = Counter()
            for k, v in r.items():
                m = re.fullmatch(r"(.+?)_boxes(?:_.+)?", k)
                if m and norm_class(m.group(1)) and v.isdigit():
                    boxes[norm_class(m.group(1))] += int(v)
            disk = on_disk.get(Path(name).name)
            rows.append({
                "path": disk,
                "boxes": boxes,
                "image_id": r.get("image_id") or Path(name).stem,
                "in_training": r.get("in_training_set", ""),
                "file": name,
                "class": primary or "unlabeled",
                "all_classes": sorted({c for c in multi + [primary] if c}),
                "source": r.get("source") or (f"{path.parent.parent.name}/{path.name}" if path.parent.name == "dataset" else path.name),
                "license": r.get("license", ""),
                "sha256": r.get("sha256") or None,
                "labeled_by": r.get("labeled_by", ""),
                "split": r.get("split", ""),
            })
    return rows


def read_pending(path: Path) -> tuple[Counter, list[str]]:
    """Counts reported by teammates who haven't sent their photos yet: rows of class,count[,who]."""
    pending, who = Counter(), []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            r = {k.strip().lower(): (v or "").strip() for k, v in r.items() if k}
            c = norm_class(r.get("class") or r.get("label") or r.get("type"))
            if c and r.get("count", "").isdigit():
                pending[c] += int(r["count"])
                if r.get("who") and r["who"] not in who:
                    who.append(r["who"])
    return pending, who


def _images_under(d: Path):
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.suffix.lower() in IMAGE_EXT:
            yield f


def read_coco(d: Path) -> list[dict]:
    rows = []
    for ann_file in sorted(d.rglob("_annotations.coco.json")):
        coco = json.loads(ann_file.read_text())
        cats = {c["id"]: norm_class(c["name"]) for c in coco.get("categories", [])}
        per_img: dict[int, Counter] = {}
        for a in coco.get("annotations", []):
            c = cats.get(a["category_id"])
            if c:
                per_img.setdefault(a["image_id"], Counter())[c] += 1
        for im in coco.get("images", []):
            f = ann_file.parent / im["file_name"]
            cnt = per_img.get(im["id"], Counter())
            rows.append(_row_from_counts(f, cnt, f"Roboflow COCO: {d.name}"))
    return rows


def read_yolo(d: Path) -> list[dict]:
    yaml = next(iter(sorted(d.rglob("data.yaml"))), None)
    names: list[str] = []
    if yaml:
        txt = yaml.read_text()
        m = re.search(r"names:\s*\[(.*?)\]", txt, re.S)
        if m:
            names = [x.strip().strip("'\"") for x in m.group(1).split(",")]
        else:
            names = [x.strip().strip("'\"") for x in re.findall(r"^\s*(?:-|\d+:)\s*(.+)$", txt.split("names:", 1)[-1], re.M)]
    rows = []
    for f in _images_under(d):
        lbl = Path(str(f.parent).replace("images", "labels")) / (f.stem + ".txt")
        cnt = Counter()
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                if line.strip():
                    i = int(line.split()[0])
                    c = norm_class(names[i]) if i < len(names) else None
                    if c:
                        cnt[c] += 1
        rows.append(_row_from_counts(f, cnt, f"Roboflow YOLO: {d.name}"))
    return rows


def read_class_folders(d: Path) -> list[dict]:
    rows = []
    for f in _images_under(d):
        cls = None
        for part in reversed(f.relative_to(d).parts[:-1]):
            if part.lower() in SPLIT_NAMES:
                continue
            cls = norm_class(part)
            if cls:
                break
        cls = cls or norm_class(f.name.split("__")[0])  # e.g. hail__123.jpg
        rows.append({"image_id": f.stem, "file": str(f), "path": f, "class": cls or "unlabeled",
                     "all_classes": [cls] if cls else [], "source": f"folder: {d.name}",
                     "license": "", "sha256": None, "labeled_by": "", "boxes": Counter()})
    return rows


def _row_from_counts(f: Path, cnt: Counter, source: str) -> dict:
    # Photo-level class = the damage type with the most boxes; ties go to the rarer class order.
    if cnt:
        primary = max(cnt, key=lambda c: (cnt[c], -CLASSES.index(c)))
    else:
        primary = "no_damage"  # Roboflow image with no boxes = background / no damage
    return {"image_id": f.stem, "file": str(f), "path": f, "class": primary,
            "all_classes": sorted(cnt) or ["no_damage"], "source": source, "license": "",
            "sha256": None, "labeled_by": "", "boxes": cnt}


def read_images_dir(d: Path) -> list[dict]:
    if any(d.rglob("_annotations.coco.json")):
        return read_coco(d)
    if any(d.rglob("data.yaml")):
        return read_yolo(d)
    return read_class_folders(d)


# ---------------------------------------------------------------- core
def split_of(r) -> str:
    if r.get("split"):
        return r["split"]
    for part in reversed(Path(r["file"]).parts[:-1]):
        p = part.lower()
        if p in ("train", "test"):
            return p
        if p in ("valid", "val", "validation"):
            return "valid"
    return ""


def collect(label_files, image_dirs, blocklist: Blocklist):
    rows = []
    for p in label_files:
        rows += read_label_csv(Path(p))
    for d in image_dirs:
        rows += read_images_dir(Path(d).expanduser())
    seen: dict[str, str] = {}
    for r in rows:
        p = r.get("path")
        if p is not None and p.exists():
            r["sha256"] = sha256(p)
        r["excluded"] = blocklist.reason(Path(r["file"]).name or r["image_id"], r["sha256"])
        if not r["excluded"] and r["sha256"]:
            if r["sha256"] in seen:
                r["excluded"] = f"duplicate of {seen[r['sha256']]}"
            else:
                seen[r["sha256"]] = r["image_id"]
    return rows


def write_outputs(rows, out: Path, title: str, pending: Counter | None = None):
    pending = pending or Counter()
    out.mkdir(parents=True, exist_ok=True)
    kept = [r for r in rows if not r["excluded"]]
    excluded = [r for r in rows if r["excluded"]]
    counts = Counter(r["class"] for r in kept)
    boxes = Counter()
    for r in kept:
        if r["class"] not in ("unusable", "unlabeled"):  # boxes on photos left out of training don't count
            boxes.update(r.get("boxes") or {})
    multi = sum(1 for r in kept if len([c for c in r["all_classes"] if c in PIE_CLASSES[:3]]) > 1)
    pie_total = sum(counts[c] for c in PIE_CLASSES)
    pend_total = sum(pending[c] for c in PIE_CLASSES)
    grand = pie_total + pend_total

    with open(out / "counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "submitted_images", "pending_images", "total_images", "share_of_total", "boxes_submitted"])
        for c in CLASSES:
            t = counts[c] + pending[c]
            share = f"{100 * t / grand:.1f}%" if c in PIE_CLASSES and grand else ""
            w.writerow([c, counts[c], pending[c], t, share, boxes[c] or ""])
        w.writerow(["TOTAL_usable", pie_total, pend_total, grand, "100%", sum(boxes[c] for c in PIE_CLASSES) or ""])
        w.writerow(["excluded_v1_or_duplicate", len(excluded), "", "", "", ""])

    with open(out / "images.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file", "split", "class", "all_classes", "source", "license", "labeled_by", "sha256", "excluded"])
        for r in rows:
            w.writerow([r["image_id"], r["file"], split_of(r), r["class"], ";".join(r["all_classes"]), r["source"],
                        r["license"], r["labeled_by"], r["sha256"] or "", r["excluded"] or ""])

    lines = [f"# {title}", "", f"Generated {dt.date.today().isoformat()} by `count_dataset.py`.", "",
             "| Class | Submitted | Boxes | Pending | Total | Share |", "|---|---:|---:|---:|---:|---:|"]
    for c in PIE_CLASSES:
        t = counts[c] + pending[c]
        share = f"{100 * t / grand:.1f}%" if grand else ""
        lines.append(f"| {DISPLAY[c]} | {counts[c]} | {boxes[c] or ''} | {pending[c]} | {t} | {share} |")
    lines += [f"| **Usable total** | **{pie_total}** | {sum(boxes[c] for c in PIE_CLASSES) or ''} | **{pend_total}** | **{grand}** | |", "",
              "Submitted = photos received and checked by this script. Pending = counts teammates reported "
              "for photos not sent yet (not checked for V1 copies or duplicates).", "",
              f"- Unusable (kept out of training): {counts['unusable']}",
              f"- Not labeled yet: {counts['unlabeled']}",
              f"- Photos with more than one damage type: {multi}",
              f"- Excluded as V1 iStock or duplicate: {len(excluded)}"]
    by_src = Counter(r["source"] for r in kept)
    splits = Counter((split_of(r), r["class"]) for r in kept)
    split_names = [x for x in ("train", "valid", "test") if any(k[0] == x for k in splits)]
    split_lines = []
    if split_names:
        split_lines = ["", "| Class | " + " | ".join(split_names) + " |", "|---|" + "---:|" * len(split_names)]
        for c in PIE_CLASSES:
            split_lines.append(f"| {DISPLAY[c]} | " + " | ".join(str(splits[(x, c)]) for x in split_names) + " |")
        missing = [f"{DISPLAY[c]} in {x}" for c in PIE_CLASSES[:3] for x in split_names if counts[c] and not splits[(x, c)]]
        if missing:
            split_lines.append("\nWARNING, no photos for: " + ", ".join(missing))
        print("\n".join(split_lines[1:]))
    lines += split_lines
    lines += ["", "Sources counted:"] + [f"- {s}: {n}" for s, n in by_src.most_common()]
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    # The chart shows one total per damage type (photos in hand + reported by teammates);
    # the have / pending split stays in counts.csv and summary.md.
    combined = Counter({c: counts[c] + pending[c] for c in CLASSES})
    draw_pie(combined, Counter(), counts["unusable"] + counts["unlabeled"], len(excluded), out, title)
    return counts, pie_total, excluded


def _tint(hex_color: str, amount: float) -> str:
    """Mix a colour toward white (amount 0..1)."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(v + (255 - v) * amount):02x}" for v in rgb)


def draw_pie(counts, pending, not_drawn, excluded, out: Path, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipped the chart (pip install matplotlib)", file=sys.stderr)
        return
    import matplotlib.lines  # noqa: F401
    import matplotlib.patches  # noqa: F401

    has_pending = sum(pending[c] for c in PIE_CLASSES) > 0
    sub_total = sum(counts[c] for c in PIE_CLASSES)
    total = sub_total + sum(pending[c] for c in PIE_CLASSES)
    ink, muted, surface, rule = "#0b0b0b", "#52514e", "#fcfcfb", "#d6d5d0"
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["hatch.color"] = surface
    plt.rcParams["hatch.linewidth"] = 1.6
    fig = plt.figure(figsize=(11, 6.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax = fig.add_axes([0.01, 0.1, 0.46, 0.72])
    ax.set_facecolor(surface)
    ax.axis("off")
    if not total:
        ax.text(0.5, 0.5, "No usable photos yet", ha="center", va="center", color=muted, fontsize=16)
    else:
        # Each damage type is one slice: the solid part is photos we have, the striped,
        # lighter part is photos teammates have reported but not sent yet.
        vals, colors, hatches = [], [], []
        for c in PIE_CLASSES:
            for v, pend in ((counts[c], False), (pending[c], True)):
                if v:
                    vals.append(v)
                    colors.append(_tint(COLORS[c], 0.45) if pend else COLORS[c])
                    hatches.append("///" if pend else None)
        wedges, _ = ax.pie(vals, colors=colors, startangle=90, counterclock=False,
                           wedgeprops=dict(edgecolor=surface, linewidth=2.0))
        for w, h in zip(wedges, hatches):
            if h:
                w.set_hatch(h)
        ax.set_aspect("equal")

    # Table on the right. Every class is listed (zeros too) so rows never move between runs.
    x0, xs, xh, xt, xsh = 0.50, 0.525, 0.80, 0.885, 0.975
    top, step = 0.70, 0.098
    cols = [("Damage type", xs, "left")]
    if has_pending:
        cols += [("Have", xh, "right"), ("Total", xt, "right")]
    else:
        cols += [("Photos", xt, "right")]
    cols += [("Share", xsh, "right")]
    for label, x, ha in cols:
        fig.text(x, top + 0.07, label, fontsize=10, color=muted, ha=ha)
    for i, c in enumerate(PIE_CLASSES):
        y = top - i * step
        fig.patches.append(matplotlib.patches.FancyBboxPatch(
            (x0, y - 0.012), 0.017, 0.032, boxstyle="round,pad=0,rounding_size=0.004",
            transform=fig.transFigure, facecolor=COLORS[c], edgecolor="none"))
        fig.text(xs, y + 0.004, DISPLAY[c], fontsize=12, color=ink, va="center")
        t = counts[c] + pending[c]
        if has_pending:
            fig.text(xh, y + 0.004, f"{counts[c]:,}", fontsize=12, color=ink, va="center", ha="right")
        fig.text(xt, y + 0.004, f"{t:,}", fontsize=12, color=ink, va="center", ha="right", fontweight="bold")
        fig.text(xsh, y + 0.004, f"{100 * t / total:.1f}%" if total else "-", fontsize=12, color=muted,
                 va="center", ha="right")
    y = top - len(PIE_CLASSES) * step + 0.025
    fig.add_artist(matplotlib.lines.Line2D([x0, xsh], [y + 0.03, y + 0.03], color=rule, lw=0.8))
    fig.text(xs, y - 0.008, "Usable total", fontsize=12, color=ink, va="center")
    if has_pending:
        fig.text(xh, y - 0.008, f"{sub_total:,}", fontsize=12, color=ink, va="center", ha="right")
    fig.text(xt, y - 0.008, f"{total:,}", fontsize=12, color=ink, va="center", ha="right", fontweight="bold")
    if has_pending:
        ky = y - 0.1
        fig.patches.append(matplotlib.patches.Rectangle((x0, ky - 0.012), 0.017, 0.032, transform=fig.transFigure,
                                                        facecolor="#8a8984", edgecolor="none"))
        fig.text(xs, ky + 0.004, "Have: received and checked", fontsize=10, color=muted, va="center")
        r = matplotlib.patches.Rectangle((x0 + 0.22, ky - 0.012), 0.017, 0.032, transform=fig.transFigure,
                                         facecolor=_tint("#8a8984", 0.45), edgecolor="none", hatch="///")
        fig.patches.append(r)
        fig.text(x0 + 0.245, ky + 0.004, "Pending: reported, not sent yet", fontsize=10, color=muted, va="center")

    fig.suptitle(title, x=0.04, y=0.955, ha="left", fontsize=17, fontweight="bold", color=ink)
    if has_pending:
        sub = f"{total:,} usable photos: {sub_total:,} in hand, {total - sub_total:,} reported by teammates (V1 iStock photos removed)"
    else:
        sub = f"{total:,} usable photos (V1 iStock photos removed)"
    fig.text(0.04, 0.88, sub, ha="left", fontsize=11.5, color=muted)
    note = []
    if not_drawn:
        note.append(f"{not_drawn} unusable / unlabeled photos not shown")
    if excluded:
        note.append(f"{excluded} V1 or duplicate photos excluded")
    note.append(f"Updated {dt.date.today().strftime('%b %d, %Y').replace(' 0', ' ')}")
    fig.text(0.04, 0.035, " · ".join(note), ha="left", fontsize=9.5, color=muted)
    fig.savefig(out / "pie.png", facecolor=surface)
    fig.savefig(out / "pie.svg", facecolor=surface)
    plt.close(fig)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["blocklist"]:
        build_blocklist(argv[1:])
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", action="append", default=[], help="label sheet CSV (repeatable)")
    ap.add_argument("--images", action="append", default=[], help="photo folder or Roboflow export (repeatable)")
    ap.add_argument("--pending", help="CSV of class,count[,who] reported for photos not sent yet")
    ap.add_argument("--blocklist", default=str(DEFAULT_BLOCKLIST), help="V1 blocklist file")
    ap.add_argument("--out", default="out", help="output folder")
    ap.add_argument("--title", default="Roof damage dataset by damage type")
    a = ap.parse_args(argv)
    if not a.labels and not a.images:
        ap.error("give at least one --labels or --images")
    rows = collect(a.labels, a.images, Blocklist(Path(a.blocklist)))
    pending = read_pending(Path(a.pending))[0] if a.pending else Counter()
    counts, total, excluded = write_outputs(rows, Path(a.out), a.title, pending)
    print(f"{'':<28}{'have':>6}{'pending':>9}{'total':>7}")
    for c in CLASSES:
        if counts[c] or pending[c]:
            print(f"{DISPLAY[c]:<28}{counts[c]:>6}{pending[c]:>9}{counts[c] + pending[c]:>7}")
    pt = sum(pending[c] for c in PIE_CLASSES)
    print(f"{'Usable total':<28}{total:>6}{pt:>9}{total + pt:>7}")
    print(f"{'Excluded (V1 / duplicate)':<28}{len(excluded):>6}")
    print(f"Wrote {a.out}/pie.png, pie.svg, counts.csv, images.csv, summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
