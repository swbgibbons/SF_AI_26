"""Draw the boxes from dataset/boxes.csv onto copies of the photos, one folder per class.

    python -m detect.review_boxes --out review/boxes

Each output file is named <class>__<photo id>.jpg. Fix a box by editing its row in boxes.csv
(coordinates are fractions of width and height: 0,0 is top-left, 1,1 is bottom-right).
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = {"wind": (0, 160, 255), "hail": (255, 200, 0), "missing_shingles": (255, 60, 60)}


def draw(image, boxes, font):
    d = ImageDraw.Draw(image)
    w, h = image.size
    for b in boxes:
        x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
        color = COLORS[b["class"]]
        d.rectangle([x1 * w, y1 * h, x2 * w - 1, y2 * h - 1], outline=color, width=4)
        label = b["class"] + (" (unsure)" if b.get("status") == "low" else "")
        tx, ty = x1 * w + 4, max(y1 * h + 4, 0)
        d.rectangle(d.textbbox((tx, ty), label, font=font), fill=color)
        d.text((tx, ty), label, fill="black", font=font)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--out", type=Path, default=Path("review/boxes"))
    args = parser.parse_args()
    font = ImageFont.load_default(size=16)
    by_path = defaultdict(list)
    for b in csv.DictReader(open(args.dataset / "boxes.csv", newline="")):
        by_path[b["path"]].append(b)
    for path, boxes in by_path.items():
        cls = boxes[0]["class"]
        out = args.out / cls / f"{cls}__{Path(path).stem.split('-')[-1]}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        draw(Image.open(args.dataset / path).convert("RGB"), boxes, font).save(out, quality=90)
    print(f"{len(by_path)} photos -> {args.out}")


if __name__ == "__main__":
    main()
