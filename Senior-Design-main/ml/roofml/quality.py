"""
Rule-based image quality gate, run before the model.

Catches photos that are technically unusable (too small, blurry, too dark or
washed out) without any training. Photos that pass can still be "unusable" for
content reasons (not a roof, wrong roof type, wide scenery); the model's
"unusable" class handles those.

CLI: python -m roofml.quality path/to/folder_or_image [...]
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class QualityConfig:
    # Shortest side in pixels. Stock previews are 280-460px (wide crops go below 320);
    # real drone photos are 3000px+. Raise this once you train on full-size photos.
    min_short_side: int = 256
    # Variance of the Laplacian, measured after resizing the long side to 512px.
    # Lower = blurrier. Tune on your own data: print the scores, look at the photos.
    min_sharpness: float = 60.0
    # Fraction of pixels that are near-black / near-white.
    max_dark_fraction: float = 0.60
    max_bright_fraction: float = 0.60


@dataclass
class QualityReport:
    usable: bool
    reasons: list = field(default_factory=list)
    width: int = 0
    height: int = 0
    sharpness: float = 0.0
    dark_fraction: float = 0.0
    bright_fraction: float = 0.0


def load_image(path) -> Image.Image:
    """Open an image upright (applies EXIF rotation) as RGB."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def _laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = (
        -4 * g[1:-1, 1:-1]
        + g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
    )
    return float(lap.var())


def check_quality(image: Image.Image, config: QualityConfig = QualityConfig()) -> QualityReport:
    w, h = image.size
    small = image.copy()
    small.thumbnail((512, 512))
    gray = np.asarray(small.convert("L"))

    report = QualityReport(
        usable=True,
        width=w,
        height=h,
        sharpness=round(_laplacian_variance(gray), 1),
        dark_fraction=round(float((gray < 20).mean()), 3),
        bright_fraction=round(float((gray > 245).mean()), 3),
    )
    if min(w, h) < config.min_short_side:
        report.reasons.append(f"low_resolution ({w}x{h})")
    if report.sharpness < config.min_sharpness:
        report.reasons.append(f"blurry (sharpness {report.sharpness})")
    if report.dark_fraction > config.max_dark_fraction:
        report.reasons.append("too_dark")
    if report.bright_fraction > config.max_bright_fraction:
        report.reasons.append("overexposed")
    report.usable = not report.reasons
    return report


def iter_images(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        else:
            yield p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--min-short-side", type=int, default=QualityConfig.min_short_side)
    ap.add_argument("--min-sharpness", type=float, default=QualityConfig.min_sharpness)
    args = ap.parse_args(argv)
    config = QualityConfig(min_short_side=args.min_short_side, min_sharpness=args.min_sharpness)

    for path in iter_images(args.paths):
        try:
            report = check_quality(load_image(path), config)
            print(json.dumps({"image": str(path), **asdict(report)}))
        except Exception as e:  # corrupt or unreadable file
            print(json.dumps({"image": str(path), "usable": False, "reasons": [f"unreadable ({e})"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
