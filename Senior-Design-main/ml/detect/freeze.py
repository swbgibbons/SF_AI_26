"""Freeze the working labels (dataset/manifest.csv, dataset/boxes.csv) as a new version.

    python -m detect.freeze v2

Copies both files to dataset/versions/v2/ and writes SHA256SUMS covering them and every photo
they reference. Refuses to overwrite an existing version, and refuses if a photo that an earlier
version used has changed, if a test photo from an earlier version moved to another split, or if
a photo marked severe is also labeled as wind, hail or missing_shingles.
"""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from detect.build_dataset import check_severe, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    root = args.dataset
    out = root / "versions" / args.version
    if out.exists():
        raise SystemExit(f"{out} already exists; frozen versions are never overwritten.")
    rows = list(csv.DictReader(open(root / "manifest.csv", newline="")))
    check_severe(rows)
    split_of = {r["path"]: r["split"] for r in rows}
    for old in sorted((root / "versions").iterdir()):
        verify(root, old.name)
        for r in csv.DictReader(open(old / "manifest.csv", newline="")):
            if r["split"] == "test" and split_of.get(r["path"], "test") != "test":
                raise SystemExit(f"{r['path']} was a {old.name} test photo; it must stay in test.")
    out.mkdir(parents=True)
    for name in ("manifest.csv", "boxes.csv"):
        shutil.copy(root / name, out / name)
    files = [f"versions/{args.version}/manifest.csv", f"versions/{args.version}/boxes.csv"]
    files += sorted({r["path"] for r in rows})
    lines = [f"{hashlib.sha256((root / f).read_bytes()).hexdigest()}  {f}" for f in files]
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"froze {len(rows)} photos as {args.version}")


if __name__ == "__main__":
    main()
