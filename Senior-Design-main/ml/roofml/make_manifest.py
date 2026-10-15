"""
Create or update a manifest and assign group-wise train/val/test splits.

    # from folders images/damaged, images/undamaged, images/unusable
    python -m roofml.make_manifest --images-root data/images --out data/manifest.csv

    # add splits to a hand-written manifest (keeps any split already set)
    python -m roofml.make_manifest --manifest data/manifest.csv --out data/manifest.csv
"""

import argparse

from .data import assign_splits, manifest_from_folders, read_manifest, summarize, write_manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images-root", help="folder with damaged/ undamaged/ unusable/ subfolders")
    src.add_argument("--manifest", help="existing manifest CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rows = manifest_from_folders(args.images_root) if args.images_root else read_manifest(args.manifest)
    rows = assign_splits(rows, val=args.val, test=args.test, seed=args.seed)
    write_manifest(rows, args.out)
    print(f"wrote {len(rows)} rows to {args.out}\n{summarize(rows)}")


if __name__ == "__main__":
    main()
