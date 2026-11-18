"""Train RF-DETR on the RoofInsight roof damage dataset (wind / hail / missing_shingles).

Works the same on Google Colab and on a plain lab or cloud GPU machine:

    python train.py --dataset /path/to/roof_damage_rfdetr.zip --output /path/to/runs/run-001

* ``--dataset`` may be the dataset folder (train/ valid/ test/) or the zip itself; a zip is
  extracted once to ``--extract-to`` (default: next to the zip) and reused afterwards.
* A resumable checkpoint (``last.ckpt``) is written to ``--output`` after every epoch, plus a
  kept archive copy every ``--checkpoint-interval`` epochs. Running the same command again
  with the same ``--output`` resumes from the newest checkpoint automatically.
* On Colab, point ``--output`` at a Google Drive folder so checkpoints survive a disconnect.
"""

import argparse
import json
import math
import platform
import sys
import time
import zipfile
from pathlib import Path


# Same ids as the v1 RF-DETR head. Changing this order breaks compatibility with v1 checkpoints.
CLASS_NAMES = {0: "wind", 1: "hail", 2: "missing_shingles"}
MODELS = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}
SPLITS = ("train", "valid", "test")


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def batch_size_value(value):
    if value == "auto":
        return value
    return positive_integer(value)


def prepare_dataset(path, extract_to=None):
    """Return a dataset folder; extract ``path`` first if it is a zip."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        return find_dataset_root(path)
    if not (path.is_file() and zipfile.is_zipfile(path)):
        raise ValueError(f"Dataset not found (expected a folder or .zip): {path}")
    target = Path(extract_to).expanduser().resolve() if extract_to else path.parent / path.stem
    marker = target / ".extracted_ok"
    if not marker.is_file():
        print(f"Extracting {path.name} to {target} ...", flush=True)
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                destination = (target / member).resolve()
                if not destination.is_relative_to(target):
                    raise ValueError(f"Unsafe path in zip: {member}")
            archive.extractall(target)
        marker.write_text(str(path), encoding="utf-8")
    return find_dataset_root(target)


def find_dataset_root(folder):
    """Accept the folder itself or a single wrapper folder (the zip contains roof_damage_rfdetr/)."""
    if (folder / "train" / "_annotations.coco.json").is_file():
        return folder
    candidates = [child for child in folder.iterdir()
                  if child.is_dir() and (child / "train" / "_annotations.coco.json").is_file()]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"No train/_annotations.coco.json found in {folder} or one folder below it.")


def validate_dataset(dataset):
    summary = {}
    for split in SPLITS:
        directory = dataset / split
        if split == "test" and not directory.exists():
            continue
        annotation_file = directory / "_annotations.coco.json"
        if not annotation_file.is_file():
            raise ValueError(f"Missing {annotation_file}. Export bounding-box annotations as COCO JSON.")
        with annotation_file.open(encoding="utf-8") as source:
            data = json.load(source)
        categories = data.get("categories", [])
        mapping = {category["id"]: category["name"] for category in categories}
        if mapping != CLASS_NAMES or len(categories) != len(CLASS_NAMES):
            raise ValueError(f"{split}: expected exactly 0=wind, 1=hail, 2=missing_shingles; found {mapping}.")
        images = data.get("images", [])
        if not images:
            raise ValueError(f"{split}: no image records.")
        image_by_id = {}
        for record in images:
            image_id = record["id"]
            if image_id in image_by_id:
                raise ValueError(f"{split}: duplicate image ID {image_id}.")
            path = (directory / record["file_name"]).resolve()
            if not path.is_relative_to(directory.resolve()) or not path.is_file():
                raise ValueError(f"{split}: missing image or path outside its split: {record['file_name']}.")
            for dimension in ("width", "height"):
                value = record.get(dimension)
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{split}: invalid image {dimension} for {image_id}.")
            image_by_id[image_id] = record
        annotations = data.get("annotations", [])
        annotation_ids = set()
        boxes_per_class = {name: 0 for name in CLASS_NAMES.values()}
        for annotation in annotations:
            annotation_id = annotation["id"]
            if annotation_id in annotation_ids:
                raise ValueError(f"{split}: duplicate annotation ID {annotation_id}.")
            annotation_ids.add(annotation_id)
            image = image_by_id.get(annotation["image_id"])
            if image is None or annotation["category_id"] not in CLASS_NAMES:
                raise ValueError(f"{split}: annotation {annotation_id} references an unknown image or class.")
            box = annotation.get("bbox", [])
            if len(box) != 4 or any(type(value) not in (int, float) or not math.isfinite(value) for value in box):
                raise ValueError(f"{split}: each box must have four finite numbers: x, y, width, height.")
            x, y, width, height = box
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(f"{split}: invalid bounding box {box}.")
            if x + width > image["width"] + 0.01 or y + height > image["height"] + 0.01:
                raise ValueError(f"{split}: bounding box extends outside image {image['id']}.")
            boxes_per_class[CLASS_NAMES[annotation["category_id"]]] += 1
        if split in ("train", "valid"):
            missing = [name for name, count in boxes_per_class.items() if count == 0]
            if missing:
                raise ValueError(f"{split}: include annotated examples of every class; none for {missing}.")
        boxed_images = {annotation["image_id"] for annotation in annotations}
        summary[split] = {
            "images": len(images),
            "boxes": len(annotations),
            "boxes_per_class": boxes_per_class,
            "no_damage_images": len(images) - len(boxed_images),
        }
    return summary


def find_resume_checkpoint(output):
    """Newest Lightning checkpoint in the run folder, or None."""
    if not output.is_dir():
        return None
    checkpoints = [path for path in output.glob("*.ckpt") if path.is_file()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def resolve_resume(args):
    if args.resume == "none":
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError(f"{args.output} is not empty and --resume none was given. "
                             "Use a new --output folder, or --resume auto to continue that run.")
        return None
    if args.resume == "auto":
        checkpoint = find_resume_checkpoint(args.output)
        if checkpoint is None and args.output.exists() and any(
                path.name != "run_info.json" for path in args.output.iterdir()):
            raise ValueError(f"{args.output} has files but no .ckpt to resume from. Use a new --output folder.")
        return checkpoint
    checkpoint = Path(args.resume).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Resume checkpoint not found: {checkpoint}")
    return checkpoint


def write_run_info(args, dataset, summary, resume):
    from importlib import metadata
    import torch
    info_path = args.output / "run_info.json"
    history = []
    if info_path.is_file():
        try:
            history = json.loads(info_path.read_text(encoding="utf-8")).get("launches", [])
        except (ValueError, OSError):
            history = []
    history.append({
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": sys.argv,
        "resumed_from": str(resume) if resume else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "rfdetr": metadata.version("rfdetr"),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    info = {
        "dataset": str(dataset),
        "classes": CLASS_NAMES,
        "splits": summary,
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "resolution": args.resolution,
        "pretrain_weights": str(args.pretrain) if args.pretrain else "RF-DETR default (COCO)",
        "launches": history,
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def run_training(args, dataset, summary, resume):
    try:
        import torch
        import rfdetr
    except ImportError as error:
        raise RuntimeError('Training packages are not installed. Run: pip install "rfdetr[train]==1.11.2"') from error
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No usable NVIDIA GPU was found by PyTorch. On Colab: Runtime > Change runtime type > GPU. "
                           "CPU is only used with an explicit --device cpu.")
    if args.device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {properties.name}, {properties.total_memory / 2**30:.1f} GB", flush=True)
    model_kwargs = {"device": args.device}
    if args.pretrain:
        model_kwargs["pretrain_weights"] = str(args.pretrain)
    if args.resolution:
        model_kwargs["resolution"] = args.resolution
    model = getattr(rfdetr, MODELS[args.model])(**model_kwargs)
    args.output.mkdir(parents=True, exist_ok=True)
    write_run_info(args, dataset, summary, resume)
    if resume:
        print(f"Resuming from {resume}", flush=True)
    train_kwargs = dict(
        dataset_dir=str(dataset),
        output_dir=str(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
        class_names=[CLASS_NAMES[index] for index in sorted(CLASS_NAMES)],
        early_stopping=args.early_stopping,
        early_stopping_patience=args.patience,
        run_test="test" in summary,
        seed=args.seed,
        device=args.device,
        progress_bar="tqdm",
    )
    if resume:
        train_kwargs["resume"] = str(resume)
    model.train(**train_kwargs)
    print(f"Training finished. Best weights: {args.output / 'checkpoint_best_total.pth'}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Dataset folder (train/ valid/ test/) or roof_damage_rfdetr.zip")
    parser.add_argument("--extract-to", type=Path,
                        help="Where to extract a zipped dataset (default: next to the zip). On Colab use /content/data")
    parser.add_argument("--output", type=Path, help="Run / checkpoint folder. Re-use it to resume.")
    parser.add_argument("--model", choices=tuple(MODELS), default="medium",
                        help="RF-DETR size (default medium; nano matches the v1 model)")
    parser.add_argument("--epochs", type=positive_integer, default=50)
    parser.add_argument("--batch-size", type=batch_size_value, default=4,
                        help="Images per GPU step, or 'auto' to size it to GPU memory (default 4)")
    parser.add_argument("--grad-accum-steps", type=positive_integer, default=4,
                        help="Effective batch = batch-size x grad-accum-steps (default 4 -> 16)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=positive_integer,
                        help="Square input size; must be divisible by 32 (nano/small/medium) or 56 (base). "
                             "Default: the model's own")
    parser.add_argument("--pretrain", type=Path,
                        help="Start from these weights instead of RF-DETR's COCO weights, e.g. the v1 "
                             "checkpoint_best_total.pth (must be the same --model size)")
    parser.add_argument("--resume", default="auto",
                        help="'auto' (default): continue from the newest .ckpt in --output if there is one; "
                             "'none': always start fresh; or a path to a .ckpt")
    parser.add_argument("--checkpoint-interval", type=positive_integer, default=10,
                        help="Keep an archive checkpoint every N epochs (last.ckpt is still written every epoch). "
                             "Minimum 2.")
    parser.add_argument("--early-stopping", action="store_true", help="Stop when validation mAP stops improving")
    parser.add_argument("--patience", type=positive_integer, default=10, help="Early-stopping patience in epochs")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--check-only", action="store_true", help="Only validate the dataset; no training")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.checkpoint_interval < 2:
            # RF-DETR only writes the every-epoch last.ckpt when the archive interval is not 1.
            raise ValueError("--checkpoint-interval must be at least 2 (last.ckpt is already saved every epoch).")
        dataset = prepare_dataset(args.dataset, args.extract_to)
        summary = validate_dataset(dataset)
        print(json.dumps({"dataset": str(dataset), "classes": CLASS_NAMES, "splits": summary}, indent=2))
        if args.check_only:
            print("Dataset checks passed. No model was downloaded or trained.")
            return
        if args.output is None:
            raise ValueError("Training requires --output (the run / checkpoint folder).")
        args.output = args.output.expanduser().resolve()
        if args.output.is_relative_to(dataset) or dataset.is_relative_to(args.output):
            raise ValueError("Keep training results and the dataset in separate folders.")
        if args.output.exists() and not args.output.is_dir():
            raise ValueError(f"--output is a file, not a folder: {args.output}")
        if args.pretrain:
            args.pretrain = args.pretrain.expanduser().resolve()
            if not args.pretrain.is_file():
                raise ValueError(f"Pretrain weights not found: {args.pretrain}")
        resume = resolve_resume(args)
        run_training(args, dataset, summary, resume)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(1, f"Training setup error: {error}\n")


if __name__ == "__main__":
    main()
