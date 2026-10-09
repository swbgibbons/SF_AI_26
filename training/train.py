"""Validate a hail/wind COCO dataset and launch a separate RF-DETR training run."""

import argparse
import json
import math
from pathlib import Path


CLASS_NAMES = {0: "wind", 1: "hail"}


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def validate_dataset(dataset):
    summary = {}
    for split in ("train", "valid", "test"):
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
        if mapping != CLASS_NAMES or len(categories) != 2:
            raise ValueError(f"{split}: expected exactly 0=wind and 1=hail; found {mapping}.")
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
        for annotation in annotations:
            annotation_id = annotation["id"]
            if annotation_id in annotation_ids:
                raise ValueError(f"{split}: duplicate annotation ID {annotation_id}.")
            annotation_ids.add(annotation_id)
            image = image_by_id.get(annotation["image_id"])
            if image is None or annotation["category_id"] not in CLASS_NAMES:
                raise ValueError(f"{split}: annotation references an unknown image or class.")
            box = annotation.get("bbox", [])
            if len(box) != 4 or any(type(value) not in (int, float) or not math.isfinite(value) for value in box):
                raise ValueError(f"{split}: each box must have four finite numbers: x, y, width, height.")
            x, y, width, height = box
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(f"{split}: invalid bounding box {box}.")
            if x + width > image["width"] + 0.01 or y + height > image["height"] + 0.01:
                raise ValueError(f"{split}: bounding box extends outside image {image['id']}.")
        if split in ("train", "valid"):
            represented = {annotation["category_id"] for annotation in annotations}
            if represented != set(CLASS_NAMES):
                raise ValueError(f"{split}: include annotated examples of both wind and hail.")
        summary[split] = {"images": len(images), "boxes": len(annotations)}
    return summary


def run_training(args):
    try:
        import torch
        from rfdetr import RFDETRNano, RFDETRSmall
    except ImportError as error:
        raise RuntimeError("Training tools are not installed. Follow training/README.md in a separate environment.") from error
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No usable NVIDIA GPU was found by PyTorch. Check the GPU setup; CPU is not selected automatically.")
    models = {"nano": RFDETRNano, "small": RFDETRSmall}
    model = models[args.model](device=args.device)
    model.train(
        dataset_dir=str(args.dataset),
        output_dir=str(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=0,
        seed=42,
        device=args.device,
        run_test=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", choices=("nano", "small"), default="nano")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=positive_integer, default=50)
    parser.add_argument("--batch-size", type=positive_integer, default=1)
    parser.add_argument("--grad-accum-steps", type=positive_integer, default=1)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    args.dataset = args.dataset.resolve()
    try:
        summary = validate_dataset(args.dataset)
        print(json.dumps({"dataset": str(args.dataset), "classes": CLASS_NAMES, "splits": summary}, indent=2))
        if args.check_only:
            print("Dataset structure checks passed. No model was downloaded or trained.")
            return
        if args.output is None:
            raise ValueError("Training requires --output pointing to a new or empty run directory.")
        args.output = args.output.resolve()
        if args.output.is_relative_to(args.dataset) or args.dataset.is_relative_to(args.output):
            raise ValueError("Keep training results and the dataset in separate directories.")
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError("Output must be a new or empty directory; existing results will not be overwritten.")
        run_training(args)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(1, f"Training setup error: {error}\n")


if __name__ == "__main__":
    main()
