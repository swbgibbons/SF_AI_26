"""Train RF-DETR on the same COCO export detect.build_dataset writes, for comparison with YOLO.

    python -m detect.train_rfdetr --data data/det-v1/coco --size nano --name rfdetr-n-v1

Runs on CPU when no GPU is found (slow; use a GPU for real runs).
Weights land in runs/rfdetr/<name>/checkpoint_best_total.pth.
"""

import argparse
from pathlib import Path

import torch

SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "base": "RFDETRBase"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/det/coco")
    parser.add_argument("--size", default="nano", choices=SIZES)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--name", default="rfdetr-v1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    import rfdetr

    model = getattr(rfdetr, SIZES[args.size])()
    model.train(
        dataset_dir=args.data, epochs=args.epochs, batch_size=args.batch, grad_accum_steps=args.grad_accum,
        lr=args.lr, output_dir=str(Path("runs/rfdetr", args.name).resolve()), seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu", num_workers=2,
    )


if __name__ == "__main__":
    main()
