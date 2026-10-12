"""Train the damage detector (YOLO) on the dataset built by detect.build_dataset.

    python -m detect.train --data data/det/yolo/data.yaml --name v1

Runs on CPU when no GPU is found. Weights land in runs/detect/<name>/weights/best.pt.
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/det/yolo/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="pretrained starting weights")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="v1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = 0 if torch.cuda.is_available() else "cpu"
    YOLO(args.model).train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=device,
        project=str(Path("runs/detect").resolve()), name=args.name, exist_ok=True, seed=args.seed, patience=50,
        # Roofs photographed from a drone have no fixed "up", so vertical flips are fair game.
        flipud=0.5, fliplr=0.5, degrees=10, mosaic=1.0, close_mosaic=10, workers=2, plots=True,
    )


if __name__ == "__main__":
    main()
