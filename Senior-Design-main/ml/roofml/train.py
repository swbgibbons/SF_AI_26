"""
Fine-tune a pretrained image classifier on damaged / undamaged / unusable.

    python -m roofml.train --manifest data/manifest.csv --images-root data/images \
        --model convnext_tiny --epochs 20 --out runs/convnext_tiny

Writes to --out: best.pt (best validation macro-F1), metrics.json, and
test_metrics.json when --eval-test is given. Only evaluate on test once you
have stopped tuning.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import timm
from torch.utils.data import DataLoader

from . import CLASSES
from .data import RoofDataset, assign_splits, build_transforms, read_manifest, summarize
from .metrics import classification_report, format_confusion


def build_model(name, pretrained=True):
    return timm.create_model(name, pretrained=pretrained, num_classes=len(CLASSES))


def class_weights(rows):
    """Inverse-frequency weights so a rare class (e.g. unusable) isn't ignored."""
    counts = Counter(r["label"] for r in rows)
    w = [len(rows) / (len(CLASSES) * counts[c]) if counts[c] else 0.0 for c in CLASSES]
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += y.tolist()
    return classification_report(y_true, y_pred)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="convnext_tiny",
                    help="any timm model name, e.g. resnet50, efficientnet_b0, convnext_tiny")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-test", action="store_true")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = assign_splits(read_manifest(args.manifest), seed=args.seed)
    print(summarize(rows))
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if not split["train"] or not split["val"]:
        raise SystemExit("Need images in both train and val splits; add more data.")

    def loader(s, train):
        ds = RoofDataset(split[s], args.images_root, build_transforms(args.image_size, train))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.workers)

    train_loader, val_loader = loader("train", True), loader("val", False)

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(split["train"]).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": total / len(split["train"]),
                        "val_macro_f1": val["macro_f1"], "val_damaged_recall": val["damaged_recall"]})
        print(f"epoch {epoch:3d}  loss {history[-1]['train_loss']:.4f}  val macro-F1 {val['macro_f1']:.3f}"
              f"  damaged recall {val['damaged_recall']:.3f}  ({time.time() - t0:.0f}s)")
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save({"model": args.model, "classes": CLASSES, "image_size": args.image_size,
                        "state_dict": model.state_dict(), "epoch": epoch, "val": val}, out / "best.pt")

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    val = evaluate(model, val_loader, device)
    print(f"\nbest epoch {ckpt['epoch']} validation:\n{format_confusion(val['confusion_matrix']['values'])}")
    json.dump({"args": vars(args), "best_epoch": ckpt["epoch"], "val": val, "history": history},
              open(out / "metrics.json", "w"), indent=2)

    if args.eval_test and split["test"]:
        test = evaluate(model, loader("test", False), device)
        print(f"\ntest:\n{format_confusion(test['confusion_matrix']['values'])}")
        json.dump(test, open(out / "test_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
