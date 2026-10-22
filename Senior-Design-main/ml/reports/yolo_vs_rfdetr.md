# YOLO vs RF-DETR on the v1 roof photos

**Verdict: RF-DETR (Nano) beat YOLO11n on this project's data in both training runs, mostly by
outlining damage better, and the gap matters most for missing shingles. Use RF-DETR as the main
detector from v2 on.** The backend already runs an RF-DETR ONNX model, so this also removes a
mismatch between the model we train and the one the website serves.

## Setup

Same frozen dataset (`dataset/versions/v1`), same train/val/test split, same scorer
(`detect/compare.py`: pycocotools box mAP plus the photo-level rule from `detect/evaluate.py`).
Both models start from their public pretrained weights. Two training runs each (seed 0 and 1), CPU only.

| | YOLO11n (current v1) | RF-DETR Nano |
|---|---|---|
| Training | 60 rounds, about 30 min on 4 CPU cores | 40 rounds, about 70 min on 4 CPU cores |
| Size | 2.6M parameters | 30M parameters |
| CPU time per photo | about 40 ms | about 145 ms |
| License | AGPL-3.0 (closed or commercial use needs an Ultralytics license) | Apache 2.0 |

## Results on the 32 held-out test photos

| | YOLO seed 0 | YOLO seed 1 | RF-DETR seed 0 | RF-DETR seed 1 |
|---|---|---|---|---|
| Box mAP50 | 0.18 | 0.19 | **0.27** | **0.24** |
| Box mAP50-95 | 0.09 | 0.11 | **0.15** | **0.14** |
| Wind AP50 | 0.48 | 0.41 | 0.53 | 0.44 |
| Missing shingles AP50 | 0.08 | 0.16 | **0.28** | **0.28** |
| Hail AP50 | 0 | 0 | 0 | 0 |
| Right damage type (conf 0.25) | 19/29 | 18/29 | 20/29 | 22/29 |
| Damage found at all (conf 0.25) | 21/29 | 23/29 | 27/29 | 26/29 |
| Right damage type (conf 0.5) | 11/29 | 8/29 | 16/29 | 10/29 |

Validation photos (33, also used to pick each model's best round, so slightly optimistic for both)
show the same order: mAP50 0.25 / 0.22 for YOLO against 0.34 / 0.35 for RF-DETR, damage type
16-17/29 against 24/29. Full numbers: `reports/compare_v1_*/metrics.json`.

## How sure is this

- **Box quality (mAP): a real gap.** RF-DETR is ahead on both splits and both seeds, by more than
  the seed-to-seed swing. Missing-shingle boxes are about twice as good.
- **Damage-type accuracy: a lead, but small.** 1 to 4 more photos out of 29. With this few test
  photos, that alone would not settle it.
- **Hail: no answer yet.** Both models score 0 because there is 1 hail photo to train on.
- Seed 1 of RF-DETR was stopped at round 37 of 40 (time limit); its best checkpoint was used.

## Why it should hold as the dataset grows

Roboflow's published fine-tuning benchmark (RF100-VL, 100 real-world datasets) shows the same
direction: RF-DETR Nano 57.7 AP50-95 against 55.3 for YOLO11n and 52.0 for YOLO26n, and the larger
RF-DETR sizes stay ahead of the matching YOLO sizes. These are the RF-DETR authors' own numbers.
RF-DETR's DINOv2 backbone is the reason it transfers well to unusual domains like roof damage
from a small dataset. YOLO's advantages are speed and size, which matter on a phone or drone
but not for a backend that scores one uploaded photo at a time.

Source: https://rfdetr.roboflow.com/latest/learn/benchmarks/

## Costs of switching

- Training is about 2-3x slower per round. A GPU makes both fast; on CPU a 40-round run is about an hour.
- Inference is about 3.5x slower on CPU (about 0.15 s per photo). Fine for the website.
- The YOLO demo page and `detect/evaluate.py` are YOLO-specific; `detect/compare.py` scores either.

## Reproduce

The trained RF-DETR seed-0 weights are in the project files at `models/rfdetr-n-v1/` (kept out of git).

```bash
pip install "rfdetr[train]" ultralytics
python -m detect.build_dataset --version v1 --out data/det-v1
python -m detect.train --data data/det-v1/yolo/data.yaml --name v1
python -m detect.train_rfdetr --data data/det-v1/coco --size nano --epochs 40 --name rfdetr-n-v1
python -m detect.compare --version v1 --split test --out reports/compare_v1_test \
    --model yolo:runs/detect/v1/weights/best.pt \
    --model rfdetr-nano:runs/rfdetr/rfdetr-n-v1/checkpoint_best_total.pth
```
