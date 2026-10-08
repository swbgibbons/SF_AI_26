# Roof damage classifier (phase 1)

Sorts a roof photo into one of three classes:

| class | meaning |
|---|---|
| `damaged` | asphalt-shingle roof with visible hail, wind or missing/broken-shingle damage |
| `undamaged` | usable roof photo, no damage |
| `unusable` | can't be judged: too small, blurry, too dark/bright, not a roof, wide scenery, or a roof type out of scope |

`unusable` comes from two places:

1. **Quality gate** (`roofml/quality.py`): fixed rules for resolution, blur and exposure. No
   training; runs before the model and short-circuits it.
2. **Model**: photos that pass the gate but still can't be judged (scenery, metal or corrugated
   roofs, close-ups of gutters only) are learned as a third class.

## Setup

```bash
cd Senior-Design-main/ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # for a GPU, install torch first from pytorch.org
python -m pytest tests            # ~10s on CPU
```

## Data

Keep images out of git. Lay them out by class:

```
data/images/damaged/*.jpg
data/images/undamaged/*.jpg
data/images/unusable/*.jpg
```

Then build the manifest (adds group-wise train/val/test splits):

```bash
python -m roofml.make_manifest --images-root data/images --out data/manifest.csv
```

Edit the `group` column so **every photo of the same roof shares one group**; splits are made by
group, so near-identical shots never end up in both train and test. Re-run with
`--manifest data/manifest.csv` to redo splits after editing (delete the `split` column first).

## Workflow

```bash
# 1. Check photo quality (tune thresholds by looking at what gets rejected)
python -m roofml.quality data/images

# 2. Train. Start with a quick baseline, then the main model.
python -m roofml.train --manifest data/manifest.csv --images-root data/images --model resnet50 --out runs/resnet50
python -m roofml.train --manifest data/manifest.csv --images-root data/images --model convnext_tiny --out runs/convnext_tiny

# 3. Export to ONNX (what the Django backend runs) and classify new photos
python -m roofml.export_onnx runs/convnext_tiny/best.pt --out runs/convnext_tiny/roof_cls.onnx
python -m roofml.predict runs/convnext_tiny/roof_cls.onnx path/to/photos
```

`train.py` keeps the epoch with the best validation macro-F1 and writes `metrics.json` with the
per-class precision/recall and confusion matrix. The number to watch is **`damaged_recall`**: the
share of damaged roofs caught. Run `--eval-test` once, when you have stopped tuning.

`predict.RoofClassifier` only needs numpy, Pillow and onnxruntime, so the backend can call it
before (or instead of) the RF-DETR detector.

## How much data

Transfer learning needs far less than training from scratch, but not single digits. Rough targets:

| class | first usable model | comfortable |
|---|---|---|
| damaged | 150-200 | 500+ |
| undamaged | 150-200 | 500+ |
| unusable | 75-100 | 200+ |

Counted in distinct roofs, not photos. Undamaged photos should come from the same kinds of sources
and cameras as damaged ones; otherwise the model learns "stock photo vs. drone photo" instead of
"damage vs. no damage".
