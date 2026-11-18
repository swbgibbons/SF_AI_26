# RF-DETR Training

Trains the RoofInsight roof damage detector on the COCO dataset `roof_damage_rfdetr.zip`
(1,583 images, train / valid / test). It is separate from the Django/React app; the website
never starts training.

Classes, with the same ids as the v1 RF-DETR model:

| id | class |
|---|---|
| 0 | wind |
| 1 | hail |
| 2 | missing_shingles |

Photos with no boxes are no-damage examples. There is no `undamaged` class.

## Files

- `train.py`: dataset checks plus the RF-DETR training entry point (Colab or any NVIDIA GPU machine).
- `train_colab.ipynb`: the same thing as a Colab notebook that saves checkpoints to Google Drive.
- `requirements.txt`: pinned RF-DETR version.
- `test_train.py`: offline tests; no GPU or RF-DETR install needed.

## Requirements

- NVIDIA GPU with 16 GB or more (a Colab T4 is fine for `nano`, `small` and `medium`).
- Python 3.10 or newer.
- About 2 GB of disk for the unzipped dataset and 3 to 5 GB for one run's checkpoints.
- Internet access for `pip install` and the first download of RF-DETR's pretrained weights.

## Option A: Google Colab

1. In Google Drive make `MyDrive/RoofInsight/` and copy in `roof_damage_rfdetr.zip` and `training/train.py`.
2. Open `train_colab.ipynb` in Colab (File > Upload notebook), choose a T4 GPU runtime.
3. Edit the settings cell if needed and run all cells.

Checkpoints go to `MyDrive/RoofInsight/runs/<RUN_NAME>/`. If Colab disconnects, reconnect and run all
cells again with the same `RUN_NAME`; training resumes from the last finished epoch.

## Option B: lab or cloud GPU machine (Linux or Windows)

```bash
python3 -m venv training/.venv
# Install a CUDA build of PyTorch first if the machine needs one: https://pytorch.org/get-started/locally/
training/.venv/bin/python -m pip install -r training/requirements.txt
training/.venv/bin/python -c "import torch; print('GPU:', torch.cuda.is_available())"

# 1. check the dataset (no GPU needed; a .zip or an unzipped folder both work)
training/.venv/bin/python training/train.py --dataset /data/roof_damage_rfdetr.zip --check-only

# 2. short pilot to check memory and speed
training/.venv/bin/python training/train.py --dataset /data/roof_damage_rfdetr.zip \
    --output /data/runs/pilot --epochs 2

# 3. full run
training/.venv/bin/python training/train.py --dataset /data/roof_damage_rfdetr.zip \
    --output /data/runs/medium-run-001 --model medium --epochs 50
```

On Windows use `training\.venv\Scripts\python.exe` instead of `training/.venv/bin/python`.
For long runs on a remote machine, start the command inside `tmux` or `screen` so it survives a dropped SSH session.

## Options

| option | default | meaning |
|---|---|---|
| `--dataset` | required | dataset folder, or the zip (extracted once, next to the zip or to `--extract-to`) |
| `--output` | required to train | run / checkpoint folder |
| `--model` | `medium` | `nano` (v1 size), `small`, `medium`, `base`, `large` |
| `--epochs` | 50 | total epochs (also the target when resuming) |
| `--batch-size` | 4 | images per step, or `auto` to size it to GPU memory |
| `--grad-accum-steps` | 4 | effective batch = batch size x this |
| `--lr` | 1e-4 | learning rate |
| `--resolution` | model default | square input size |
| `--pretrain` | RF-DETR COCO weights | start from other weights, e.g. the v1 `checkpoint_best_total.pth` (same model size) |
| `--resume` | `auto` | `auto` resumes from the newest `.ckpt` in `--output`; `none` refuses a non-empty folder; or a `.ckpt` path |
| `--checkpoint-interval` | 10 | keep an archive checkpoint every N epochs |
| `--early-stopping`, `--patience` | off, 10 | stop when validation mAP stops improving |
| `--device` | `cuda` | `cpu` only for tiny experiments; there is no silent CPU fallback |

If you get "CUDA out of memory", halve `--batch-size` and double `--grad-accum-steps`.

## What a run writes

| file | what it is |
|---|---|
| `last.ckpt` | full training state, rewritten after every epoch; this is what resume uses (about 0.5 GB) |
| `checkpoint_<epoch>.ckpt` | archive copies every `--checkpoint-interval` epochs (about 0.5 GB each) |
| `checkpoint_best_total.pth` | **the best weights by validation mAP; keep this one** |
| `checkpoint_best_ema.pth`, `last_ema.pth` | other weight snapshots |
| `metrics.csv` | per-epoch validation metrics, plus test metrics at the end |
| `training_config.json`, `run_info.json` | exact settings, versions, GPU and every launch / resume |

To save Drive space, set `--checkpoint-interval` equal to `--epochs` so only `last.ckpt` is kept.

## Using the trained model

```python
from rfdetr import RFDETRMedium   # same class as --model
model = RFDETRMedium(pretrain_weights="runs/medium-run-001/checkpoint_best_total.pth")
detections = model.predict("roof.jpg", threshold=0.4)   # class_id 0 wind, 1 hail, 2 missing_shingles
```

A checkpoint is not automatically connected to the backend. Evaluate on the test split, then export
(`model.export()` writes ONNX) and check the class mapping before wiring it into the app.

## What has been verified

On a CPU-only machine with a 22-image subset of this dataset: the full dataset passes the checks; `nano` and
`medium` each train one epoch and write all files above; a second launch with the same `--output` resumed from
`last.ckpt`; and `--pretrain` with the v1 fine-tuned nano weights loads. A full 50-epoch GPU run has not
been done yet, so speed and final accuracy are still to be measured.

## Offline tests

```bash
python -B -m unittest discover -s training -p "test_*.py" -v
```

## References

- [RF-DETR repository](https://github.com/roboflow/rf-detr)
- [Training API](https://rfdetr.roboflow.com/latest/learn/train/)
- [Pinned package release](https://pypi.org/project/rfdetr/1.11.2/)
