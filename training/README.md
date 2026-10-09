# RF-DETR Training Setup

This directory trains a bounding-box detector for `wind` and `hail`. It is separate
from the Django/React application, but belongs to the same repository. Training is
not started by the website. No inherited checkpoint is required: RF-DETR starts
from its general pretrained weights and downloads them when the model is created.

This entry point is for object detection, not whole-photo `damaged`/`not damaged`
classification. Photo-level labels can support a separate classifier, but do not
meet this script's bounding-box dataset contract. If the team starts with a
classifier, localization still requires a separate training step with box labels;
this folder does not implement that classifier or an automatic conversion from
photo labels into boxes.

## Status

The entry point and dependency specification are prepared for a future GPU machine.
Dependencies have not been installed here and an actual RF-DETR training run has
not been verified. The team is arranging access to a suitable GPU computer and
preparing its dataset. The current laptop has a 4 GB NVIDIA GPU and recently ran
out of storage during model recovery; it is not the planned full-training machine.
An SSD adds storage, not GPU memory. The old exported model is not used by this script.

## Files

- `train.py`: dataset checks and the RF-DETR training entry point.
- `requirements.txt`: pinned RF-DETR version and training dependencies.
- `test_train.py`: offline tests; no RF-DETR install or GPU required.
- `.gitignore`: excludes local environments, datasets, runs, and model artifacts.

## Dataset Contract

Use a COCO object-detection export with this layout:

```text
roof-dataset/
    train/
        _annotations.coco.json
        image files referenced by that JSON
    valid/
        _annotations.coco.json
        image files referenced by that JSON
    test/                         optional; keep held out for final evaluation
        _annotations.coco.json
        image files referenced by that JSON
```

Each JSON must declare exactly these categories:

```json
[{"id": 0, "name": "wind"}, {"id": 1, "name": "hail"}]
```

Annotations must contain an ID, image ID, category ID, and a `bbox` in pixel units:
`[left, top, width, height]`. Image records need ID, filename, width, and height.
Both train and validation splits need annotated examples of both classes. Negative
photos remain in the image list with no damage annotations; do not add an
`undamaged` object class. Whole-photo labels alone do not satisfy this contract.
If the dataset exporter uses different category IDs or an extra parent category,
coordinate a reviewed conversion first; this script does not silently relabel data.

Keep photos from the same roof out of multiple splits. The checks validate JSON,
image-file existence, class IDs, and box bounds, not labeling quality, actual image
decoding, or duplicate roofs. Keep large datasets outside GitHub and do not upload
private inspection photos to a service without the team's authorization.

## GPU Machine Setup

Use Python 3.11 and an NVIDIA GPU accessible to PyTorch. A school GPU machine or an
approved cloud GPU is suitable. GPU memory needs depend on model, image size, and
batch size; the script starts with Nano and batch size 1, not a guaranteed hardware
minimum. Run a short pilot before planning a long job. Confirm storage, permissions,
available training time, and cloud costs with the machine owner.

Clone the project onto that machine. Training does not require starting the backend
or frontend, or initializing the frontend submodule. From the repository root on Windows:

```powershell
py -3.11 -m venv training/.venv
```

Before installing requirements, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/)
to choose a GPU-compatible PyTorch build for the machine's OS and driver. Run its
installation command with `training/.venv/Scripts/python.exe -m pip` instead of bare
`pip`, so it installs into this environment. Do not install into the Django environment.
Then:

```powershell
training/.venv/Scripts/python.exe -m pip install -r training/requirements.txt
training/.venv/Scripts/python.exe -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

On Linux, create the environment with `python3.11 -m venv training/.venv` and use
`training/.venv/bin/python` in place of the Windows interpreter path. Internet access
is needed for dependency installation and the initial pretrained-weight download.
The RF-DETR version is pinned; its transitive dependencies are not fully locked yet.
Record the working environment after the first verified GPU run.

## Check and Train

Structure checks work with standard Python alone, before installing training packages:

```powershell
py -3.11 training/train.py --dataset D:/roof-dataset --check-only
```

Replace `D:` with the actual dataset drive. No GPU, downloads, or output writes occur
with `--check-only`. Run a short pilot on the configured GPU machine:

```powershell
training/.venv/Scripts/python.exe training/train.py --dataset D:/roof-dataset --output D:/roof-runs/pilot-001 --epochs 1
```

After reviewing the pilot, start a separate full run:

```powershell
training/.venv/Scripts/python.exe training/train.py --dataset D:/roof-dataset --output D:/roof-runs/run-001 --epochs 50
```

The epoch count is a starting setting, not a promise of accuracy. `--model small`,
`--batch-size`, and `--grad-accum-steps` are configurable. CPU training is an explicit
`--device cpu` option for limited experiments, not an automatic GPU fallback. Workers
default to zero for portability. Existing nonempty run directories are refused;
this starter does not implement resuming interrupted training. Test-set evaluation
is not automatically run during training.

RF-DETR writes its checkpoints and training logs into the output directory. Preserve
the dataset version, class mapping, command, environment versions, and results together.
Do not commit model files or datasets into Git. A trained checkpoint is not automatically
an ONNX file and is not automatically connected to the application. After training,
evaluate on held-out roofs, export a selected model, and verify its input/output layout
and class mapping before integrating it with the existing backend.

## Offline Tests

From the repository root:

```powershell
py -3.11 -B -m unittest discover -s training -p "test_*.py" -v
```

## References

- [RF-DETR repository](https://github.com/roboflow/rf-detr)
- [Training API](https://rfdetr.roboflow.com/latest/learn/train/)
- [Dataset formats](https://rfdetr.roboflow.com/latest/learn/train/dataset-formats/)
- [Pinned package release](https://pypi.org/project/rfdetr/1.11.2/)

The online `latest` documentation can change; review it against the pinned package
before modifying this setup.
