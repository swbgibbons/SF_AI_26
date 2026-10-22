# Roof damage detector

Draws a box around each damaged area of a roof photo and names it `wind`, `hail` or
`missing_shingles`. A photo's overall damage type is the class of its most confident box;
no box means no damage found.

## Data

- `dataset/manifest.csv`: one row per photo, with the photo-level label, damage type and a
  train/val/test split made by roof group (shots of the same roof never cross splits).
- `dataset/boxes.csv`: one row per box. Coordinates are fractions of the photo's width and height
  (`0,0` top-left, `1,1` bottom-right). `status` is `draft` (first pass, needs a human check) or
  `low` (labeler unsure).

Rules: one damage type per photo; when wind tore shingles off, it is `wind`. Undamaged photos are
used as background (no boxes). Unusable photos are left out.

## Run it

```bash
cd Senior-Design-main/ml
pip install ultralytics            # pulls in torch; install a CUDA torch first for a GPU
python -m detect.build_dataset --version v1 --out data/det   # frozen v1 -> YOLO + COCO (RF-DETR layout)
python -m detect.train --name v1   # CPU works: ~170 photos at 640px
python -m detect.evaluate --weights runs/detect/v1/weights/best.pt --split test --out reports/v1
python -m detect.predict --weights runs/detect/v1/weights/best.pt --images path/to/photos --out preds
pip install gradio && python -m detect.demo   # upload-a-photo demo at http://127.0.0.1:7860
```

Trained v1 weights (`best.pt`, `best.onnx`) are kept out of git; ask the team for the copy in the
project files (`models/v1/`) and place it at `runs/detect/v1/weights/`.

To train RF-DETR on the same data and score it against YOLO with one scorer
(results in `reports/yolo_vs_rfdetr.md`):

```bash
pip install "rfdetr[train]"
python -m detect.train_rfdetr --data data/det-v1/coco --size nano --epochs 40 --name rfdetr-n-v1
python -m detect.compare --version v1 --split test --out reports/compare_v1_test \
    --model yolo:runs/detect/v1/weights/best.pt \
    --model rfdetr-nano:runs/rfdetr/rfdetr-n-v1/checkpoint_best_total.pth
```

`evaluate` reports box mAP per class and photo-level results (right type, damage found, clean
roofs left clean), and saves every test photo with its predicted boxes, named
`<truth>__as__<prediction>__<file>`.

## Checking and fixing boxes

```bash
python -m detect.review_boxes --out review/boxes
```

writes each damaged photo with its boxes drawn, one folder per class. Fix a box by editing its row
in `boxes.csv`, then rebuild and retrain.

## Class order

`wind=0, hail=1, missing_shingles=2`. This keeps the backend's existing `["wind", "hail"]` order
and appends the new class, so `core/services.py` only needs the third name added.
