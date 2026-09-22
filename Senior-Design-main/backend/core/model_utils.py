"""
ONNX-only inference utilities for RD-DETR.

Replaces the previous RF-DETR Python-model + supervision dependency with a
lightweight ONNXRuntime pipeline to reduce resource usage. Provides:
- RFDETR_ONNX: minimal ONNX wrapper
- run_rfdetr_inference: single-image inference and visualization
- run_rfdetr_inference_tiled: simple pass-through to normal (tiling not used)
"""

import io
import os
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MAX_NUMBER_BOXES = 300


def open_image(path: str) -> Image.Image:
    """Load image from local path or URL."""
    if path.startswith("http://") or path.startswith("https://"):
        return Image.open(io.BytesIO(requests.get(path).content))
    if os.path.exists(path):
        return Image.open(path)
    raise FileNotFoundError(f"The file {path} does not exist.")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def box_cxcywh_to_xyxyn(x):
    cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)


class RFDETR_ONNX:
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]

    def __init__(self, onnx_model_path: str):
        try:
            self.ort_session = ort.InferenceSession(onnx_model_path)
            input_info = self.ort_session.get_inputs()[0]
            self.input_height, self.input_width = input_info.shape[2:]
        except Exception as e:
            raise RuntimeError(
                f"Failed to load ONNX model from '{onnx_model_path}'."
            ) from e

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize(
            (self.input_width, self.input_height), resample=Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        image = ((image - self.MEANS) / self.STDS).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return image

    def _post_process(
        self,
        outputs,
        origin_height: int,
        origin_width: int,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_number_boxes: int = DEFAULT_MAX_NUMBER_BOXES,
    ):
        masks = outputs[2] if len(outputs) == 3 else None
        prob = sigmoid(outputs[1])
        # logits = outputs[1]  # (1, num_queries, num_classes [+1])
        # prob = softmax(logits, axis=-1)
        #
        scores = np.max(prob, axis=2).squeeze()
        labels = np.argmax(prob, axis=2).squeeze()
        sorted_idx = np.argsort(scores)[::-1]
        scores = scores[sorted_idx][:max_number_boxes]
        labels = labels[sorted_idx][:max_number_boxes]
        boxes = outputs[0].squeeze()[sorted_idx][:max_number_boxes]
        if masks is not None:
            masks = masks.squeeze()[sorted_idx][:max_number_boxes]

        boxes = box_cxcywh_to_xyxyn(boxes)
        boxes[..., [0, 2]] *= origin_width
        boxes[..., [1, 3]] *= origin_height

        if masks is not None:
            new_w, new_h = origin_width, origin_height
            masks = np.stack([
                np.array(Image.fromarray(img).resize((new_w, new_h)))
                for img in masks
            ], axis=0)
            masks = (masks > 0).astype(np.uint8) * 255

        confidence_mask = scores > confidence_threshold
        scores = scores[confidence_mask]
        labels = labels[confidence_mask]
        boxes = boxes[confidence_mask]
        if masks is not None:
            masks = masks[confidence_mask]

        return scores, labels, boxes, masks

    def predict(
        self,
        image_path: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_number_boxes: int = DEFAULT_MAX_NUMBER_BOXES,
    ):
        image = open_image(image_path).convert("RGB")
        origin_width, origin_height = image.size
        input_image = self._preprocess(image)
        input_name = self.ort_session.get_inputs()[0].name
        outputs = self.ort_session.run(None, {input_name: input_image})
        # print([o.shape for o in outputs])
        return self._post_process(outputs, origin_height, origin_width, confidence_threshold, max_number_boxes)

    def save_detections(self, image_path: str, boxes, labels, masks, save_image_path: Path, class_names=None):
        base = open_image(image_path).convert("RGBA")
        result = base.copy()

        label_colors = {
            label: (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                100,
            )
            for label in np.unique(labels)
        }

        if masks is not None:
            for i in range(masks.shape[0]):
                label = labels[i]
                color = label_colors[label]
                mask_overlay = Image.fromarray(masks[i]).convert("L")
                mask_overlay = ImageOps.autocontrast(mask_overlay)
                overlay_color = Image.new("RGBA", base.size, color)
                overlay_masked = Image.new("RGBA", base.size)
                overlay_masked.paste(overlay_color, (0, 0), mask_overlay)
                result = Image.alpha_composite(result, overlay_masked)

        result_rgb = result.convert("RGB")
        draw = ImageDraw.Draw(result_rgb)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except Exception:
            font = ImageFont.load_default()

        for i, box in enumerate(boxes.astype(int)):
            label = int(labels[i])
            box_color = tuple(label_colors[label][:3])
            text_label = (
                class_names[label]
                if class_names and 0 <= label < len(class_names)
                else str(label)
            )
            draw.rectangle(box.tolist(), outline=box_color, width=4)
            draw.text((box[0] + 5, box[1] + 5), text_label,
                      fill=(255, 255, 255), font=font)

        result_rgb.save(save_image_path)


# ================================================================
#                      NORMAL INFERENCE
# ================================================================
def run_rfdetr_inference(model: RFDETR_ONNX, image_path: str, class_names=None, save_dir="saved_predictions", threshold=0.5):
    """Run ONNX RF-DETR inference on one image and save visualization."""
    scores, labels, boxes, masks = model.predict(
        image_path,
        confidence_threshold=threshold,
    )

    if scores is None or len(scores) == 0:
        print("No detections found.")
        return None, None

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{Path(image_path).stem}_pred.jpg"
    model.save_detections(image_path, boxes, labels, masks,
                          save_path, class_names=class_names)

    detections = {
        "scores": scores.tolist(),
        "labels": labels.tolist(),
        "boxes": boxes.tolist(),
    }

    return detections, str(save_path)


# ================================================================
#                   SAHI-STYLE / TILED (fallback)
# ================================================================
def _iou_xyxy(a, b):
    """IoU between one box a (4,) and many boxes b (N,4), xyxy format."""
    xA = np.maximum(a[0], b[:, 0])
    yA = np.maximum(a[1], b[:, 1])
    xB = np.minimum(a[2], b[:, 2])
    yB = np.minimum(a[3], b[:, 3])

    inter_w = np.maximum(0.0, xB - xA)
    inter_h = np.maximum(0.0, yB - yA)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, a[2] - a[0]) * np.maximum(0.0, a[3] - a[1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * \
        np.maximum(0.0, b[:, 3] - b[:, 1])

    union = area_a + area_b - inter + 1e-9
    return inter / union


def nms_numpy(boxes, scores, iou_thres=0.5):
    """Standard NMS. boxes: (N,4), scores: (N,)"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = _iou_xyxy(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thres]

    return np.array(keep, dtype=np.int64)


def classwise_nms(boxes, scores, labels, iou_thres=0.5):
    """Run NMS per class, then merge kept indices."""
    keep_all = []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        keep = nms_numpy(boxes[idx], scores[idx], iou_thres=iou_thres)
        keep_all.append(idx[keep])
    if not keep_all:
        return np.array([], dtype=np.int64)
    keep_all = np.concatenate(keep_all)
    # sort final by score desc
    keep_all = keep_all[np.argsort(scores[keep_all])[::-1]]
    return keep_all


def run_rfdetr_inference_tiled(
    model: RFDETR_ONNX,
    image_path: str,
    class_names=None,
    tile_size=640,
    overlap=0.2,
    conf_thres=0.35,
    iou_thres=0.5,
    max_number_boxes=300,
    save_dir="saved_predictions_tiled",
):
    """
    ONNX-only tiled inference (SAHI-style sliding window).
    - runs model on tiles
    - maps tile detections back to global coords
    - merges via (class-wise) NMS
    """
    image = open_image(image_path).convert("RGB")
    W, H = image.size

    step = int(tile_size * (1.0 - overlap))
    if step <= 0:
        raise ValueError("overlap too large; step becomes <= 0")

    input_name = model.ort_session.get_inputs()[0].name

    all_boxes = []
    all_scores = []
    all_labels = []

    x_starts = list(range(0, max(W - tile_size, 0) + 1, step))
    y_starts = list(range(0, max(H - tile_size, 0) + 1, step))
    if len(x_starts) == 0:
        x_starts = [0]
    if len(y_starts) == 0:
        y_starts = [0]
    if x_starts[-1] != max(W - tile_size, 0):
        x_starts.append(max(W - tile_size, 0))
    if y_starts[-1] != max(H - tile_size, 0):
        y_starts.append(max(H - tile_size, 0))

    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + tile_size, W)
            y1 = min(y0 + tile_size, H)

            tile = image.crop((x0, y0, x1, y1))
            tile_w, tile_h = tile.size

            inp = model._preprocess(tile)  # resizes to model input internally
            outputs = model.ort_session.run(None, {input_name: inp})

            scores, labels, boxes, masks = model._post_process(
                outputs,
                origin_height=tile_h,
                origin_width=tile_w,
                confidence_threshold=conf_thres,
                max_number_boxes=max_number_boxes,
            )

            if scores is None or len(scores) == 0:
                continue

            # map tile boxes -> global boxes
            boxes = boxes.copy()
            boxes[:, [0, 2]] += x0
            boxes[:, [1, 3]] += y0

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

    if not all_boxes:
        print("No detections found.")
        return None, None

    all_boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
    all_scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    all_labels = np.concatenate(all_labels, axis=0).astype(np.int64)

    # NMS merge (class-wise is usually correct for detectors)
    keep = classwise_nms(all_boxes, all_scores,
                         all_labels, iou_thres=iou_thres)
    final_boxes = all_boxes[keep]
    final_scores = all_scores[keep]
    final_labels = all_labels[keep]

    # save visualization (no masks)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{Path(image_path).stem}_tiled_pred.jpg"
    model.save_detections(image_path, final_boxes, final_labels,
                          None, save_path, class_names=class_names)

    detections = {
        "scores": final_scores.tolist(),
        "labels": final_labels.tolist(),
        "boxes": final_boxes.tolist(),
    }
    return detections, str(save_path)
