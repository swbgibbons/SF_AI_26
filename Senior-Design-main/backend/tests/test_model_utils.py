import numpy as np
import pytest
import io
from PIL import Image

import core.model_utils as mu


# Test open_image with a local file
# Covers lines 30–31
def test_open_image_local(tmp_path):

    img_path = tmp_path / "test.jpg"

    # create a simple image file
    img = Image.new("RGB", (10, 10))
    img.save(img_path)

    loaded = mu.open_image(str(img_path))

    assert loaded.size == (10, 10)



# Test open_image with URL (mocked request)
# Covers lines 28–29
def test_open_image_url(monkeypatch):

    img = Image.new("RGB", (5, 5))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    class FakeResponse:
        content = buf.read()

    monkeypatch.setattr(mu.requests, "get", lambda url: FakeResponse())

    loaded = mu.open_image("https://example.com/image.jpg")

    assert loaded.size == (5, 5)


# Test sigmoid function
# Ensures the mathematical sigmoid behaves correctly
def test_sigmoid_basic():
    x = np.array([0.0])
    y = mu.sigmoid(x)

    # sigmoid(0) should equal 0.5
    assert y.shape == (1,)
    assert float(y[0]) == pytest.approx(0.5, rel=1e-6)



# Test bounding box conversion
# Converts center format (cx, cy, w, h) → (xmin, ymin, xmax, ymax)
def test_box_cxcywh_to_xyxyn():
    # Example box
    # cx=10, cy=20, w=4, h=6
    inp = np.array([[10.0, 20.0, 4.0, 6.0]], dtype=np.float32)

    out = mu.box_cxcywh_to_xyxyn(inp)

    # Expected coordinates
    # xmin = 10 - 2
    # ymin = 20 - 3
    # xmax = 10 + 2
    # ymax = 20 + 3
    assert out.shape == (1, 4)
    assert out[0].tolist() == pytest.approx([8.0, 17.0, 12.0, 23.0])


# Test IoU calculation between boxes
# Ensures overlap and non-overlap cases are handled correctly
def test_iou_xyxy_simple_overlap():
    a = np.array([0.0, 0.0, 2.0, 2.0], dtype=np.float32)

    b = np.array([
        [0.0, 0.0, 2.0, 2.0],   # identical box
        [1.0, 1.0, 3.0, 3.0],   # partial overlap
        [3.0, 3.0, 4.0, 4.0],   # no overlap
    ], dtype=np.float32)

    ious = mu._iou_xyxy(a, b)

    # identical boxes → IoU ≈ 1
    assert float(ious[0]) == pytest.approx(1.0, rel=1e-6)

    # no overlap → IoU = 0
    assert float(ious[2]) == pytest.approx(0.0, abs=1e-6)

    # partial overlap → between 0 and 1
    assert 0.0 < float(ious[1]) < 1.0



# Test Non-Maximum Suppression (NMS)
# Ensures overlapping boxes with lower scores are removed
def test_nms_numpy_keeps_best_when_overlap_high():

    boxes = np.array([
        [0.0, 0.0, 2.0, 2.0],   # best box
        [0.2, 0.2, 2.2, 2.2],   # overlapping box (should be removed)
        [5.0, 5.0, 6.0, 6.0],   # separate box (should remain)
    ], dtype=np.float32)

    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    keep = mu.nms_numpy(boxes, scores, iou_thres=0.5)

    # Only best overlapping box + independent box remain
    assert keep.tolist() == [0, 2]



# Edge case: NMS with no boxes
# Should return an empty array
def test_nms_numpy_empty_returns_empty():
    keep = mu.nms_numpy(
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32)
    )

    assert keep.dtype == np.int64
    assert keep.size == 0



# Test class-wise NMS
# Ensures suppression happens independently per class
def test_classwise_nms_runs_per_class_and_sorts():

    boxes = np.array([
        [0.0, 0.0, 2.0, 2.0],      # class 0 (best)
        [0.2, 0.2, 2.2, 2.2],      # class 0 (overlap → removed)
        [10.0, 10.0, 12.0, 12.0],  # class 1
    ], dtype=np.float32)

    scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)
    labels = np.array([0, 0, 1], dtype=np.int64)

    keep = mu.classwise_nms(boxes, scores, labels, iou_thres=0.5)

    # Highest score kept per class
    # Sorted by score descending
    assert keep.tolist() == [0, 2]
