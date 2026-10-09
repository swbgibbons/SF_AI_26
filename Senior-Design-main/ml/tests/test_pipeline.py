import csv
import json

import numpy as np
import pytest
from PIL import Image, ImageFilter

from roofml import CLASSES
from roofml.data import assign_splits, read_manifest
from roofml.metrics import classification_report
from roofml.quality import QualityConfig, check_quality


def textured(size=(640, 480), seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def test_quality_gate_passes_sharp_image():
    assert check_quality(textured()).usable


@pytest.mark.parametrize("image,reason", [
    (textured((200, 150)), "low_resolution"),
    (textured().filter(ImageFilter.GaussianBlur(8)), "blurry"),
    (Image.new("RGB", (640, 480), (5, 5, 5)), "too_dark"),
    (Image.new("RGB", (640, 480), (252, 252, 252)), "overexposed"),
])
def test_quality_gate_rejects(image, reason):
    report = check_quality(image, QualityConfig())
    assert not report.usable
    assert any(r.startswith(reason) for r in report.reasons)


def test_splits_keep_groups_together():
    rows = [{"path": f"{i}.jpg", "label": CLASSES[i % 3], "group": f"house{i // 2}"} for i in range(60)]
    assign_splits(rows, seed=1)
    by_group = {}
    for r in rows:
        assert by_group.setdefault(r["group"], r["split"]) == r["split"]
    assert {r["split"] for r in rows} == {"train", "val", "test"}


def test_manifest_rejects_unknown_label(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("path,label,group\na.jpg,hail,h1\n")
    with pytest.raises(ValueError):
        read_manifest(p)


def test_metrics():
    rep = classification_report([0, 0, 1, 2], [0, 1, 1, 2])
    assert rep["damaged_recall"] == 0.5
    assert rep["confusion_matrix"]["values"][0] == [1, 1, 0]


def test_train_export_predict_smoke(tmp_path):
    """Tiny random-init model on synthetic images: checks the plumbing, not accuracy."""
    pytest.importorskip("torch")
    from roofml import export_onnx, predict, train

    root = tmp_path / "images"
    rows = []
    for c_i, label in enumerate(CLASSES):
        (root / label).mkdir(parents=True)
        for k in range(4):
            textured(seed=c_i * 10 + k).save(root / label / f"{k}.jpg")
            split = "train" if k < 2 else "val" if k == 2 else "test"
            rows.append({"path": f"{label}/{k}.jpg", "label": label, "group": f"{label}{k}", "split": split})
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "group", "split"])
        w.writeheader()
        w.writerows(rows)

    out = tmp_path / "run"
    train.main(["--manifest", str(manifest), "--images-root", str(root), "--out", str(out),
                "--model", "resnet18", "--no-pretrained", "--image-size", "64", "--epochs", "1",
                "--batch-size", "4", "--workers", "0", "--eval-test"])
    assert json.load(open(out / "metrics.json"))["best_epoch"] == 1
    assert (out / "test_metrics.json").exists()

    onnx_path = str(out / "m.onnx")
    export_onnx.export(out / "best.pt", onnx_path)
    clf = predict.RoofClassifier(onnx_path)
    assert clf.classes == CLASSES
    assert clf.predict(textured())["label"] in CLASSES
    assert clf.predict(textured((100, 100)))["reason"] == "quality_gate"


def test_review_export_and_apply(tmp_path):
    from roofml import review

    root = tmp_path / "images"
    root.mkdir()
    fields = ["path", "label", "confidence", "damage_type", "hail", "wind", "missing_shingles", "types_confidence", "notes"]
    rows = [
        ["1000001.jpg", "damaged", "high", "wind", "0", "1", "0", "high", ""],
        ["1000002.jpg", "undamaged", "medium", "", "", "", "", "", "looks fine"],
    ]
    for r in rows:
        textured().save(root / r[0])
    manifest = tmp_path / "m.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)

    out = tmp_path / "review"
    assert review.export(manifest, root, out) == {"wind": 1, "undamaged": 1}
    assert (out / "wind" / "wind__1000001.jpg").exists()

    # a reviewer moves the wind photo to hail
    (out / "hail").mkdir()
    (out / "wind" / "wind__1000001.jpg").rename(out / "hail" / "wind__1000001.jpg")
    assert review.apply(manifest, out) == [("1000001", "wind", "hail")]
    updated = {r["path"]: r for r in csv.DictReader(open(manifest))}
    assert updated["1000001.jpg"]["damage_type"] == "hail"
    assert (updated["1000001.jpg"]["hail"], updated["1000001.jpg"]["wind"]) == ("1", "0")
    assert review.apply(manifest, out) == []
