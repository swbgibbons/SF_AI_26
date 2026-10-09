"""Exercise setup behavior offline; fake images are not model-training fixtures."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import train


class TrainingSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.data = {
            "categories": [{"id": 0, "name": "wind"}, {"id": 1, "name": "hail"}],
            "images": [{"id": 1, "file_name": "roof.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20]},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 20, 20]},
            ],
        }
        for split in ("train", "valid"):
            directory = self.dataset / split
            directory.mkdir(parents=True)
            (directory / "roof.jpg").write_bytes(b"file-existence-only fixture")
            self.write_split(split, self.data)

    def write_split(self, split, data):
        (self.dataset / split / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")

    def test_valid_dataset(self):
        self.assertEqual(train.validate_dataset(self.dataset), {
            "train": {"images": 1, "boxes": 2}, "valid": {"images": 1, "boxes": 2},
        })

    def test_missing_photos(self):
        (self.dataset / "train" / "roof.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "missing image"):
            train.validate_dataset(self.dataset)

    def test_whole_photo_labels_are_not_accepted(self):
        data = copy.deepcopy(self.data)
        data["annotations"] = []
        self.write_split("train", data)
        with self.assertRaisesRegex(ValueError, "both wind and hail"):
            train.validate_dataset(self.dataset)

    def test_reversed_classes_are_not_silently_changed(self):
        data = copy.deepcopy(self.data)
        data["categories"][0]["name"] = "hail"
        data["categories"][1]["name"] = "wind"
        self.write_split("train", data)
        with self.assertRaisesRegex(ValueError, "0=wind and 1=hail"):
            train.validate_dataset(self.dataset)

    def test_invalid_boxes(self):
        for box in ([0, 0, -1, 5], [0, 0, float("nan"), 5], [95, 95, 10, 10]):
            with self.subTest(box=box):
                data = copy.deepcopy(self.data)
                data["annotations"][0]["bbox"] = box
                self.write_split("train", data)
                with self.assertRaises(ValueError):
                    train.validate_dataset(self.dataset)

    def test_unknown_image_reference(self):
        data = copy.deepcopy(self.data)
        data["annotations"][0]["image_id"] = 999
        self.write_split("train", data)
        with self.assertRaisesRegex(ValueError, "unknown image or class"):
            train.validate_dataset(self.dataset)

    def test_check_only_does_not_start_training_or_write_output(self):
        output = self.root / "results"
        with patch("train.run_training") as start, redirect_stdout(io.StringIO()):
            train.main(["--dataset", str(self.dataset), "--output", str(output), "--check-only"])
            start.assert_not_called()
        self.assertFalse(output.exists())

    def test_nonempty_output_is_preserved(self):
        output = self.root / "results"
        output.mkdir()
        existing = output / "existing.ckpt"
        existing.write_bytes(b"keep")
        with patch("train.run_training") as start, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                train.main(["--dataset", str(self.dataset), "--output", str(output)])
            self.assertEqual(error.exception.code, 1)
            start.assert_not_called()
        self.assertEqual(existing.read_bytes(), b"keep")

    def test_output_inside_dataset_is_rejected(self):
        with patch("train.run_training") as start, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                train.main(["--dataset", str(self.dataset), "--output", str(self.dataset / "results")])
            start.assert_not_called()

    def test_training_options_forwarded(self):
        constructor = Mock()
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
            "rfdetr": SimpleNamespace(RFDETRNano=constructor, RFDETRSmall=Mock()),
        }
        output = self.root / "results"
        with patch.dict("sys.modules", modules), redirect_stdout(io.StringIO()):
            train.main(["--dataset", str(self.dataset), "--output", str(output), "--epochs", "1"])
        constructor.assert_called_once_with(device="cuda")
        constructor.return_value.train.assert_called_once_with(
            dataset_dir=str(self.dataset.resolve()), output_dir=str(output.resolve()),
            epochs=1, batch_size=1, grad_accum_steps=1, num_workers=0, seed=42,
            device="cuda", run_test=False,
        )

    def test_missing_gpu_does_not_fall_back_to_cpu(self):
        constructor = Mock()
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            "rfdetr": SimpleNamespace(RFDETRNano=constructor, RFDETRSmall=Mock()),
        }
        with patch.dict("sys.modules", modules):
            with self.assertRaisesRegex(RuntimeError, "No usable NVIDIA GPU"):
                train.run_training(SimpleNamespace(device="cuda"))
        constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
