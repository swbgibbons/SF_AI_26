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
            "categories": [{"id": 0, "name": "wind"}, {"id": 1, "name": "hail"}, {"id": 2, "name": "missing_shingles"}],
            "images": [{"id": 1, "file_name": "roof.jpg", "width": 100, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20]},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 20, 20]},
                {"id": 3, "image_id": 1, "category_id": 2, "bbox": [30, 60, 10, 10]},
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
        expected = {"images": 1, "boxes": 3, "no_damage_images": 0,
                    "boxes_per_class": {"wind": 1, "hail": 1, "missing_shingles": 1}}
        self.assertEqual(train.validate_dataset(self.dataset), {"train": expected, "valid": expected})

    def test_zip_with_wrapper_folder_is_extracted(self):
        import shutil
        archive = shutil.make_archive(str(self.root / "roof_damage_rfdetr"), "zip", self.root, "dataset")
        found = train.prepare_dataset(archive, self.root / "extracted")
        self.assertEqual(found, (self.root / "extracted" / "dataset").resolve())
        self.assertIn("train", train.validate_dataset(found))

    def test_missing_photos(self):
        (self.dataset / "train" / "roof.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "missing image"):
            train.validate_dataset(self.dataset)

    def test_whole_photo_labels_are_not_accepted(self):
        data = copy.deepcopy(self.data)
        data["annotations"] = []
        self.write_split("train", data)
        with self.assertRaisesRegex(ValueError, "every class"):
            train.validate_dataset(self.dataset)

    def test_reversed_classes_are_not_silently_changed(self):
        data = copy.deepcopy(self.data)
        data["categories"][0]["name"] = "hail"
        data["categories"][1]["name"] = "wind"
        self.write_split("train", data)
        with self.assertRaisesRegex(ValueError, "0=wind, 1=hail, 2=missing_shingles"):
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
        existing = output / "notes.txt"
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

    def test_existing_checkpoint_is_resumed(self):
        output = self.root / "results"
        output.mkdir()
        (output / "last.ckpt").write_bytes(b"ckpt")
        with patch("train.run_training") as start, redirect_stdout(io.StringIO()):
            train.main(["--dataset", str(self.dataset), "--output", str(output)])
        self.assertEqual(start.call_args.args[3], (output / "last.ckpt").resolve())

    def test_training_options_forwarded(self):
        constructor = Mock()
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda index: SimpleNamespace(name="T4", total_memory=16 * 2**30))),
            "rfdetr": SimpleNamespace(RFDETRMedium=constructor),
        }
        output = self.root / "results"
        with patch.dict("sys.modules", modules), patch("train.write_run_info"), redirect_stdout(io.StringIO()):
            train.main(["--dataset", str(self.dataset), "--output", str(output), "--epochs", "1"])
        constructor.assert_called_once_with(device="cuda")
        kwargs = constructor.return_value.train.call_args.kwargs
        self.assertEqual(kwargs["dataset_dir"], str(self.dataset.resolve()))
        self.assertEqual(kwargs["output_dir"], str(output.resolve()))
        self.assertEqual(kwargs["class_names"], ["wind", "hail", "missing_shingles"])
        self.assertEqual((kwargs["epochs"], kwargs["batch_size"], kwargs["grad_accum_steps"]), (1, 4, 4))
        self.assertNotIn("resume", kwargs)

    def test_missing_gpu_does_not_fall_back_to_cpu(self):
        constructor = Mock()
        modules = {
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            "rfdetr": SimpleNamespace(RFDETRMedium=constructor),
        }
        with patch.dict("sys.modules", modules):
            with self.assertRaisesRegex(RuntimeError, "No usable NVIDIA GPU"):
                train.run_training(SimpleNamespace(device="cuda", model="medium"), self.dataset, {}, None)
        constructor.assert_not_called()

if __name__ == "__main__":
    unittest.main()
