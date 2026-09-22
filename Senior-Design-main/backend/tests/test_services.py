import pytest
from core.services import RFDETRService


def test_predict_normal(monkeypatch):
    monkeypatch.setattr(
        RFDETRService,
        "_load_model",
        classmethod(lambda cls: ("MODEL", ["wind"], "ONNX")),
    )

    def fake_run(model, image_path, class_names, threshold):
        return ["det"], "output.jpg"

    monkeypatch.setattr(RFDETRService, "_run_normal", fake_run, raising=False)

    detections, path = RFDETRService.predict("image.jpg", mode="normal", threshold=0.55)

    assert detections == ["det"]
    assert path == "output.jpg"


def test_predict_tiled(monkeypatch):
    monkeypatch.setattr(
        RFDETRService,
        "_load_model",
        classmethod(lambda cls: ("MODEL", ["wind"], "ONNX")),
    )

    monkeypatch.setattr(RFDETRService, "_download_image", lambda url: "downloaded.jpg")

    def fake_run_tiled(**kwargs):
        return ["det"], "tiled.jpg"

    monkeypatch.setattr(RFDETRService, "_run_tiled", fake_run_tiled, raising=False)

    detections, path = RFDETRService.predict(
        "http://example.com/img.jpg",
        mode="tiled",
        threshold=0.2,
        tile_size=600,
    )

    assert detections == ["det"]
    assert path == "tiled.jpg"

def test_load_model_returns_cached_model(monkeypatch):
    from core.services import RFDETRService

    RFDETRService._model = "MODEL"

    model, class_names, model_type = RFDETRService._load_model()

    assert model == "MODEL"
    assert class_names == ["wind", "hail"]
    assert model_type == "ONNX"

    RFDETRService._model = None  # cleanup

import pytest

def test_load_model_raises_when_base_dir_missing(monkeypatch):
    from core.services import RFDETRService

    class FakePath:
        def resolve(self):
            return self

        @property
        def parent(self):
            return self

        def exists(self):
            return False

    monkeypatch.setattr("core.services.Path", lambda *args, **kwargs: FakePath())

    with pytest.raises(RuntimeError, match="BASE_DIR does not exist"):
        RFDETRService._load_model()

def test_load_model_raises_when_onnx_missing(monkeypatch):
    from core.services import RFDETRService

    class FakePath:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self):
            return self

        @property
        def parent(self):
            return self

        def exists(self):
            return True  # BASE_DIR exists

    class FakeOnnxPath(FakePath):
        def exists(self):
            return False  # ONNX missing

    monkeypatch.setattr("core.services.Path", lambda *args, **kwargs: FakePath())
    monkeypatch.setattr(
        "core.services.Path",
        lambda *args, **kwargs: FakeOnnxPath() if "exported_models" in str(args) else FakePath(),
    )

    with pytest.raises(RuntimeError, match="No ONNX model found"):
        RFDETRService._load_model()

def test_load_model_success(monkeypatch):
    from core.services import RFDETRService

    class FakePath:
        def __init__(self, *args, **kwargs):
            pass

        def resolve(self):
            return self

        @property
        def parent(self):
            return self

        def exists(self):
            return True

    monkeypatch.setattr("core.services.Path", lambda *args, **kwargs: FakePath())
    monkeypatch.setattr("core.services.RFDETR_ONNX", lambda path: "MODEL")

    model, class_names, model_type = RFDETRService._load_model()

    assert model == "MODEL"
    assert class_names == ["wind", "hail"]
    assert model_type == "ONNX"

    RFDETRService._model = None

def test_download_image_success(monkeypatch):
    from core.services import RFDETRService

    class FakeResponse:
        content = b"data"

        def raise_for_status(self):
            pass

    monkeypatch.setattr("core.services.requests.get", lambda url, timeout: FakeResponse())

    path = RFDETRService._download_image("http://example.com/test.jpg")

    assert path.endswith(".jpg")


def test_download_image_failure(monkeypatch):
    from core.services import RFDETRService

    def boom(url, timeout):
        raise Exception("network error")

    monkeypatch.setattr("core.services.requests.get", boom)

    with pytest.raises(RuntimeError, match="Failed to download image"):
        RFDETRService._download_image("http://example.com/test.jpg")
