from types import SimpleNamespace
from rest_framework.test import APIRequestFactory
from rest_framework.exceptions import PermissionDenied
from django.http import HttpResponse
import core.views as views
import pytest


def test_delete_existing_prediction_returns_early_without_url(monkeypatch):
    img = SimpleNamespace(predicted_url=None)

    monkeypatch.setenv("BUCKET_NAME", "my-bucket")
    called = {"delete": False}

    monkeypatch.setattr(views, "delete_file_from_bucket", lambda *args: called.update(delete=True))

    views._delete_existing_prediction(img)

    assert called["delete"] is False

def test_delete_existing_prediction_clears_fields_when_deleted(monkeypatch):
    class FakeImg:
        def __init__(self):
            self.predicted_url = "https://example.com/pred.jpg"
            self.predicted_at = "now"
            self.saved = False
            self.update_fields = None

        def save(self, update_fields=None):
            self.saved = True
            self.update_fields = update_fields

    img = FakeImg()

    monkeypatch.setenv("BUCKET_NAME", "my-bucket")
    monkeypatch.delenv("SKIP_CLOUD_UPLOAD", raising=False)
    monkeypatch.setattr(views, "delete_file_from_bucket", lambda url, bucket: True)

    views._delete_existing_prediction(img)

    assert img.predicted_url is None
    assert img.predicted_at is None
    assert img.saved is True
    assert img.update_fields == ["predicted_url", "predicted_at"]

def test_delete_existing_prediction_handles_delete_exception(monkeypatch):
    class FakeImg:
        def __init__(self):
            self.predicted_url = "https://example.com/pred.jpg"
            self.predicted_at = "now"
            self.saved = False

        def save(self, update_fields=None):
            self.saved = True

    img = FakeImg()

    monkeypatch.setenv("BUCKET_NAME", "my-bucket")
    monkeypatch.delenv("SKIP_CLOUD_UPLOAD", raising=False)

    def boom(url, bucket):
        raise Exception("bucket error")

    monkeypatch.setattr(views, "delete_file_from_bucket", boom)

    views._delete_existing_prediction(img)

    assert img.predicted_url == "https://example.com/pred.jpg"
    assert img.predicted_at == "now"
    assert img.saved is False

def test_run_prediction_for_image_success(monkeypatch, tmp_path):
    import core.views as views
    import core.services as services

    class FakeImg:
        def __init__(self):
            self.id = 1
            self.image_url = "img.jpg"
            self.predicted_url = None
            self.predicted_at = None
            self.detections = None
            self.saved = False
            self.update_fields = None

        def save(self, update_fields=None):
            self.saved = True
            self.update_fields = update_fields

    img = FakeImg()

    monkeypatch.setattr(views, "_delete_existing_prediction", lambda x: None)

    class FakeService:
        @staticmethod
        def predict(**kwargs):
            fake_path = tmp_path / "pred.jpg"
            fake_path.write_text("fake")
            return {"det": 1}, str(fake_path)

    monkeypatch.setattr(services, "RFDETRService", FakeService)

    monkeypatch.setattr(
        views,
        "upload_local_file_to_bucket",
        lambda path, bucket_name: "uploaded_url",
    )

    monkeypatch.setenv("BUCKET_NAME", "bucket")

    removed = {"called": False}
    monkeypatch.setattr(views.os, "remove", lambda path: removed.update(called=True))

    result = views._run_prediction_for_image(img, "normal", 0.5, 256)

    assert result["image_id"] == 1
    assert result["original_image"] == "img.jpg"
    assert result["predicted_image"] == "uploaded_url"
    assert result["detections"] == {"det": 1}

    assert img.predicted_url == "uploaded_url"
    assert img.detections == {"det": 1}
    assert img.saved is True
    assert img.update_fields == ["predicted_url", "predicted_at", "detections"]
    assert removed["called"] is True


def test_run_prediction_for_image_cleanup_error(monkeypatch):
    import core.views as views
    import core.services as services

    class FakeImg:
        def __init__(self):
            self.id = 1
            self.image_url = "img.jpg"
            self.predicted_url = None
            self.predicted_at = None
            self.detections = None

        def save(self, update_fields=None):
            self.update_fields = update_fields

    img = FakeImg()

    monkeypatch.setattr(views, "_delete_existing_prediction", lambda x: None)

    class FakeService:
        @staticmethod
        def predict(**kwargs):
            return {"det": 1}, "fake_path.jpg"

    monkeypatch.setattr(services, "RFDETRService", FakeService)
    monkeypatch.setattr(views, "upload_local_file_to_bucket", lambda *args, **kwargs: "url")
    monkeypatch.setenv("BUCKET_NAME", "bucket")
    monkeypatch.setattr(views.os.path, "exists", lambda path: True)

    def boom(path):
        raise Exception("delete failed")

    monkeypatch.setattr(views.os, "remove", boom)

    result = views._run_prediction_for_image(img, "normal", 0.5, 256)

    assert result["predicted_image"] == "url"
    assert result["detections"] == {"det": 1}


def test_sign_in_default_url(monkeypatch):
    import core.views as views

    class FakeRequest:
        pass

    captured = {}

    def fake_render(request, template, context):
        captured["template"] = template
        captured["context"] = context
        return "response"

    monkeypatch.setattr(views, "render", fake_render)
    monkeypatch.delenv("SIGN_IN_URL", raising=False)

    response = views.sign_in(FakeRequest())

    assert captured["template"] == "backend/sign_in.html"
    assert captured["context"]["sign_in_url"] == "http://localhost:8000/api/google/auth/"

def test_sign_in_adds_https(monkeypatch):
    import core.views as views

    class FakeRequest:
        pass

    captured = {}

    def fake_render(request, template, context):
        captured["context"] = context
        return "response"

    monkeypatch.setattr(views, "render", fake_render)
    monkeypatch.setenv("SIGN_IN_URL", "www.example.com")

    response = views.sign_in(FakeRequest())

    assert captured["context"]["sign_in_url"] == "https://www.example.com"
