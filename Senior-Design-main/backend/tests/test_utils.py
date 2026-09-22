import os
import io
import types
import pytest

import core.utils as utils


# -----------------------
# Helpers (fake GCS stack)
# -----------------------
class FakeBlob:
    def __init__(self):
        self.upload_from_file_called = False
        self.upload_from_filename_called = False
        self.deleted = False

    def upload_from_file(self, file_obj, content_type=None, timeout=None):
        self.upload_from_file_called = True
        self.last_file_obj = file_obj
        self.last_content_type = content_type
        self.last_timeout = timeout

    def upload_from_filename(self, local_path, timeout=None):
        self.upload_from_filename_called = True
        self.last_local_path = local_path
        self.last_timeout = timeout

    def delete(self):
        self.deleted = True


class FakeBucket:
    def __init__(self):
        self.last_blob_name = None
        self.blobs = {}

    def blob(self, name):
        self.last_blob_name = name
        b = self.blobs.get(name)
        if b is None:
            b = FakeBlob()
            self.blobs[name] = b
        return b


class FakeClient:
    def __init__(self, bucket: FakeBucket):
        self._bucket = bucket
        self.last_bucket_name = None

    def bucket(self, bucket_name):
        self.last_bucket_name = bucket_name
        return self._bucket


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # Ensure env vars don't leak between tests
    monkeypatch.delenv("SKIP_CLOUD_UPLOAD", raising=False)
    monkeypatch.delenv("GCS_UPLOAD_TIMEOUT", raising=False)


def test_upload_file_to_bucket_skips_when_env_set(monkeypatch):
    monkeypatch.setenv("SKIP_CLOUD_UPLOAD", "1")

    f = types.SimpleNamespace(name="photo.jpg", content_type="image/jpeg")
    assert utils.upload_file_to_bucket(f, "my-bucket") is None


def test_upload_file_to_bucket_success(monkeypatch):
    # Stable uuid so we can assert the URL and blob name
    monkeypatch.setattr(utils.uuid, "uuid4", lambda: "fixed-uuid")

    fake_bucket = FakeBucket()
    fake_client = FakeClient(fake_bucket)
    monkeypatch.setattr(utils, "get_gcs_client", lambda: fake_client)

    # File-like object must have .name; upload reads the object but our fake doesn't
    file_obj = types.SimpleNamespace(name="x.png", content_type="image/png")

    url = utils.upload_file_to_bucket(file_obj, "my-bucket")

    assert url == "https://storage.googleapis.com/my-bucket/fixed-uuid.png"
    assert fake_client.last_bucket_name == "my-bucket"
    assert fake_bucket.last_blob_name == "fixed-uuid.png"
    assert fake_bucket.blobs["fixed-uuid.png"].upload_from_file_called is True


def test_upload_file_to_bucket_returns_none_if_client_init_fails(monkeypatch):
    def boom():
        raise Exception("no creds")

    monkeypatch.setattr(utils, "get_gcs_client", boom)

    file_obj = types.SimpleNamespace(name="x.png", content_type="image/png")
    assert utils.upload_file_to_bucket(file_obj, "my-bucket") is None


def test_upload_local_file_to_bucket_missing_file_returns_none(monkeypatch):
    monkeypatch.setattr(utils.os.path, "isfile", lambda p: False)
    assert utils.upload_local_file_to_bucket("/nope.jpg", "my-bucket") is None


def test_upload_local_file_to_bucket_success(monkeypatch):
    monkeypatch.setattr(utils.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(utils.uuid, "uuid4", lambda: "fixed-uuid")
    monkeypatch.setenv("GCS_UPLOAD_TIMEOUT", "12.5")

    fake_bucket = FakeBucket()
    fake_client = FakeClient(fake_bucket)
    monkeypatch.setattr(utils, "get_gcs_client", lambda: fake_client)

    url = utils.upload_local_file_to_bucket("/tmp/image.jpg", "my-bucket")

    assert url == "https://storage.googleapis.com/my-bucket/fixed-uuid.jpg"
    blob = fake_bucket.blobs["fixed-uuid.jpg"]
    assert blob.upload_from_filename_called is True
    assert blob.last_local_path == "/tmp/image.jpg"
    assert blob.last_timeout == 12.5


def test_delete_file_from_bucket_success(monkeypatch):
    fake_bucket = FakeBucket()
    fake_client = FakeClient(fake_bucket)
    monkeypatch.setattr(utils, "get_gcs_client", lambda: fake_client)

    ok = utils.delete_file_from_bucket(
        "https://storage.googleapis.com/my-bucket/somefile.png",
        "my-bucket",
    )

    assert ok is True
    assert fake_bucket.last_blob_name == "somefile.png"
    assert fake_bucket.blobs["somefile.png"].deleted is True


def test_delete_file_from_bucket_returns_false_on_failure(monkeypatch):
    class ExplodingBucket(FakeBucket):
        def blob(self, name):
            b = super().blob(name)
            def boom():
                raise Exception("delete failed")
            b.delete = boom
            return b

    fake_bucket = ExplodingBucket()
    fake_client = FakeClient(fake_bucket)
    monkeypatch.setattr(utils, "get_gcs_client", lambda: fake_client)

    ok = utils.delete_file_from_bucket(
        "https://storage.googleapis.com/my-bucket/somefile.png",
        "my-bucket",
    )
    assert ok is False

def test_get_gcs_client_returns_client(monkeypatch):
    import core.utils as utils

    monkeypatch.setattr(utils.storage, "Client", lambda: "CLIENT")

    client = utils.get_gcs_client()

    assert client == "CLIENT"

def test_upload_file_to_bucket_handles_upload_exception(monkeypatch):
    import core.utils as utils

    monkeypatch.delenv("SKIP_CLOUD_UPLOAD", raising=False)

    class FakeBlob:
        def upload_from_file(self, *args, **kwargs):
            raise Exception("upload fail")

    class FakeBucket:
        def blob(self, name):
            return FakeBlob()

    class FakeClient:
        def bucket(self, name):
            return FakeBucket()

    monkeypatch.setattr(utils, "get_gcs_client", lambda: FakeClient())

    class FakeFile:
        name = "test.jpg"
        content_type = "image/jpeg"

    result = utils.upload_file_to_bucket(FakeFile(), "bucket")

    assert result is None

def test_upload_local_file_to_bucket_skips_when_env_set(monkeypatch):
    import core.utils as utils

    monkeypatch.setenv("SKIP_CLOUD_UPLOAD", "1")

    result = utils.upload_local_file_to_bucket("file.jpg", "bucket")

    assert result is None
