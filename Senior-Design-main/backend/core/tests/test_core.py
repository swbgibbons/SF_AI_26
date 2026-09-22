import json
import tempfile
import io
from django.test import TestCase, override_settings, RequestFactory
from django.http import HttpResponse
from rest_framework.test import APIRequestFactory
from rest_framework.exceptions import AuthenticationFailed
from unittest.mock import patch, Mock
from django.contrib.auth import get_user_model
from core import authentication, middleware, model_utils, services, views
import os
import sys

# Ensure the project 'backend' package (inner folder) is on sys.path so
# `import backend.core` works when tests are collected from the repository root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


User = get_user_model()


class AuthenticationTests(TestCase):
    def setUp(self):
        self.rf = RequestFactory()

    def test_no_cookie_returns_none(self):
        auth = authentication.CookieJWTAuthentication()
        request = self.rf.get("/")
        request.COOKIES = {}
        assert auth.authenticate(request) is None

    def test_invalid_token_raises_authentication_failed(self):
        auth = authentication.CookieJWTAuthentication()
        request = self.rf.get("/")
        request.COOKIES = {"access": "bad"}

        with patch.object(authentication.CookieJWTAuthentication, "get_validated_token", side_effect=authentication.InvalidToken()):
            with self.assertRaises(AuthenticationFailed):
                auth.authenticate(request)

    def test_valid_token_returns_user_and_token(self):
        auth = authentication.CookieJWTAuthentication()
        request = self.rf.get("/")
        request.COOKIES = {"access": "tok"}

        dummy_user = User.objects.create(username="u1", email="u1@example.com")

        with patch.object(authentication.CookieJWTAuthentication, "get_validated_token", return_value="valid"):
            with patch.object(authentication.CookieJWTAuthentication, "get_user", return_value=dummy_user):
                user, token = auth.authenticate(request)
                assert user == dummy_user
                assert token == "valid"


class MiddlewareTests(TestCase):
    def setUp(self):
        self.rf = RequestFactory()

    @override_settings(DEBUG=True)
    def test_debug_session_user_middleware_sets_user(self):
        user = User.objects.create(username="agent", email="agent@example.com")
        request = self.rf.get("/")
        # provide a simple session dict on the request
        request.session = {"debug_user": {"email": user.email}}

        def dummy_get_response(req):
            return HttpResponse()

        mw = middleware.DebugSessionUserMiddleware(dummy_get_response)
        resp = mw(request)
        # if the middleware found the debug_user it should set request.user
        assert getattr(request, "user", None) is not None

    def test_jwt_auth_middleware_sets_user_when_valid(self):
        user = User.objects.create(username="u2", email="u2@example.com")
        request = self.rf.get("/")
        request.COOKIES = {"access": "sometoken"}

        # patch jwt.decode to return the user's id
        with patch("jwt.decode", return_value={"user_id": user.id}):
            mw = middleware.JWTAuthMiddleware(lambda r: HttpResponse())
            # call the middleware's process_request
            mw.process_request(request)
            assert getattr(request, "user", None) is not None


class ModelUtilsTests(TestCase):
    def test_sigmoid(self):
        val = model_utils.sigmoid(0)
        assert abs(val - 0.5) < 1e-6

    def test_box_conversion(self):
        import numpy as np
        arr = np.array([[0.5, 0.5, 0.2, 0.2]])
        out = model_utils.box_cxcywh_to_xyxyn(arr)
        assert out.shape == (1, 4)

    def test_nms_and_classwise_nms(self):
        import numpy as np
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [
                         50, 50, 60, 60]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = model_utils.nms_numpy(boxes, scores, iou_thres=0.1)
        assert len(keep) >= 1

        labels = np.array([0, 0, 1], dtype=np.int64)
        keep2 = model_utils.classwise_nms(boxes, scores, labels, iou_thres=0.1)
        assert len(keep2) >= 1

    def test_open_image_local_and_missing(self):
        # create a small temp image
        from PIL import Image
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        img.save(tmp.name)
        tmp.close()

        opened = model_utils.open_image(tmp.name)
        assert opened.size == (10, 10)
        # close the PIL image to release file handle on Windows
        try:
            opened.close()
        except Exception:
            pass

        os.unlink(tmp.name)

        with self.assertRaises(FileNotFoundError):
            model_utils.open_image("/path/does/not/exist.png")


class ServicesTests(TestCase):
    def test_download_image_creates_tempfile(self):
        url = "http://example.com/img.jpg"
        fake_resp = Mock()
        fake_resp.content = b"PNGDATA"
        fake_resp.raise_for_status = Mock()

        with patch("requests.get", return_value=fake_resp):
            tmp_path = services.RFDETRService._download_image(url)
            assert os.path.exists(tmp_path)
            os.unlink(tmp_path)

    def test_load_model_uses_local_onnx_when_present(self):
        # create dummy exported_models/inference_model.onnx under backend folder
        core_dir = os.path.dirname(model_utils.__file__)
        backend_dir = os.path.dirname(core_dir)
        exported_dir = os.path.join(backend_dir, "exported_models")
        os.makedirs(exported_dir, exist_ok=True)
        onnx_path = os.path.join(exported_dir, "inference_model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(b"dummy")

        class DummyModel:
            def __init__(self, p):
                self.path = p

        # Patch the RFDETR_ONNX reference used by services
        with patch("core.services.RFDETR_ONNX", DummyModel):
            model, class_names, model_type = services.RFDETRService._load_model()
            assert model is not None

        # cleanup
        try:
            os.remove(onnx_path)
        except Exception:
            pass


class ViewsTests(TestCase):
    def setUp(self):
        self.af = APIRequestFactory()

    def test_get_csrf_returns_ok(self):
        req = RequestFactory().get("/csrf")
        resp = views.get_csrf(req)
        data = json.loads(resp.content)
        assert data.get("status") == "ok"

    def test_auth_receive_missing_credential(self):
        req = self.af.post("/google/auth", data={}, format='json')
        resp = views.auth_receive(req)
        assert resp.status_code == 400

    def test_image_prediction_not_found(self):
        req = self.af.post("/image/pred/", data={}, format='json')
        resp = views.image_prediction(req, image_id=9999999)
        assert resp.status_code == 404

    def test_run_prediction_house_not_found(self):
        req = self.af.post("/run/pred/", data={}, format='json')
        resp = views.run_prediction(req, house_id=9999999)
        assert resp.status_code == 404

    def test_sign_in_renders_with_sign_in_url(self):
        req = RequestFactory().get("/signin")
        # patch render to avoid template requirements
        with patch("django.shortcuts.render", return_value=HttpResponse('ok')) as r:
            resp = views.sign_in(req)
            assert resp.status_code == 200

    def test_sign_out_deletes_cookies_and_redirects(self):
        req = RequestFactory().get("/signout")
        req.META = {"HTTP_HOST": "localhost"}

        class FakeResponse(HttpResponse):
            def __init__(self):
                super().__init__(content=b'redirect')
                self.deleted = []

            def delete_cookie(self, name, **kwargs):
                self.deleted.append((name, kwargs))

        with patch("core.views.redirect", return_value=FakeResponse()) as redir:
            resp = views.sign_out(req)
            # our fake response should have delete_cookie called at least twice
            assert hasattr(resp, "deleted")
            assert any(c[0] == "access" for c in resp.deleted)

    def test_debug_or_jwt_permission(self):
        perm = views.DebugOrJWTAuthenticated()
        # debug session
        req = RequestFactory().get("/")
        req.session = {"debug_user": {"email": "a@b.com"}}
        assert perm.has_permission(req, None) is True

        # authenticated user
        req2 = RequestFactory().get("/")

        class U:
            pass
        u = U()
        u.is_authenticated = True
        req2.user = u
        assert perm.has_permission(req2, None) is True

    def test_is_agent_owner(self):
        user = User.objects.create(username="owner", email="o@example.com")
        other = User.objects.create(username="other", email="x@example.com")
        customer = models = None
        from core.models import Customer, House, HouseImage
        cust = Customer.objects.create(
            agent=user, name="C", email="c@example.com")
        house = House.objects.create(customer=cust, address="1 A St")
        img = HouseImage.objects.create(house=house, image_url="http://x")

        perm = views.IsAgentOwner()
        req = RequestFactory().get("/")
        req.user = user
        assert perm.has_object_permission(req, None, cust) is True
        assert perm.has_object_permission(req, None, house) is True
        assert perm.has_object_permission(req, None, img) is True

        req.user = other
        assert perm.has_object_permission(req, None, cust) is False

    def test_customer_viewset_behaviors(self):
        vs = views.CustomerViewSet()
        user = User.objects.create(username="u3", email="u3@example.com")
        vs.request = Mock()
        vs.request.user = user

        # perform_create should save with agent set
        serializer = Mock()
        vs.perform_create(serializer)
        serializer.save.assert_called_with(agent=user)

        # perform_update should raise when not owner
        other = User.objects.create(username="other2", email="o2@example.com")
        inst = Mock()
        inst.agent = other
        serializer2 = Mock()
        serializer2.instance = inst
        vs.request.user = user
        from rest_framework import exceptions as drf_exceptions
        with self.assertRaises(drf_exceptions.PermissionDenied):
            vs.perform_update(serializer2)

    def test_delete_existing_prediction_clears_fields(self):
        from core.models import HouseImage
        user = User.objects.create(username="u4", email="u4@example.com")
        cust = __import__("core.models", fromlist=["Customer"]).Customer.objects.create(
            agent=user, name="C2", email="c2@example.com")
        house = __import__("core.models", fromlist=["House"]).House.objects.create(
            customer=cust, address="2 B St")
        img = HouseImage.objects.create(
            house=house, image_url="http://orig", predicted_url="http://bucket/x")

        with patch("core.views.delete_file_from_bucket", return_value=True):
            os.environ["BUCKET_NAME"] = "mybucket"
            views._delete_existing_prediction(img)
            # reload from db
            img.refresh_from_db()
            assert img.predicted_url is None

    def test_run_prediction_for_image_updates_image(self):
        from core.models import HouseImage
        user = User.objects.create(username="u5", email="u5@example.com")
        cust = __import__("core.models", fromlist=["Customer"]).Customer.objects.create(
            agent=user, name="C3", email="c3@example.com")
        house = __import__("core.models", fromlist=["House"]).House.objects.create(
            customer=cust, address="3 C St")
        img = HouseImage.objects.create(house=house, image_url="http://orig2")

        # create a fake pred file
        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmpf.write(b"x")
        tmpf.close()

        detections = {"scores": [0.9], "labels": [0], "boxes": [[1, 2, 3, 4]]}
        with patch("core.services.RFDETRService.predict", return_value=(detections, tmpf.name)):
            with patch("core.views.upload_local_file_to_bucket", return_value="http://bucket/pred.jpg"):
                res = views._run_prediction_for_image(
                    img, mode="normal", threshold=0.4, tile_size=560)
                assert res["predicted_image"] == "http://bucket/pred.jpg"
                img.refresh_from_db()
                assert img.predicted_url == "http://bucket/pred.jpg"
        try:
            os.remove(tmpf.name)
        except Exception:
            pass

    def test_image_prediction_success(self):
        from core.models import HouseImage
        user = User.objects.create(username="u6", email="u6@example.com")
        cust = __import__("core.models", fromlist=["Customer"]).Customer.objects.create(
            agent=user, name="C4", email="c4@example.com")
        house = __import__("core.models", fromlist=["House"]).House.objects.create(
            customer=cust, address="4 D St")
        img = HouseImage.objects.create(house=house, image_url="http://orig3")

        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmpf.write(b"x")
        tmpf.close()

        detections = {"scores": [0.8], "labels": [1], "boxes": [[5, 6, 7, 8]]}
        with patch("core.services.RFDETRService.predict", return_value=(detections, tmpf.name)):
            with patch("core.views.upload_local_file_to_bucket", return_value="http://bucket/pred2.jpg"):
                req = self.af.post("/image/pred/", data={}, format='json')
                resp = views.image_prediction(req, image_id=img.id)
                assert resp.status_code == 200
                data = resp.data
                assert data.get("image_id") == img.id
        try:
            os.remove(tmpf.name)
        except Exception:
            pass

    def test_run_prediction_success(self):
        from core.models import HouseImage, House
        user = User.objects.create(username="u7", email="u7@example.com")
        cust = __import__("core.models", fromlist=["Customer"]).Customer.objects.create(
            agent=user, name="C5", email="c5@example.com")
        house = House.objects.create(customer=cust, address="5 E St")
        img1 = HouseImage.objects.create(house=house, image_url="http://i1")
        img2 = HouseImage.objects.create(house=house, image_url="http://i2")

        tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmpf.write(b"x")
        tmpf.close()

        detections = {"scores": [0.7], "labels": [0], "boxes": [[1, 2, 3, 4]]}
        with patch("core.services.RFDETRService.predict", return_value=(detections, tmpf.name)):
            with patch("core.views.upload_local_file_to_bucket", return_value="http://bucket/pred3.jpg"):
                req = self.af.post("/run/pred/", data={}, format='json')
                resp = views.run_prediction(req, house_id=house.id)
                assert resp.status_code == 200
                assert resp.data["total_images"] == 2
        try:
            os.remove(tmpf.name)
        except Exception:
            pass
