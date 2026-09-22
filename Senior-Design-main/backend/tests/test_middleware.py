import pytest
from types import SimpleNamespace
import core.middleware as mw

def test_debug_session_user_middleware_sets_user_when_debug_and_user_exists(monkeypatch):
    monkeypatch.setattr(mw.settings, "DEBUG", True)

    fake_user = object()

    class FakeManager:
        @staticmethod
        def get(email):
            assert email == "test@example.com"
            return fake_user

    monkeypatch.setattr(mw.User, "objects", FakeManager())

    captured = {}

    def get_response(request):
        captured["user"] = request.user
        return "ok"

    middleware = mw.DebugSessionUserMiddleware(get_response)

    request = SimpleNamespace(
        session={"debug_user": {"email": "test@example.com"}},
        user=None,
    )

    response = middleware(request)

    assert response == "ok"
    assert captured["user"] is fake_user

def test_debug_session_user_middleware_ignores_missing_user(monkeypatch):
    monkeypatch.setattr(mw.settings, "DEBUG", True)

    class FakeManager:
        @staticmethod
        def get(email):
            raise mw.User.DoesNotExist()

    monkeypatch.setattr(mw.User, "objects", FakeManager())

    def get_response(request):
        return "ok"

    middleware = mw.DebugSessionUserMiddleware(get_response)

    request = SimpleNamespace(
        session={"debug_user": {"email": "missing@example.com"}},
        user="original-user",
    )

    response = middleware(request)

    assert response == "ok"
    assert request.user == "original-user"

def test_debug_session_user_middleware_skips_when_not_debug(monkeypatch):
    monkeypatch.setattr(mw.settings, "DEBUG", False)

    def get_response(request):
        return "ok"

    middleware = mw.DebugSessionUserMiddleware(get_response)

    request = SimpleNamespace(
        session={"debug_user": {"email": "test@example.com"}},
        user="original-user",
    )

    response = middleware(request)

    assert response == "ok"
    assert request.user == "original-user"

def test_jwt_auth_middleware_returns_when_no_token():
    middleware = mw.JWTAuthMiddleware(lambda request: None)
    request = SimpleNamespace(COOKIES={}, user=None)

    result = middleware.process_request(request)

    assert result is None
    assert request.user is None

def test_jwt_auth_middleware_sets_user_from_valid_token(monkeypatch):
    middleware = mw.JWTAuthMiddleware(lambda request: None)

    monkeypatch.setattr(
        mw.jwt,
        "decode",
        lambda token, key, algorithms: {"user_id": 123},
    )

    fake_user = object()

    class FakeManager:
        @staticmethod
        def get(id):
            assert id == 123
            return fake_user

    monkeypatch.setattr(mw.User, "objects", FakeManager())

    request = SimpleNamespace(COOKIES={"access": "good-token"}, user=None)

    middleware.process_request(request)

    assert request.user is fake_user


def test_jwt_auth_middleware_handles_decode_error(monkeypatch):
    middleware = mw.JWTAuthMiddleware(lambda request: None)

    def boom(token, key, algorithms):
        raise Exception("bad jwt")

    monkeypatch.setattr(mw.jwt, "decode", boom)

    request = SimpleNamespace(COOKIES={"access": "bad-token"}, user="original-user")

    middleware.process_request(request)

    assert request.user == "original-user"
