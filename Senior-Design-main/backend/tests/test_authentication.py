import pytest
from rest_framework.exceptions import AuthenticationFailed
from rest_framework_simplejwt.exceptions import InvalidToken
from core.authentication import CookieJWTAuthentication


class DummyRequest:
    def __init__(self, cookies):
        self.COOKIES = cookies


def test_authenticate_returns_none_when_no_access_cookie():
    auth = CookieJWTAuthentication()
    request = DummyRequest(cookies={})

    result = auth.authenticate(request)

    assert result is None


def test_authenticate_returns_user_and_token(monkeypatch):
    auth = CookieJWTAuthentication()
    request = DummyRequest(cookies={"access": "valid_token"})

    monkeypatch.setattr(auth, "get_validated_token", lambda token: "validated_token")
    monkeypatch.setattr(auth, "get_user", lambda validated: "test_user")

    result = auth.authenticate(request)

    assert result == ("test_user", "validated_token")


def test_authenticate_raises_when_token_invalid(monkeypatch):
    auth = CookieJWTAuthentication()
    request = DummyRequest(cookies={"access": "bad_token"})

    def fake_get_validated_token(token):
        raise InvalidToken("invalid token")

    monkeypatch.setattr(auth, "get_validated_token", fake_get_validated_token)

    with pytest.raises(AuthenticationFailed, match="Invalid or expired JWT"):
        auth.authenticate(request)
