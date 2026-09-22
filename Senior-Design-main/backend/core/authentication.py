# core/authentication.py

import logging

from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.exceptions import AuthenticationFailed
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError

logger = logging.getLogger(__name__)


class CookieJWTAuthentication(JWTAuthentication):
    def authenticate(self, request):
        access_token = request.COOKIES.get("access")
        if not access_token:
            return None  # no cookie => no auth attempt

        try:
            validated_token = self.get_validated_token(access_token)
            user = self.get_user(validated_token)
            return (user, validated_token)

        except (InvalidToken, TokenError) as e:
            # This is the normal “bad token” path
            raise AuthenticationFailed("Invalid or expired JWT") from e

        except Exception:
            # This is a real bug/config problem. Don’t hide it.
            logger.exception("Unexpected error during cookie JWT auth")
            raise
