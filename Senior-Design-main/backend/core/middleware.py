from django.utils.deprecation import MiddlewareMixin
from django.conf import settings
from django.contrib.auth import get_user_model
import jwt


class DebugSessionUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):

        # Only override user IF DEBUG=true
        if settings.DEBUG:
            debug = request.session.get("debug_user")
            if debug:
                try:
                    request.user = User.objects.get(email=debug["email"])
                except User.DoesNotExist:
                    pass

        return self.get_response(request)


User = get_user_model()


class JWTAuthMiddleware(MiddlewareMixin):
    def process_request(self, request):

        token = request.COOKIES.get("access")

        if not token:
            return

        try:
            decoded = jwt.decode(
                token,
                settings.SIMPLE_JWT["SIGNING_KEY"],
                algorithms=[settings.SIMPLE_JWT["ALGORITHM"]],
            )

            request.user = User.objects.get(id=decoded["user_id"])
        except Exception as e:
            print("JWT ERROR:", e)
