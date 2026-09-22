"""
Core API views for the backend.

This module provides:
- Authentication entrypoints (Google OAuth callback, sign-in/out helpers)
- DRF ViewSets for `Customer`, `House`, `HouseImage`, and agent logs
- Utility permission classes for JWT or session-based debug access
- A prediction endpoint that runs RF-DETR inference and stores results
"""

from .serializers import (
    CustomerSerializer,
    HouseSerializer,
    HouseImageSerializer,
    AgentCustomerLogSerializer
)
import traceback
import os
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render, redirect
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from google.oauth2 import id_token
from google.auth.transport import requests
from rest_framework.response import Response
from rest_framework import viewsets, permissions
from rest_framework.exceptions import PermissionDenied
from .authentication import CookieJWTAuthentication

from .models import AgentCustomerLog, HouseImage, House, Customer
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.permissions import BasePermission
from django.contrib.auth import get_user_model
from .utils import upload_file_to_bucket, upload_local_file_to_bucket, delete_file_from_bucket

from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
User = get_user_model()


@ensure_csrf_cookie
def get_csrf(request):
    """Endpoint to set a CSRF cookie for frontend clients.

    This can be used by the React app to obtain a CSRF token for subsequent
    requests. The presence of the cookie can also be used as a simple check
    that the backend is reachable.
    """
    return JsonResponse({"status": "ok"})


def _delete_existing_prediction(img):
    """Remove any prior predicted image from the bucket and clear db fields."""
    existing_url = getattr(img, "predicted_url", None)
    bucket_name = os.getenv("BUCKET_NAME")

    if not existing_url or not bucket_name or os.getenv("SKIP_CLOUD_UPLOAD") == "1":
        return

    try:
        deleted = delete_file_from_bucket(existing_url, bucket_name)
        if deleted:
            img.predicted_url = None
            img.predicted_at = None
            img.save(update_fields=["predicted_url", "predicted_at"])
    except Exception as e:
        print(f"[WARN] Failed to delete previous prediction from bucket: {e}")


def _run_prediction_for_image(img, mode: str, threshold: float, tile_size: int):
    """Common prediction flow used by both single and batch endpoints."""
    from .services import RFDETRService

    _delete_existing_prediction(img)

    detections, pred_path = RFDETRService.predict(
        image_path_or_url=img.image_url,
        mode=mode,
        threshold=threshold,
        tile_size=tile_size,
    )

    predicted_url = None
    if pred_path:
        predicted_url = upload_local_file_to_bucket(
            pred_path, bucket_name=os.getenv("BUCKET_NAME")
        )

        img.predicted_url = predicted_url
        img.predicted_at = timezone.now()
        img.detections = detections
        img.save(update_fields=["predicted_url", "predicted_at", "detections"])

    try:
        if pred_path and os.path.exists(pred_path):
            os.remove(pred_path)
    except Exception as cleanup_error:
        print(
            f"[WARN] Failed to delete temp file {pred_path}: {cleanup_error}")

    return {
        "image_id": img.id,
        "original_image": img.image_url,
        "predicted_image": predicted_url,
        "detections": detections,
    }


class DebugOrJWTAuthenticated(BasePermission):
    """Permission that allows access for debug sessions or JWT users.

    Access is granted when either of these is true:
    - `debug_user` exists in the Django session (development/testing aid)
    - The request has a valid authenticated user (Cookie JWT)
    """

    def has_permission(self, request, view):
        """Return True if the request is from a debug session or JWT user."""
        sess = getattr(request, "session", None)
        if sess and "debug_user" in sess:
            return True
        return request.user and request.user.is_authenticated


@authentication_classes([CookieJWTAuthentication])
def sign_in(request):
    """Render the sign-in page.

    Returns the template `backend/sign_in.html`. The actual JWT auth
    happens after Google login via `auth_receive` which sets HttpOnly
    cookies for access/refresh tokens.
    """
    SIGN_IN_URL = os.environ.get("SIGN_IN_URL")
    if not SIGN_IN_URL:
        SIGN_IN_URL = "http://localhost:8000/api/google/auth/"
    elif not SIGN_IN_URL.startswith("http://") and not SIGN_IN_URL.startswith("https://"):
        # Ensure Google gets an absolute redirect URI with scheme
        SIGN_IN_URL = f"https://{SIGN_IN_URL}"
    return render(request, "backend/sign_in.html", {"sign_in_url": SIGN_IN_URL})


def google_login_modal(request):
    """Render the Google login modal partial used by the frontend UI."""
    return render(request, "backend/modals/google_login_modal.html")

# @csrf_exempt


@api_view(["POST"])
@permission_classes([AllowAny])
@authentication_classes([])
def auth_receive(request):
    """Handle Google OAuth callback, mint JWTs, and redirect to frontend.

    - URL: `/google/auth`
    - Method: POST
    - Body: `{ "credential": "<Google ID Token>" }`

    Verifies the Google ID token, creates or fetches the user, issues
    SimpleJWT refresh/access tokens, sets them as HttpOnly cookies, and
    redirects to the React app customers page.
    """

    token = request.data.get("credential")
    if not token:
        # print(token)
        return Response({"error": "Missing credential"}, status=400)

    try:
        # print("Verifying Google ID token...")

        user_data = id_token.verify_oauth2_token(
            token,
            requests.Request(),
            os.environ.get("GOOGLE_CLIENT_ID")
        )
    except ValueError:
        print("Invalid token", token)
        return Response({"error": "Invalid token."}, status=400)

    # if not email:
    #     return Response({"error": "Missing email from Google token."}, status=400)

    # # Optional but recommended
    # if not user_data.get("email_verified", False):
    #     return Response({"error": "Email is not verified by Google."}, status=403)

    # # Exact domain match
    # domain = email.rsplit("@", 1)[-1] if "@" in email else ""
    # if domain != "statefarm.com":
    #     return Response({"error": "Unauthorized email domain."}, status=403)

    user, _ = User.objects.get_or_create(
        email=user_data["email"],
        defaults={
            "username": user_data["email"],
            "first_name": user_data.get("name", "")
        }
    )

    refresh = RefreshToken.for_user(user)
    # print("Google authentication successful for:", user.email,
    #       "Created new user:", created, "Refresh token:", str(refresh))

    # -----------------------------
    # REDIRECT TO FRONTEND (configurable)
    # -----------------------------
    frontend_base = os.getenv("FRONTEND_BASE_URL", "http://localhost:8000")
    redirect_url = os.getenv(
        "FRONTEND_CUSTOMERS_URL",
        f"{frontend_base.rstrip('/')}/customers",
    )

    response = HttpResponseRedirect(redirect_url)

    # Align cookie domain across set/delete so production logout works.
    # Only apply the configured domain when it matches the current host; this
    # keeps localhost flows working without the prod domain.
    configured_domain = os.getenv("COOKIE_DOMAIN") or None
    host = request.get_host()
    cookie_domain = configured_domain if configured_domain and host.endswith(
        configured_domain) else None
    cookie_kwargs = {
        "httponly": True,
        "secure": True,
        "samesite": "None",
        "path": "/",
    }
    if cookie_domain:
        cookie_kwargs["domain"] = cookie_domain

    response.set_cookie(
        key="access",
        value=str(refresh.access_token),
        **cookie_kwargs,
    )
    response.set_cookie(
        key="refresh",
        value=str(refresh),
        **cookie_kwargs,
    )

    return response


@api_view(["GET"])
@permission_classes([AllowAny])
def sign_out(request):
    """Clear JWT cookies and redirect to the `login` named route."""
    response = redirect("login")
    configured_domain = os.getenv("COOKIE_DOMAIN") or None
    host = request.get_host()
    cookie_domain = configured_domain if configured_domain and host.endswith(
        configured_domain) else None
    # Django delete_cookie does not accept "secure"; it matches by name/path/domain.
    delete_kwargs = {"samesite": "None", "path": "/"}

    # Delete both with and without domain to cover host-only and domain-scoped cookies.
    response.delete_cookie("access", **delete_kwargs)
    response.delete_cookie("refresh", **delete_kwargs)

    if cookie_domain:
        delete_kwargs_domain = {**delete_kwargs, "domain": cookie_domain}
        response.delete_cookie("access", **delete_kwargs_domain)
        response.delete_cookie("refresh", **delete_kwargs_domain)
    return response


class AgentCustomerLogViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only viewset to list logs for the authenticated agent."""
    serializer_class = AgentCustomerLogSerializer
    permission_classes = [DebugOrJWTAuthenticated]

    def get_queryset(self):
        """Return logs scoped to the authenticated agent only."""
        return AgentCustomerLog.objects.filter(agent=self.request.user)


class IsAgentOwner(permissions.BasePermission):
    """Object-level permission that ensures the agent owns the resource.

    Supports `Customer`, `House`, and `HouseImage` by walking the
    ownership chain to verify `request.user` is the agent.
    """

    def has_object_permission(self, request, view, obj):
        """Return True if the current user owns the object or its parent."""
        # Customer
        if hasattr(obj, "agent"):
            return obj.agent == request.user

        # House
        if hasattr(obj, "customer"):
            return obj.customer.agent == request.user

        # HouseImage
        if hasattr(obj, "house"):
            return obj.house.customer.agent == request.user

        return False


class CustomerViewSet(viewsets.ModelViewSet):
    """CRUD endpoints for customers owned by the authenticated agent.

    Examples:
    - List:    GET    /customers/
    - Create:  POST   /customers/
    - Detail:  GET    /customers/{id}/
    - Update:  PUT    /customers/{id}/
    - Partial: PATCH  /customers/{id}/
    - Delete:  DELETE /customers/{id}/
    """
    queryset = Customer.objects.all()
    serializer_class = CustomerSerializer

    def get_permissions(self):
        """Use ownership checks for object-level operations."""
        if self.action in ["retrieve", "update", "partial_update", "destroy"]:
            # Object-level operations require ownership check
            return [DebugOrJWTAuthenticated(), IsAgentOwner()]
        return [DebugOrJWTAuthenticated()]

    # get create
    def get_queryset(self):
        """Limit to customers owned by the authenticated agent."""
        return self.queryset.filter(agent=self.request.user)

    # post create
    def perform_create(self, serializer):
        """Automatically set the `agent` field to the current user."""
        serializer.save(agent=self.request.user)

    # update
    def perform_update(self, serializer):
        """Validate agent ownership on update."""
        if serializer.instance.agent != self.request.user:
            raise PermissionDenied("You do not own this customer.")
        serializer.save()

    # delete
    def destroy(self, request, *args, **kwargs):
        """Delete a Customer instance."""
        instance = self.get_object()
        self.perform_destroy(instance)
        return Response(status=204)


class HouseViewSet(viewsets.ModelViewSet):
    """CRUD endpoints for houses belonging to the agent's customers."""
    queryset = House.objects.all()
    serializer_class = HouseSerializer
    permission_classes = [DebugOrJWTAuthenticated,
                          IsAgentOwner]

    def get_queryset(self):
        """Limit to houses whose customers are owned by the agent."""
        return self.queryset.filter(customer__agent=self.request.user)

    def perform_create(self, serializer):
        """Validate customer ownership and create the house record."""
        customer = serializer.validated_data["customer"]
        if customer.agent != self.request.user:
            raise PermissionDenied("You do not own this customer.")

        house_img = self.request.FILES.get("default_image")
        if not house_img:
            raise Exception("No default image uploaded")

        url = upload_file_to_bucket(
            house_img, bucket_name=os.getenv("BUCKET_NAME"))

        if not url:
            raise Exception("Failed to upload image")

        serializer.save(default_image=url)

    def perform_update(self, serializer):
        """Validate customer ownership on update as well."""
        customer = serializer.validated_data.get(
            "customer", serializer.instance.customer)
        if customer.agent != self.request.user:
            raise PermissionDenied("You do not own this customer.")

        serializer.save()

    def destroy(self, request, *args, **kwargs):
        """Delete a House instance."""
        instance = self.get_object()
        bucket_name = os.getenv("BUCKET_NAME")
        images = instance.images.all()
        for img in images:
            # Delete the file from cloud storage
            if img.image_url:
                try:
                    delete_file_from_bucket(img.image_url, bucket_name)
                except Exception as e:
                    print(f"[WARN] Failed to delete file from bucket: {e}")
        self.perform_destroy(instance)
        return Response(status=204)


class HouseImageViewSet(viewsets.ModelViewSet):
    """CRUD endpoints for house images, uploading to cloud storage on create."""
    queryset = HouseImage.objects.all()
    serializer_class = HouseImageSerializer
    permission_classes = [DebugOrJWTAuthenticated, IsAgentOwner]

    def get_queryset(self):
        """Limit to images under houses owned by the agent's customers."""
        return HouseImage.objects.filter(house__customer__agent=self.request.user)

    def destroy(self, request, *args, **kwargs):
        """Delete the HouseImage instance and its associated file from storage."""
        instance = self.get_object()
        bucket_name = os.getenv("BUCKET_NAME")

        # Delete the file from cloud storage
        if instance.image_url:
            try:
                # Assuming the function to delete a file from the bucket is defined
                delete_file_from_bucket(instance.image_url, bucket_name)
            except Exception as e:
                print(f"[WARN] Failed to delete file from bucket: {e}")

        # Delete the HouseImage instance from the database
        self.perform_destroy(instance)
        return Response(status=204)

    def perform_create(self, serializer):
        """Upload the provided file to storage and persist its URL.

        Expects a multipart field named `file` containing the image.
        """
        file_obj = self.request.FILES.get("file")

        if not file_obj:
            raise Exception("No file uploaded")

        url = upload_file_to_bucket(
            file_obj, bucket_name=os.getenv("BUCKET_NAME"))

        if not url:
            raise Exception("Failed to upload image")

        serializer.save(image_url=url)

    def perform_update(self, serializer):
        """
        On update, if a new file is provided, delete the old one from storage and upload the new one.
        """
        self.destroy(self.request, pk=serializer.instance.pk)
        self.perform_create(serializer)
        # serializer.save()


def redirect_404(request, exception):
    # Send 404s to the named login route (has trailing slash) to avoid loops
    return redirect('login')


@csrf_exempt
@api_view(["POST"])
@permission_classes([AllowAny])
@authentication_classes([])
def image_prediction(request, image_id):
    """Run RF-DETR inference on a single image by ID.

    Expects the same body parameters as `run_prediction`. This is a lower-level
    endpoint that can be used for testing or individual image inference.
    """
    try:
        img = HouseImage.objects.get(id=image_id)
    except HouseImage.DoesNotExist:
        return Response({"error": "Image not found"}, status=404)

    mode = request.data.get("mode", "normal")
    threshold = float(request.data.get("threshold", 0.4))
    tile_size = int(request.data.get("tile_size", 560))

    try:
        result = _run_prediction_for_image(img, mode, threshold, tile_size)
        return Response(result)
    except Exception as e:
        traceback.print_exc()
        return Response({
            "image_id": img.id,
            "original_image": img.image_url,
            "error": f"{type(e).__name__}: {e}"
        }, status=500)


@csrf_exempt
@api_view(["GET", "POST"])
@permission_classes([AllowAny])
@authentication_classes([])
def run_prediction(request, house_id):
    """Run RF-DETR inference for all images of the given house.

    Body (JSON):
    - `mode` (str): "normal" or "tiled" (default: "normal")
    - `threshold` (float): detection confidence threshold (default: 0.4)
    - `tile_size` (int): tile size if tiled mode used (default: 560)

    For each image, stores a predicted image in cloud storage and updates
    `predicted_url` and `predicted_at`. Returns a summary payload.

    """
    try:
        house = (
            House.objects
            .prefetch_related("images")
            .get(id=house_id)
        )
    except House.DoesNotExist:
        return Response({"error": "House not found"}, status=404)

    house_images = list(house.images.all())
    if not house_images:
        return Response({"message": "No images found for this house."})

    if request.method == "GET":
        # print("[INFO] house images:     ", house_images)
        return render(request, "backend/run_prediction.html", {"house_id": house_id, "images": house_images, "local_dev": settings.DEBUG})

    mode = request.data.get("mode", "normal")
    threshold = float(request.data.get("threshold", 0.4))
    tile_size = int(request.data.get("tile_size", 560))
    print(
        f"[INFO] Running prediction for house {house_id} with mode={mode}, threshold={threshold}, tile_size={tile_size}")
    results = []

    for img in house_images:
        try:
            results.append(
                _run_prediction_for_image(img, mode, threshold, tile_size)
            )
        except Exception as e:
            traceback.print_exc()
            results.append({
                "image_id": img.id,
                "original_image": img.image_url,
                "error": f"{type(e).__name__}: {e}"
            })

    return Response({
        "house_id": house_id,
        "total_images": len(results),
        "results": results
    })
