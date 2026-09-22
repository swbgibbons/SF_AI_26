from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register("customers", views.CustomerViewSet, basename="customer")
router.register("houses", views.HouseViewSet, basename="house")
router.register("house-images", views.HouseImageViewSet, basename="houseimage")
router.register("agent-logs", views.AgentCustomerLogViewSet,
                basename="agentlog")

urlpatterns = [
    path("", views.sign_in, name="home"),
    # Accept both with and without trailing slash to avoid 404->redirect loops
    path("v1/", include(router.urls)),
    path("login/", views.sign_in, name="login"),
    path("google/auth/", views.auth_receive, name="google_auth"),
    path("sign-out/", views.sign_out, name="sign_out"),
    path("login/google/modal/", views.google_login_modal,
         name="google_login_modal"),
    path("csrf/", views.get_csrf, name="get_csrf"),
    path("v1/houses/<int:house_id>/predict/",
         views.run_prediction, name="run_prediction"),
    path("houses/<int:house_id>/run_prediction/",
         views.run_prediction, name="run_prediction_alt"),
    path("v1/<int:image_id>/predict/",
         views.image_prediction, name="image_prediction"),
]
