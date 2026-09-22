# Core App Guide

This file explains what each main part of the `core` app does.

## What this app is responsible for

The `core` app contains the main backend business logic for:

- sign-in and sign-out behavior
- Google OAuth callback handling
- CSRF cookie setup
- customer, house, and house image API endpoints
- prediction endpoints for house images
- authentication and permission checks
- storage upload and delete helpers used by the views

## Files and what they do

### `views.py`

This is the main request-handling file.

- `get_csrf` sets a CSRF cookie for the frontend.
- `sign_in` renders the login page.
- `google_login_modal` renders the Google login modal.
- `auth_receive` handles the Google login callback, creates or finds the user, and sets JWT cookies.
- `sign_out` clears auth cookies and redirects to login.
- `index` is a simple session/auth check endpoint.
- `AgentCustomerLogViewSet` returns the authenticated agent's logs.
- `CustomerViewSet` handles customer CRUD.
- `HouseViewSet` handles house CRUD and uploads the default image.
- `HouseImageViewSet` handles image CRUD and uploads image files.
- `image_prediction` runs prediction for one image.
- `run_prediction` runs prediction for all images in a house.

Internal helper functions:

- `_delete_existing_prediction` removes a previous prediction file from storage and clears saved prediction fields.
- `_run_prediction_for_image` performs the prediction flow for one image and saves results.

Permission classes:

- `DebugOrJWTAuthenticated` allows access from either a debug session or a valid JWT user.
- `IsAgentOwner` makes sure the current user owns the object being accessed.

### `urls.py`

This file maps URL paths to the views in `views.py`.

Important routes:

- `/login/` -> sign-in page
- `/google/auth/` -> Google OAuth callback
- `/sign-out/` -> clears auth cookies
- `/csrf/` -> CSRF cookie endpoint
- `/v1/` -> REST API routes for customers, houses, house images, and logs
- `/v1/houses/<house_id>/predict/` -> prediction for a whole house
- `/v1/<image_id>/predict/` -> prediction for one image

### `authentication.py`

Contains custom JWT authentication.

- `CookieJWTAuthentication` reads the access token from the `access` cookie.
- If the token is valid, it sets the authenticated user.
- If the token is missing, it allows the request to continue unauthenticated.

### `middleware.py`

Contains request middleware used during development and JWT handling.

- `DebugSessionUserMiddleware` can set `request.user` from a debug session user.
- `JWTAuthMiddleware` reads the JWT from cookies and loads the user.

### `models.py`

Defines the database models.

Typical responsibilities:

- customer records
- house records
- house image records
- agent log records

### `serializers.py`

Converts model objects to and from JSON for the API.

- `CustomerSerializer`
- `HouseSerializer`
- `HouseImageSerializer`
- `AgentCustomerLogSerializer`

### `services.py`

Contains reusable service logic.

This is where the prediction service lives and where the model inference logic is separated from the views.

### `utils.py`

Contains helper functions used across the app.

Common responsibilities:

- upload files to cloud storage
- upload local prediction files to storage
- delete files from storage

### `model_utils.py`

Contains model-related helper functions used by inference or data handling.

### `apps.py`

Registers the Django app configuration.

## Request flow overview

### Login flow

1. The frontend opens the sign-in page.
2. Google returns a credential to `auth_receive`.
3. The backend verifies the token.
4. The backend issues JWT cookies.
5. The user is redirected to the frontend.

### API flow

1. The frontend sends a request to `/api/v1/...`.
2. Authentication is handled by the cookie JWT auth class or debug session middleware.
3. Permissions check whether the user is allowed to access the resource.
4. ViewSets return or modify model data.

### Prediction flow

1. A house or image is requested for prediction.
2. The backend loads the image path or URL.
3. The model service runs inference.
4. The predicted result is uploaded to storage.
5. The database is updated with prediction metadata.

## Where to look first

If you want to understand the app quickly, read in this order:

1. `urls.py`
2. `views.py`
3. `authentication.py`
4. `middleware.py`
5. `serializers.py`
6. `models.py`

## Notes

- The main API entrypoints are in `views.py`.
- The actual route wiring is in `urls.py`.
- The prediction code is intentionally split so the heavy model logic stays out of the request handlers as much as possible.

## Tests (core/tests/test_core.py)

What the tests cover:

- Authentication flows and `CookieJWTAuthentication` behavior
- `DebugSessionUserMiddleware` and `JWTAuthMiddleware` middleware
- Utility helpers in `model_utils.py` (sigmoid, box conversion, NMS)
- `services.RFDETRService` download/load behavior (model loading, image download)
- View helpers and endpoints in `views.py` (CSRF, sign-in/out, prediction flows)

How to run the `core` tests locally (Windows PowerShell):

```powershell
cd "F:\senior design\back-end\backend"
# activate your venv, for example:
.\.venv\Scripts\Activate
$env:DJANGO_SETTINGS_MODULE='backend.settings'
pytest core/tests/test_core.py -q
```

Unix / bash:

```bash
cd "F:/senior design/back-end/backend"
python -m venv .venv
source .venv/bin/activate
export DJANGO_SETTINGS_MODULE=backend.settings
pytest core/tests/test_core.py -q
```

Notes and prerequisites:

- Ensure a virtual environment is active and the project dependencies are installed (Django, DRF, pytest, pytest-django, Pillow).

```powershell
pip install django djangorestframework pytest pytest-django pillow
```

- Run the tests from the `backend` folder so the `core` package imports resolve cleanly.
- The single test file `core/tests/test_core.py` is intentionally comprehensive and uses mocking for external services (requests, model predict/upload/delete) so it can run without cloud access or the real model binary.
