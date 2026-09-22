# Backend Test Documentation

This document provides a comprehensive overview of all tests in the backend directory. The test suite uses **pytest** as the testing framework and follows Django testing conventions for database-related tests.

## Quick Reference

| Test File                                 | Test Count | Type             | Purpose                                   |
| ----------------------------------------- | ---------- | ---------------- | ----------------------------------------- |
| `test_api_customers.py`                   | 2          | API/Integration  | Customer API endpoints and authentication |
| `test_authentication.py`                  | 3          | Unit             | JWT cookie-based authentication           |
| `test_customer_house_integration.py`      | 1          | Integration      | Agent → Customer → House relationships    |
| `test_house_image_logging_integration.py` | 1          | Integration      | House image upload and audit logging      |
| `test_integration.py`                     | 2          | Integration      | Basic database connectivity               |
| `test_middleware.py`                      | 2+         | Unit             | Debug session user middleware             |
| `test_model_utils.py`                     | 3+         | Unit             | Image processing and model utilities      |
| `test_services.py`                        | 4+         | Unit             | RFDETR model prediction service           |
| `test_utils.py`                           | 8+         | Unit             | GCS file upload utilities                 |
| `test_views.py`                           | 3+         | Unit             | View helper functions                     |
| `core/tests/test_core.py`                 | 3+         | Unit/Integration | Core authentication and middleware        |

---

## Test File Details

### 1. `test_api_customers.py`

**Purpose:** Tests the Customer API endpoints and validates authentication/authorization.

**Tests:**

- **`test_customers_list_returns_only_authenticated_users_customers()`**
  - Creates two agents with different customers
  - Verifies that an authenticated agent only sees their own customers via the GET `/api/v1/customers/` endpoint
  - Confirms response status is 200 and customer list is filtered correctly
  - **Database:** ✓ Uses Django test DB

- **`test_customers_create_creates_customer_for_logged_in_agent()`**
  - Tests POST `/api/v1/customers/` endpoint
  - Verifies a logged-in agent can create a new customer
  - Confirms response status is 200 or 201
  - Validates the customer is created with the correct agent association
  - **Database:** ✓ Uses Django test DB

**Setup:** Uses `APIClient` and forces authentication with `client.force_authenticate()`

---

### 2. `test_authentication.py`

**Purpose:** Tests JWT authentication via HTTP cookies.

**Tests:**

- **`test_authenticate_returns_none_when_no_access_cookie()`**
  - Verifies that `CookieJWTAuthentication.authenticate()` returns `None` when no `access` cookie is present
  - Ensures unauthenticated requests are handled gracefully

- **`test_authenticate_returns_user_and_token(monkeypatch)`**
  - Mocks `get_validated_token()` and `get_user()` methods
  - Verifies that authentication returns a tuple of `(user, token)` when a valid `access` cookie is provided
  - Uses monkeypatch for dependency injection

- **`test_authenticate_raises_when_token_invalid(monkeypatch)`**
  - Mocks `get_validated_token()` to raise `InvalidToken`
  - Verifies that `AuthenticationFailed` exception is raised with message "Invalid or expired JWT"

**Key Dependencies:** `CookieJWTAuthentication` from `core.authentication`

---

### 3. `test_customer_house_integration.py`

**Purpose:** Integration test validating relationships between Agent → Customer → House entities.

**Tests:**

- **`test_agent_creates_customer_and_house()`**
  - Creates an agent (Django user)
  - Agent creates a customer
  - Customer gets a house assignment
  - Validates all foreign key relationships:
    - `customer.agent == agent`
    - `house.customer == customer`
    - `customer.houses.count() == 1` (reverse relationship)
    - House address is correctly stored
  - **Database:** ✓ Uses Django test DB with real queries

---

### 4. `test_house_image_logging_integration.py`

**Purpose:** Integration test for house image uploads and audit logging.

**Tests:**

- **`test_house_image_upload_creates_audit_log()`**
  - Creates an agent, customer, and house in the database
  - Simulates a house image upload with:
    - `image_url`: Original image URL
    - `predicted_url`: ML prediction output URL
  - Creates an `AgentCustomerLog` record for audit trail
  - Logs details including `house_id`
  - **Database:** ✓ Uses Django test DB
  - **Purpose:** Verifies that image uploads trigger proper audit/logging for compliance

---

### 5. `test_integration.py`

**Purpose:** Basic integration tests for database connectivity and test setup.

**Tests:**

- **`test_database_connection()`**
  - Simple smoke test verifying test database connection
  - Queries user count from the test database
  - Prints debug info: "Test DB has {count} users"
  - **Scope:** Ensures pytest and Django test infrastructure is working

- **`test_simple()`**
  - Trivial test asserting `True`
  - Used as a sanity check

---

### 6. `test_middleware.py`

**Purpose:** Tests custom middleware for debug session user handling.

**Tests:**

- **`test_debug_session_user_middleware_sets_user_when_debug_and_user_exists(monkeypatch)`**
  - Patches `settings.DEBUG = True`
  - Mocks `User.objects.get()` to return a fake user
  - Verifies middleware extracts email from session `debug_user` dict and sets `request.user`
  - Confirms the response passes through (`get_response()` is called)

- **`test_debug_session_user_middleware_ignores_missing_user(monkeypatch)`**
  - Patches `settings.DEBUG = True`
  - Mocks `User.objects.get()` to raise `User.DoesNotExist()` exception
  - Verifies middleware gracefully handles the case where debug user doesn't exist
  - Confirms request processing continues

**Middleware Class:** `DebugSessionUserMiddleware` from `core.middleware`

**Note:** This middleware is debug-only; it allows session-based user injection for local development.

---

### 7. `test_model_utils.py`

**Purpose:** Tests image processing utilities and ML model helper functions.

**Tests:**

- **`test_open_image_local(tmp_path)`**
  - Uses pytest's `tmp_path` fixture to create a temporary test image (10×10 RGB)
  - Calls `mu.open_image()` with a local file path
  - Verifies the loaded image has correct dimensions (10, 10)
  - **Covers:** Local file image loading

- **`test_open_image_url(monkeypatch)`**
  - Mocks `requests.get()` to return a fake JPEG image response
  - Calls `mu.open_image()` with a URL
  - Verifies the loaded image has correct dimensions (5, 5)
  - **Covers:** Remote/HTTP image loading

- **`test_sigmoid_basic()`**
  - Tests the sigmoid activation function: $\sigma(x) = \frac{1}{1+e^{-x}}$
  - Verifies `sigmoid(0.0) ≈ 0.5`
  - Ensures mathematical correctness

**Module:** `core.model_utils`

---

### 8. `test_services.py`

**Purpose:** Tests the RFDETR (Regional FCOS Detection with Transformers) model service for damage detection.

**Tests:**

- **`test_predict_normal(monkeypatch)`**
  - Mocks `_load_model()` to return a fake model, class names `["wind"]`, and model type
  - Mocks `_run_normal()` to return fake detections `["det"]` and output path
  - Calls `RFDETRService.predict("image.jpg", mode="normal", threshold=0.55)`
  - Verifies correct return values

- **`test_predict_tiled(monkeypatch)`**
  - Mocks `_load_model()` and `_download_image()` helpers
  - Mocks `_run_tiled()` for large image tiling inference
  - Calls `RFDETRService.predict()` with `mode="tiled"`, `tile_size=600`
  - Verifies tiling mode returns correct detections and output path

- **`test_load_model_returns_cached_model(monkeypatch)`**
  - Sets `RFDETRService._model = "MODEL"` to simulate cached model
  - Calls `_load_model()` and verifies:
    - Cached model is returned
    - Class names are `["wind", "hail"]`
    - Model type is `"ONNX"`
  - **Purpose:** Ensures model caching prevents redundant loads

- **`test_load_model_raises_when_base_dir_missing(monkeypatch)`**
  - Mocks `Path` to simulate missing `BASE_DIR` directory
  - Verifies `_load_model()` raises `RuntimeError("BASE_DIR does not exist")`

- **`test_load_model_raises_when_onnx_missing(monkeypatch)`**
  - Mocks `Path` to show `BASE_DIR` exists but ONNX model file missing
  - Verifies appropriate error is raised

**Service Class:** `RFDETRService` from `core.services`

**Use Case:** Roof damage detection (wind, hail) in insurance assessments

---

### 9. `test_utils.py`

**Purpose:** Tests utility functions for Google Cloud Storage (GCS) file uploads.

**Tests:**

- **`clean_env(monkeypatch)` (Fixture)**
  - Autouse fixture ensuring environment variables don't leak between tests
  - Cleans up `SKIP_CLOUD_UPLOAD` and `GCS_UPLOAD_TIMEOUT`

- **`test_upload_file_to_bucket_skips_when_env_set(monkeypatch)`**
  - Sets `SKIP_CLOUD_UPLOAD=1` environment variable
  - Verifies `upload_file_to_bucket()` returns `None` (skips upload)
  - **Purpose:** Allows local development without actual GCS calls

- **`test_upload_file_to_bucket_success(monkeypatch)`**
  - Mocks GCS client and bucket
  - Mocks `uuid.uuid4()` to return a fixed UUID for deterministic testing
  - Uploads a test file (name: "x.png", type: "image/png")
  - Verifies:
    - Correct GCS bucket name is used
    - Correct blob name is created (UUID-based)
    - File is uploaded successfully
    - Returns proper GCS URL: `https://storage.googleapis.com/{bucket}/{uuid}.png`

- **`test_upload_file_to_bucket_returns_none_if_client_init_fails(monkeypatch)`**
  - Mocks `get_gcs_client()` to raise an exception ("no creds")
  - Verifies `upload_file_to_bucket()` returns `None` (handles error gracefully)

- **`test_upload_local_file_to_bucket_missing_file_returns_none(monkeypatch)`**
  - Tests uploading a local file that doesn't exist
  - Verifies function returns `None` instead of crashing

**Additional Tests:** The file contains ~8+ tests covering various GCS scenarios

**Module:** `core.utils`

**Note:** Uses fake GCS objects (`FakeBucket`, `FakeBlob`, `FakeClient`) for testing without credentials

---

### 10. `test_views.py`

**Purpose:** Tests view helper functions, particularly image prediction cleanup.

**Tests:**

- **`test_delete_existing_prediction_returns_early_without_url(monkeypatch)`**
  - Creates a fake image object with `predicted_url=None`
  - Mocks `delete_file_from_bucket()` utility
  - Calls `_delete_existing_prediction(img)`
  - Verifies the delete function is **not called** early return (optimization)

- **`test_delete_existing_prediction_clears_fields_when_deleted(monkeypatch)`**
  - Creates a fake image with `predicted_url="https://example.com/pred.jpg"`
  - Sets `BUCKET_NAME` environment variable
  - Calls `_delete_existing_prediction(img)`
  - Verifies:
    - `predicted_url` is set to `None`
    - `predicted_at` is set to `None`
    - Image is saved with `update_fields=["predicted_url", "predicted_at"]`
    - **Purpose:** Cleanup ensures stale predictions don't clutter the database

- **`test_delete_existing_prediction_handles_delete_exception(monkeypatch)`**
  - Tests exception handling when GCS deletion fails
  - Ensures the view doesn't crash if file deletion from cloud storage fails

**Module:** `core.views`

**Use Case:** Cleanup when predicting new damage detection results

---

### 11. `core/tests/test_core.py`

**Purpose:** Core tests for authentication, middleware, and model functionality in a single comprehensive test file.

**Test Classes:**

- **`AuthenticationTests` (Django TestCase)**
  - `test_no_cookie_returns_none()`: Verifies no JWT cookie = no authentication
  - `test_invalid_token_raises_authentication_failed()`: Tests invalid JWT handling
  - `test_valid_token_returns_user_and_token()`: Tests successful JWT authentication with patching
  - Uses Django's `RequestFactory` and `patch` for mocking

**Note:** This file mirrors and supplements the unit tests in `test_authentication.py` using Django's TestCase instead of pytest

---

## Test Execution

### Running All Tests

```bash
pytest backend/
```

### Running a Specific Test File

```bash
pytest backend/test_api_customers.py
```

### Running a Specific Test

```bash
pytest backend/test_api_customers.py::test_customers_list_returns_only_authenticated_users_customers
```

### Running Tests with Coverage

```bash
pytest backend/ --cov=core --cov-report=html
```

### Running Tests with Verbose Output

```bash
pytest backend/ -v
```

---

## Test Configuration

### pytest.ini

Located at `backend/pytest.ini`, configures:

- Django settings module
- Test discovery patterns
- Database setup

### conftest.py

Located at `backend/conftest.py`, provides:

- Pytest fixtures for Django tests
- Monkeypatch utility functions
- Database test setup

### Sample Environment

`backend/sample.env` provides example environment variables needed for tests:

- `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` for database
- `BUCKET_NAME` for GCS testing
- `SKIP_CLOUD_UPLOAD` to bypass cloud operations

---

## Testing Patterns & Best Practices

### Markers Used

- **`@pytest.mark.django_db`**: Marks tests that require database access
- Database is rolled back after each test automatically

### Mocking & Monkeypatching

- Uses `monkeypatch` fixture for:
  - Environment variable manipulation
  - Module function/class patching
  - Dependency injection
- Uses `unittest.mock.patch()` for class method mocking

### Fixtures

- **`tmp_path`**: Pytest built-in for temporary files (image tests)
- **`monkeypatch`**: Pytest built-in for runtime patching
- **`clean_env`**: Custom fixture in `test_utils.py` for environment isolation

### Fake Objects

Tests use lightweight fake implementations:

- `FakeBlob`, `FakeBucket`, `FakeClient` for GCS testing
- `NameSpace`/`SimpleNamespace` for mocking request objects
- Class methods patched with lambda functions for model services

---

## Key Test Dependencies

| Module                          | Used By           | Purpose                            |
| ------------------------------- | ----------------- | ---------------------------------- |
| `django.contrib.auth`           | All auth tests    | Django user model                  |
| `rest_framework.test.APIClient` | API tests         | REST endpoint testing              |
| `unittest.mock`                 | Service tests     | Mocking and patching               |
| `pytest`                        | All tests         | Testing framework                  |
| `core.models`                   | Integration tests | Customer, House, HouseImage models |
| `core.services`                 | Service tests     | RFDETR model prediction            |
| `core.utils`                    | Utils tests       | GCS file upload                    |

---

## Coverage Notes

Each test module targets a specific layer:

- **API Layer**: `test_api_customers.py`, `test_authentication.py`
- **Integration Layer**: `test_customer_house_integration.py`, `test_house_image_logging_integration.py`, `test_integration.py`
- **Middleware/View Layer**: `test_middleware.py`, `test_views.py`, `core/tests/test_core.py`
- **Model/Service Layer**: `test_model_utils.py`, `test_services.py`
- **Utility Layer**: `test_utils.py`

---

## Running Tests in Docker

When running inside the container:

```bash
cd /workspaces/Senior-Design/backend
pytest
```

Tests automatically use the test database configuration from `pytest.ini` and `conftest.py`.

---

## Troubleshooting

### Test Database Issues

- Ensure `DB_HOST`, `DB_NAME` are set correctly in environment
- Check `conftest.py` for database fixture setup

### GCS Credential Issues

- Set `SKIP_CLOUD_UPLOAD=1` to bypass GCS in tests
- Or provide valid GCS credentials for integration tests

### Import Errors

- Ensure backend is on `sys.path` (see `core/tests/test_core.py` for example)
- Check `PYTHONPATH` includes project root

---

## Last Updated

May 1, 2026

For additional information about specific test implementations, refer to individual test files in the `backend/` directory.
