import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pytest

# ---------------------------------------------------------
# Load test environment variables BEFORE Django starts
# ---------------------------------------------------------
load_dotenv(".env.test", override=True)

@pytest.fixture(autouse=True, scope="session")
def prevent_production_database_usage():
    prod_hosts = {"34.171.229.5"}
    prod_db_names = {"roofvision"}

    assert os.getenv("DB_HOST") not in prod_hosts, (
        "Tests attempted to run against PRODUCTION DB HOST"
    )
    assert os.getenv("DB_NAME") not in prod_db_names, (
        "Tests attempted to run against PRODUCTION DB NAME"
    )


# Ensure backend is on Python path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))


# Initialize Django
import django
from django.conf import settings

if not settings.configured:
    django.setup()
