import pytest

@pytest.mark.django_db
def test_database_connection():
    """Test that we can connect to the test database"""
    from django.contrib.auth.models import User
    
    # This uses the test database
    count = User.objects.count()
    assert count >= 0
    print(f"Test DB has {count} users")

def test_simple():
    """Simple test to verify pytest works"""
    assert True
