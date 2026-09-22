import pytest
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from core.models import Customer

User = get_user_model()

CUSTOMERS_URL = "/api/v1/customers/"


@pytest.mark.django_db
def test_customers_list_returns_only_authenticated_users_customers():
    agent1 = User.objects.create_user(username="agent1", password="pass1234")
    agent2 = User.objects.create_user(username="agent2", password="pass1234")

    Customer.objects.create(agent=agent1, name="A1 Cust", email="a1@example.com", phone="+15550000001")
    Customer.objects.create(agent=agent2, name="A2 Cust", email="a2@example.com", phone="+15550000002")

    client = APIClient()
    client.force_authenticate(user=agent1)

    resp = client.get(CUSTOMERS_URL, HTTP_ACCEPT="application/json")
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, list)

    emails = {item.get("email") for item in data}
    assert "a1@example.com" in emails
    assert "a2@example.com" not in emails


@pytest.mark.django_db
def test_customers_create_creates_customer_for_logged_in_agent():
    agent = User.objects.create_user(username="agent1", password="pass1234")

    client = APIClient()
    client.force_authenticate(user=agent)

    payload = {"name": "John Homeowner", "email": "john@example.com", "phone": "+14155552671",}

    resp = client.post(CUSTOMERS_URL, payload, format="json", HTTP_ACCEPT="application/json")

    if resp.status_code not in (200, 201):
        print("STATUS:", resp.status_code)
        print("DATA:", resp.data)

    assert resp.status_code in (200, 201)
    assert Customer.objects.filter(agent=agent, email="john@example.com").exists()
