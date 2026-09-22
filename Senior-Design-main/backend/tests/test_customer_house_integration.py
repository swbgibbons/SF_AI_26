import pytest
from django.contrib.auth import get_user_model
from core.models import Customer, House

User = get_user_model()


@pytest.mark.django_db
def test_agent_creates_customer_and_house():
    """
    Integration test:
    Agent -> Customer -> House
    Verifies real DB relationships and foreign keys
    """

    # Create an agent (auth user)
    agent = User.objects.create_user(
        username="agent_test",
        password="password123"
    )

    # Agent creates a customer
    customer = Customer.objects.create(
        agent=agent,
        name="John Homeowner",
        email="john@example.com",
        phone="555-1234"
    )

    # Customer gets a house
    house = House.objects.create(
        customer=customer,
        address="123 Main St",
        description="Two-story house"
    )

    # Assertions (real DB reads)
    assert customer.agent == agent
    assert house.customer == customer
    assert customer.houses.count() == 1
    assert customer.houses.first().address == "123 Main St"
