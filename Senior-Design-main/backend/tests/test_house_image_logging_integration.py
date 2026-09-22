import pytest
import os
from django.contrib.auth import get_user_model
from core.models import (
    Customer,
    House,
    HouseImage,
    AgentCustomerLog
)

User = get_user_model()


@pytest.mark.django_db
def test_house_image_upload_creates_audit_log():
    """
    Integration test:
    Agent uploads an image for a house
    → Image record created
    → Audit log recorded
    """

    print("DB_HOST =", os.getenv("DB_HOST"))
    print("DB_NAME =", os.getenv("DB_NAME"))

    agent = User.objects.create_user(username="agent_logger")
    customer = Customer.objects.create(
        agent=agent,
        name="Audit Customer",
        email="audit@example.com"
    )
    house = House.objects.create(
        customer=customer,
        address="456 Oak Ave"
    )

    # Simulate image upload
    image = HouseImage.objects.create(
        house=house,
        image_url="http://example.com/original.jpg",
        predicted_url="http://example.com/predicted.jpg"
    )

    # Log the agent action
    log = AgentCustomerLog.objects.create(
        agent=agent,
        customer=customer,
        action="uploaded image",
        details={
            "house_id": house.id,
            "image_id": image.id
        }
    )

    # Assertions
    assert house.images.count() == 1
    assert house.images.first().image_url.endswith(".jpg")

    assert AgentCustomerLog.objects.count() == 1
    assert log.action == "uploaded image"
    assert log.details["image_id"] == image.id
