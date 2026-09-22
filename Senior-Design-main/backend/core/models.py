from django.conf import settings
from django.db import models
from phonenumber_field.modelfields import PhoneNumberField
# uv run python manage.py graph_models -a -g -o erd.png
#  to visualize the ERD


class Customer(models.Model):
    agent = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="customers"
    )
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True, default="johndoe@example.com")
    phone = PhoneNumberField(blank=True, default="+123-456-7890", unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} ({self.email})"


class House(models.Model):
    customer = models.ForeignKey(
        Customer,
        on_delete=models.CASCADE,
        related_name="houses"
    )
    address = models.CharField(max_length=255, unique=True)
    roof_type = models.CharField(max_length=100, blank=True, null=True)
    severity = models.IntegerField(blank=True, null=True)
    damage_types = models.JSONField(blank=True, null=True)
    price_estimate = models.DecimalField(
        max_digits=12, decimal_places=2, blank=True, null=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    default_image = models.URLField(blank=True, null=True)

    def __str__(self):
        return f"House at {self.address} for {self.customer.name}"


class HouseImage(models.Model):
    house = models.ForeignKey(
        House,
        on_delete=models.CASCADE,
        related_name="images"  # access with house.images.all()
    )
    image_url = models.URLField()
    uploaded_at = models.DateTimeField(auto_now_add=True)
    predicted_url = models.URLField(blank=True, null=True)
    predicted_at = models.DateTimeField(blank=True, null=True)
    detections = models.JSONField(blank=True, null=True)

    def __str__(self):
        return f"Image for {self.house.address}"


class AgentCustomerLog(models.Model):
    agent = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="customer_logs"
    )
    customer = models.ForeignKey(
        Customer,
        on_delete=models.CASCADE,
        related_name="agent_logs"
    )
    # e.g., "viewed profile", "uploaded image"
    action = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    details = models.JSONField(blank=True, null=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.agent.username} {self.action} {self.customer.name} at {self.timestamp}"
