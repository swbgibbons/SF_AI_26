from rest_framework import serializers
from .models import Customer, House, HouseImage, AgentCustomerLog

# this serializer file defines serializers for the models, meaning that it converts model instances to JSON and vice versa, which is rules for how data is sent and received through the API. It also includes some custom validation and logic for handling file uploads and restricting querysets based on the current user, ensuring that agents can only access their own customers and houses.


class HouseImageSerializer(serializers.ModelSerializer):
    file = serializers.ImageField(write_only=True)

    class Meta:
        model = HouseImage
        fields = ["id", "house", "image_url", "file",
                  "uploaded_at", "predicted_url", "predicted_at"]
        read_only_fields = ["id", "image_url",
                            "uploaded_at", "predicted_url", "predicted_at"]

    def create(self, validated_data):
        validated_data.pop("file", None)  # remove non-model field
        return super().create(validated_data)

    def __init__(self, *args, **kwargs):
        """
        Restricts the selectable house foreign key options inside this serializer to houses owned by the current user.
        """
        super().__init__(*args, **kwargs)
        request = self.context.get("request")
        if request:
            # Only allow houses owned by this agent to be selected
            self.fields["house"].queryset = House.objects.filter(
                customer__agent=request.user
            )


class HouseSerializer(serializers.ModelSerializer):
    images = HouseImageSerializer(many=True, read_only=True)
    default_image = serializers.ImageField(write_only=True)

    roof_type_list = [
        ("Asphalt Shingles", "Asphalt Shingles"),
        ("Slate Shingles", "Slate Shingles"),
        ("Metal", "Metal"),
        ("Clay Tiles", "Clay Tiles"),
        ("Synthetic/Composite", "Synthetic/Composite"),
        ('Wood', 'Wood'),
        ("Other", "Other"),
    ]
    roof_type = serializers.ChoiceField(
        choices=roof_type_list, allow_blank=False, required=True
    )

    class Meta:
        model = House
        fields = ["id", "customer", "address",
                  "roof_type", "severity", "damage_types", "description", "created_at", "images", "price_estimate", "default_image"]
        read_only_fields = ["id", "created_at", "images"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        request = self.context.get("request")
        if request:
            # Only allow customers owned by this agent
            self.fields["customer"].queryset = Customer.objects.filter(
                agent=request.user
            )

    def create(self, validated_data):
        validated_data.pop("default_image", None)  # remove non-model field
        return super().create(validated_data)

    # def validate(self, address):


class CustomerSerializer(serializers.ModelSerializer):
    houses = HouseSerializer(many=True, read_only=True)

    class Meta:
        model = Customer
        fields = ["id", "agent", "name", "email",
                  "phone", "created_at", "houses"]
        read_only_fields = ["id", "created_at", "agent", "houses"]


class AgentCustomerLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentCustomerLog
        fields = ["id", "agent", "customer", "action", "timestamp", "details"]
        read_only_fields = ["id", "timestamp"]
