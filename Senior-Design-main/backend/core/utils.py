from google.cloud import storage
import uuid
import os
from typing import Optional

"""
Serve as a utility module for handling Google Cloud Storage (GCS) interactions, including uploading and deleting files. This module abstracts away the complexities of GCS operations, providing simple functions to upload files from both file-like objects and local file paths, as well as a function to delete files from a GCS bucket. It also includes error handling and logging to ensure that any issues during these operations are properly reported.
"""


def get_gcs_client() -> storage.Client:
    """
    Create a GCS client using Application Default Credentials (ADC).

    This works with:
    - GOOGLE_APPLICATION_CREDENTIALS (file path)
    - Attached service accounts (Cloud Run / GKE / GCE)
    - Local gcloud ADC
    """
    return storage.Client()


def upload_file_to_bucket(file_obj, bucket_name: str) -> Optional[str]:
    """
    Upload a file-like object to a Google Cloud Storage bucket
    and return its public URL.

    Args:
        file_obj: File-like object (must have .name and .content_type)
        bucket_name: GCS bucket name

    Returns:
        Public URL string or None if upload failed
    """
    if os.getenv("SKIP_CLOUD_UPLOAD") == "1":
        print("[INFO] SKIP_CLOUD_UPLOAD=1; skipping GCS upload")
        return None

    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print("[ERROR] GCS client initialization failed:", e)
        return None

    ext = os.path.splitext(file_obj.name)[1]
    unique_name = f"{uuid.uuid4()}{ext}"

    try:
        blob = bucket.blob(unique_name)
        blob.upload_from_file(
            file_obj,
            content_type=getattr(file_obj, "content_type", None),
            timeout=float(os.getenv("GCS_UPLOAD_TIMEOUT", "30")),
        )

        return f"https://storage.googleapis.com/{bucket_name}/{unique_name}"

    except Exception as e:
        print("[ERROR] GCS upload failed:", e)
        return None


def upload_local_file_to_bucket(local_path: str, bucket_name: str) -> Optional[str]:
    """
    Upload a local file to a Google Cloud Storage bucket
    and return its public URL.
    """
    if os.getenv("SKIP_CLOUD_UPLOAD") == "1":
        print("[INFO] SKIP_CLOUD_UPLOAD=1; skipping GCS upload:", local_path)
        return None

    if not os.path.isfile(local_path):
        print("[ERROR] Local file does not exist:", local_path)
        return None

    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print("[ERROR] GCS client initialization failed:", e)
        return None

    ext = os.path.splitext(local_path)[1]
    unique_name = f"{uuid.uuid4()}{ext}"

    try:
        blob = bucket.blob(unique_name)
        blob.upload_from_filename(
            local_path,
            timeout=float(os.getenv("GCS_UPLOAD_TIMEOUT", "30")),
        )

        return f"https://storage.googleapis.com/{bucket_name}/{unique_name}"

    except Exception as e:
        print("[ERROR] GCS upload failed:", e)
        return None


def delete_file_from_bucket(file_url: str, bucket_name: str) -> bool:
    """
    Delete a file from a GCS bucket given its public URL.

    Args:
        file_url: Public GCS URL
        bucket_name: GCS bucket name

    Returns:
        True if deletion succeeded, False otherwise
    """
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
    except Exception as e:
        print("[ERROR] GCS client initialization failed:", e)
        return False

    try:
        blob_name = file_url.split("/")[-1]
        blob = bucket.blob(blob_name)
        blob.delete()
        return True

    except Exception as e:
        print("[ERROR] GCS deletion failed:", e)
        return False
