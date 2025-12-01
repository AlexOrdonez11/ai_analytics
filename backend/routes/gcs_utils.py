import os
import uuid
import io
from google.cloud import storage

GCS_BUCKET = os.getenv("GCS_BUCKET", "analytics-ai-bucket")
_storage_client = storage.Client()

def new_image_path(prefix: str, ext: str = "png") -> str:
    return f"{prefix.rstrip('/')}/{uuid.uuid4().hex}.{ext}"

def upload_image_and_get_url(data: bytes, path: str, content_type: str = "image/png") -> str:
    """
    Uploads an image as PUBLIC and returns its public URL.
    """
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET env var not set")

    bucket = _storage_client.bucket(GCS_BUCKET)
    blob = bucket.blob(path)

    # Upload the image
    blob.upload_from_file(
        io.BytesIO(data),
        content_type=content_type,
    )
    
    # Make it public
    blob.make_public()

    return blob.public_url
