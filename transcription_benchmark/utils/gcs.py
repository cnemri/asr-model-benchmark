"""Utility functions for parsing transcription results from GCS."""

import re
import logging
from google.cloud import storage

logger = logging.getLogger(__name__)


def list_gcs_files(gcs_uri: str, file_extension: str = None) -> list[str]:
    """
    Lists all files in a GCS folder, optionally filtering by file extension.
    Excludes empty "directory" objects.
    """
    match = re.match(r"gs://([^/]+)/(.*)", gcs_uri)
    if not match:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")

    bucket_name, prefix = match.groups()
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    # Filter out directory placeholders
    files = [f"gs://{bucket_name}/{blob.name}" for blob in blobs if not blob.name.endswith('/')]
    
    if file_extension:
        return [f for f in files if f.endswith(file_extension)]
    
    return files
