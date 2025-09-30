#!/usr/bin/env python3
"""Upload local dataset to Google Cloud Storage bucket."""

import os
import sys
import logging
from pathlib import Path
from google.cloud import storage

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from transcription_benchmark.utils.logging import setup_logger
from transcription_benchmark.utils.config import load_config

logger = logging.getLogger(__name__)


def upload_directory_to_gcs(
    local_dir: Path,
    bucket_name: str,
    gcs_prefix: str,
    file_extension: str = None
) -> int:
    """
    Upload all files from a local directory to GCS.

    Args:
        local_dir: Local directory containing files
        bucket_name: GCS bucket name
        gcs_prefix: Destination prefix in GCS (folder path)
        file_extension: Optional filter (e.g., '.mp3')

    Returns:
        Number of files uploaded
    """
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Get all files
    if file_extension:
        files = list(local_dir.glob(f"*{file_extension}"))
    else:
        files = [f for f in local_dir.iterdir() if f.is_file()]

    if not files:
        logger.warning(f"No files found in {local_dir}")
        return 0

    logger.info(f"Found {len(files)} files to upload")

    uploaded = 0
    for file_path in files:
        blob_name = f"{gcs_prefix}{file_path.name}"
        blob = bucket.blob(blob_name)

        # Skip if already exists (optional optimization)
        # if blob.exists():
        #     logger.debug(f"Skipping existing file: {blob_name}")
        #     continue

        logger.info(f"Uploading {file_path.name} → gs://{bucket_name}/{blob_name}")
        blob.upload_from_filename(str(file_path))
        uploaded += 1

    return uploaded


def main():
    """Upload dataset to GCS bucket configured in config.yaml."""
    # Setup logging
    setup_logger(level=logging.INFO)

    logger.info("="*60)
    logger.info("Dataset Upload to Google Cloud Storage")
    logger.info("="*60)

    # Load configuration
    try:
        config = load_config("config.yaml")
    except FileNotFoundError:
        logger.error("config.yaml not found. Please create it from config.yaml.example")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Set GCS project
    os.environ['GOOGLE_CLOUD_PROJECT'] = config.google_cloud.project_id

    # Define local dataset paths
    dataset_root = project_root / "dataset"
    audio_dir = dataset_root / "MP3"
    groundtruth_dir = dataset_root / "TXT"

    if not dataset_root.exists():
        logger.error(f"Dataset directory not found: {dataset_root}")
        logger.info("Please create dataset/ directory with MP3/ and TXT/ subdirectories")
        sys.exit(1)

    bucket_name = config.gcs.bucket

    # Upload audio files
    if audio_dir.exists():
        logger.info(f"\nUploading audio files from {audio_dir}")
        try:
            count = upload_directory_to_gcs(
                local_dir=audio_dir,
                bucket_name=bucket_name,
                gcs_prefix=config.gcs.audio_folder,
                file_extension=config.audio.format
            )
            logger.info(f"✓ Uploaded {count} audio files to gs://{bucket_name}/{config.gcs.audio_folder}")
        except Exception as e:
            logger.error(f"✗ Failed to upload audio files: {e}")
    else:
        logger.warning(f"Audio directory not found: {audio_dir}")

    # Upload ground truth files
    if groundtruth_dir.exists():
        logger.info(f"\nUploading ground truth files from {groundtruth_dir}")
        try:
            count = upload_directory_to_gcs(
                local_dir=groundtruth_dir,
                bucket_name=bucket_name,
                gcs_prefix=config.gcs.groundtruth_folder,
                file_extension=".txt"
            )
            logger.info(f"✓ Uploaded {count} ground truth files to gs://{bucket_name}/{config.gcs.groundtruth_folder}")
        except Exception as e:
            logger.error(f"✗ Failed to upload ground truth files: {e}")
    else:
        logger.warning(f"Ground truth directory not found: {groundtruth_dir}")

    logger.info("\n" + "="*60)
    logger.info("Upload complete!")
    logger.info("="*60)
    logger.info(f"\nVerify with:")
    logger.info(f"  gsutil ls gs://{bucket_name}/{config.gcs.audio_folder} | head")
    logger.info(f"  gsutil ls gs://{bucket_name}/{config.gcs.groundtruth_folder} | head")


if __name__ == "__main__":
    main()
