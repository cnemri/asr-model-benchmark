"""Ground truth parser for transcription benchmark."""

import io
import logging
import pandas as pd
from transcription_benchmark.parsing.base_parser import BaseParser
from transcription_benchmark.utils.gcs import list_gcs_files
from google.cloud import storage
import re

logger = logging.getLogger(__name__)


class GroundTruthParser(BaseParser):
    """Parser for ground truth files from GCS."""

    def _parse_content(self, content: str) -> str:
        """
        Parse ground truth content.

        Handles two formats:
        1. Simple text file with single sentence (current format)
        2. TSV format with dialogue (time, speaker_id, gender, text)

        Args:
            content: File content as string

        Returns:
            Parsed transcript text
        """
        content = content.strip()

        # Check if it's TSV format (has tabs)
        if '\t' in content:
            try:
                df = pd.read_csv(
                    io.StringIO(content),
                    sep='\t',
                    header=None,
                    engine='python',
                    names=['time', 'speaker_id', 'gender', 'text']
                )
                df['speaker_id'] = df['speaker_id'].astype(str)
                dialogue_df = df[df['speaker_id'] != '0']
                return " ".join(dialogue_df['text'].astype(str))
            except Exception as e:
                logger.warning(f"Failed to parse as TSV, treating as plain text: {e}")
                return content
        else:
            # Simple text format - just return the content
            return content

    def parse(self, gcs_folder_path: str) -> dict:
        """
        Parse ground truth files from GCS.

        Args:
            gcs_folder_path: GCS URI to folder containing .txt files

        Returns:
            Dictionary mapping recording_N to transcript text

        Raises:
            ValueError: If GCS path is invalid or no files found
        """
        logger.info(f"Parsing ground truth files from GCS: {gcs_folder_path}")
        all_transcripts = {}

        storage_client = storage.Client()
        match = re.match(r"gs://([^/]+)/(.+)", gcs_folder_path)
        if not match:
            raise ValueError(f"Invalid GCS URI: {gcs_folder_path}")

        bucket_name, prefix = match.groups()
        bucket = storage_client.bucket(bucket_name)

        files_to_process = sorted([
            f"gs://{bucket_name}/{blob.name}"
            for blob in bucket.list_blobs(prefix=prefix)
            if blob.name.endswith('.txt')
        ])

        if not files_to_process:
            raise ValueError(f"No .txt files found in {gcs_folder_path}")

        logger.info(f"Found {len(files_to_process)} ground truth files")

        for i, gcs_uri in enumerate(files_to_process):
            try:
                blob_name = gcs_uri.replace(f"gs://{bucket_name}/", "")
                blob = bucket.blob(blob_name)
                content = blob.download_as_string().decode('utf-8')
                transcript = self._parse_content(content)
                if transcript:
                    all_transcripts[f"recording_{i}"] = transcript
                else:
                    logger.warning(f"Empty transcript for {gcs_uri}")
            except Exception as e:
                logger.error(f"Error processing {gcs_uri}: {e}")

        logger.info(f"Successfully parsed {len(all_transcripts)} ground truth files")
        return all_transcripts
