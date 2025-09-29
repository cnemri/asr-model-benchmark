"""Chirp transcription result parser."""

import json
import re
import logging
from google.cloud import storage
from transcription_benchmark.parsing.base_parser import BaseParser
from transcription_benchmark.utils.gcs import list_gcs_files

logger = logging.getLogger(__name__)


class ChirpParser(BaseParser):
    """Parser for Chirp transcription results."""

    def _parse_response(self, chirp_json_path: str) -> str:
        content = ""
        if chirp_json_path.startswith("gs://"):
            try:
                storage_client = storage.Client()
                match = re.match(r"gs://([^/]+)/(.+)", chirp_json_path)
                bucket_name, blob_name = match.groups()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                content = blob.download_as_string()
            except Exception as e:
                logger.error(f"Error downloading {chirp_json_path}: {e}")
                return ""
        else:
            with open(chirp_json_path, 'r', encoding='utf-8') as f:
                content = f.read()

        data = json.loads(content)
        if not data.get("results") or not data["results"][0].get("alternatives"):
            logger.warning(f"No results found in Chirp response for {chirp_json_path}")
            return ""
        return data["results"][0]["alternatives"][0].get("transcript", "")

    def parse(self, gcs_folder_path: str) -> dict:
        """
        Parse Chirp batch transcription results.

        Args:
            gcs_folder_path: GCS URI to folder containing JSON result files

        Returns:
            Dictionary mapping recording_N to transcript text
        """
        logger.info(f"Parsing Chirp result files from {gcs_folder_path}")
        json_files = sorted(list_gcs_files(gcs_folder_path, file_extension=".json"))

        if not json_files:
            logger.warning(f"No JSON files found in {gcs_folder_path}")
            return {}

        logger.info(f"Found {len(json_files)} Chirp result files")
        all_transcripts = {}

        for i, gcs_uri in enumerate(json_files):
            transcript = self._parse_response(gcs_uri)
            if transcript:
                all_transcripts[f"recording_{i}"] = transcript
            else:
                logger.warning(f"Empty transcript for {gcs_uri}")

        logger.info(f"Successfully parsed {len(all_transcripts)} Chirp transcripts")
        return all_transcripts
