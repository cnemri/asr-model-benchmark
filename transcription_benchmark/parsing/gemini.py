"""Gemini transcription result parser."""

import json
import re
import logging
from google.cloud import storage
from transcription_benchmark.parsing.base_parser import BaseParser
from transcription_benchmark.utils.gcs import list_gcs_files

logger = logging.getLogger(__name__)


class GeminiParser(BaseParser):
    """Parser for Gemini transcription batch results."""

    def parse(self, gcs_folder_path: str) -> dict:
        """
        Parse Gemini batch transcription results.

        Args:
            gcs_folder_path: GCS URI to folder containing JSONL result files

        Returns:
            Dictionary mapping recording_N to transcript text
        """
        logger.info(f"Parsing Gemini result files from {gcs_folder_path}")

        all_files = list_gcs_files(gcs_folder_path)
        if not all_files:
            logger.warning(f"No result files found in {gcs_folder_path}")
            return {}

        all_lines = []
        storage_client = storage.Client()

        for file_uri in all_files:
            try:
                match = re.match(r"gs://([^/]+)/(.+)", file_uri)
                bucket_name, blob_name = match.groups()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                content = blob.download_as_string().decode('utf-8')
                all_lines.extend(content.splitlines())
            except Exception as e:
                logger.error(f"Error downloading or reading {file_uri}: {e}")
                continue

        if not all_lines:
            logger.warning("No content found in result files")
            return {}

        transcripts = {}
        lines = all_lines
        sorted_lines = sorted(lines, key=lambda line: json.loads(line)['request']['contents'][0]['parts'][1]['fileData']['fileUri'])

        for i, line in enumerate(sorted_lines):
            try:
                data = json.loads(line)
                if data.get('response') and data['response']['candidates']:
                    response_str = data['response']['candidates'][0]['content']['parts'][0]['text']

                    # Strip markdown code blocks if present
                    if response_str.startswith("```json"):
                        response_str = response_str.strip("```json\n").strip("```")

                    try:
                        response_json = json.loads(response_str)

                        # Handle new format: {"text": "transcript"}
                        if isinstance(response_json, dict) and 'text' in response_json:
                            transcripts[f"recording_{i}"] = response_json['text']
                        # Handle old format: [{"speaker": "...", "text": "..."}, ...]
                        elif isinstance(response_json, list):
                            full_transcript = " ".join([turn['text'] for turn in response_json])
                            transcripts[f"recording_{i}"] = full_transcript
                        else:
                            logger.warning(f"Unexpected JSON format for recording_{i}, using raw text")
                            transcripts[f"recording_{i}"] = response_str
                    except json.JSONDecodeError:
                        # If not valid JSON, use the raw text
                        logger.warning(f"Could not parse JSON for recording_{i}, using raw text")
                        transcripts[f"recording_{i}"] = response_str
                else:
                    error_msg = f"Error: {data.get('status', 'No response found')}"
                    logger.error(f"No valid response for recording_{i}: {error_msg}")
                    transcripts[f"recording_{i}"] = error_msg
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.error(f"Skipping line due to parsing error: {e}")
                continue

        logger.info(f"Successfully parsed {len(transcripts)} Gemini transcripts")
        return transcripts
