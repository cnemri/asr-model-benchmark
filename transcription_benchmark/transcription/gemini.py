"""Gemini transcriber implementation using Google GenAI batch API."""

import os
import re
import time
import json
import logging
from typing import Optional
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from transcription_benchmark.transcription.base_transcriber import BaseTranscriber
from transcription_benchmark.utils.gcs import list_gcs_files
from transcription_benchmark.utils.retry import retry_with_backoff
from transcription_benchmark.utils.constants import (
    DEFAULT_LOCATION,
    BATCH_JOB_POLL_INTERVAL,
    UNLIMITED_THINKING_BUDGET,
    PRO_MODEL_LIMITED_THINKING_BUDGET,
    FLASH_MODEL_LIMITED_THINKING_BUDGET,
)

logger = logging.getLogger(__name__)


class GeminiTranscriber(BaseTranscriber):
    """Transcriber for Google's Gemini models using batch prediction API."""

    def __init__(
        self,
        model: str,
        thinking: bool,
        project_id: Optional[str] = None,
        location: str = DEFAULT_LOCATION,
        audio_mime_type: str = "audio/wav",
    ):
        """
        Initialize Gemini transcriber.

        Args:
            model: Gemini model name (e.g., 'gemini-2.5-pro')
            thinking: Whether to enable extended thinking mode
            project_id: Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
            location: Vertex AI location
            audio_mime_type: MIME type for audio files (e.g., 'audio/wav', 'audio/mpeg')

        Raises:
            ValueError: If project_id is not provided and not in environment
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT must be set in environment or passed as argument")

        self.location = location
        self.model = model
        self.thinking = thinking
        self.audio_mime_type = audio_mime_type

        os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'True'
        self.client = genai.Client(
            project=self.project_id,
            location=self.location,
            http_options=HttpOptions(api_version="v1")
        )
        logger.info(
            f"Initialized GeminiTranscriber: model={model}, thinking={thinking}, "
            f"project={self.project_id}, location={location}"
        )

    @retry_with_backoff(max_retries=3, exceptions=(GoogleAPIError, Exception))
    def _create_jsonl_for_batch(self, file_uris: list[str], bucket_name: str) -> tuple[str, str]:
        """
        Create JSONL file with transcription requests and upload to GCS.

        Args:
            file_uris: List of GCS URIs for audio files
            bucket_name: GCS bucket name for storing the request file

        Returns:
            Tuple of (GCS URI to JSONL file, blob name)

        Raises:
            GoogleAPIError: If upload to GCS fails after retries
        """
        logger.info(f"Creating batch request JSONL for {len(file_uris)} files")
        requests = []

        for uri in file_uris:
            # Determine thinking budget based on configuration
            if self.thinking:
                thinking_budget = UNLIMITED_THINKING_BUDGET
            else:
                thinking_budget = (
                    PRO_MODEL_LIMITED_THINKING_BUDGET if "pro" in self.model
                    else FLASH_MODEL_LIMITED_THINKING_BUDGET
                )

            request = {
                "contents": [
                    {"role": "user", "parts": [
                        {"text": "Transcribe this audio file. Return the exact spoken text as a JSON object with a single 'text' field. The audio contains a single speaker reading one sentence."},
                        {"fileData": {"fileUri": uri, "mimeType": self.audio_mime_type}}
                    ]}
                ],
                "generationConfig": {
                    "thinkingConfig": {"thinkingBudget": thinking_budget},
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "text": {
                                "type": "STRING",
                                "description": "The complete transcribed text from the audio."
                            }
                        },
                        "required": ["text"]
                    }
                }
            }
            requests.append(json.dumps({"request": request}))

        jsonl_content = "\n".join(requests)

        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        thinking_str = "thinking" if self.thinking else "no_thinking"
        blob_name = f"batch_requests/request-{self.model}-{thinking_str}-{int(time.time())}.jsonl"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(jsonl_content)

        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        logger.info(f"Uploaded batch request file to {gcs_uri}")
        return gcs_uri, blob_name

    def transcribe(self, gcs_audio_folder: str, gcs_output_folder: str, audio_extension: str = ".wav") -> Optional[str]:
        """
        Transcribe audio files using Gemini model.

        Args:
            gcs_audio_folder: GCS URI of folder containing audio files
            gcs_output_folder: GCS URI where transcription results should be saved
            audio_extension: Audio file extension to filter (default: .wav)

        Returns:
            GCS path to folder containing transcription results, or None if job fails

        Raises:
            ValueError: If no audio files are found or bucket name cannot be extracted
            RuntimeError: If batch job fails
        """
        logger.info(
            f"Starting Gemini transcription: model={self.model}, thinking={self.thinking}, "
            f"folder={gcs_audio_folder}"
        )

        files_to_transcribe = list_gcs_files(gcs_audio_folder, file_extension=audio_extension)
        if not files_to_transcribe:
            logger.warning(f"No audio files with extension {audio_extension} found in {gcs_audio_folder}")
            raise ValueError(f"No audio files found in {gcs_audio_folder}")

        logger.info(f"Found {len(files_to_transcribe)} audio files to transcribe")

        match = re.match(r"gs://([^/]+)/", gcs_audio_folder)
        if not match:
            raise ValueError(f"Invalid GCS URI format: {gcs_audio_folder}")
        bucket_name = match.group(1)

        jsonl_uri, jsonl_blob = self._create_jsonl_for_batch(files_to_transcribe, bucket_name)

        logger.info(f"Creating batch job for model {self.model}")
        job = self.client.batches.create(
            model=self.model,
            src=jsonl_uri,
            config=CreateBatchJobConfig(dest=gcs_output_folder),
        )

        logger.info(f"Batch job created: {job.name}")
        logger.info("Polling for job completion...")
        completed_states = {JobState.JOB_STATE_SUCCEEDED, JobState.JOB_STATE_FAILED, JobState.JOB_STATE_CANCELLED}

        while job.state not in completed_states:
            time.sleep(BATCH_JOB_POLL_INTERVAL)
            job = self.client.batches.get(name=job.name)
            logger.info(f"Job state: {job.state}")

        # Clean up the temporary request file
        self._cleanup_request_file(bucket_name, jsonl_blob, jsonl_uri)

        if job.state != JobState.JOB_STATE_SUCCEEDED:
            error_msg = f"Gemini batch job failed with state: {job.state}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info("Gemini transcription job completed successfully")
        return gcs_output_folder

    def _cleanup_request_file(self, bucket_name: str, blob_name: str, gcs_uri: str) -> None:
        """
        Clean up temporary request file from GCS.

        Args:
            bucket_name: GCS bucket name
            blob_name: Blob name within bucket
            gcs_uri: Full GCS URI for logging
        """
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            bucket.blob(blob_name).delete()
            logger.info(f"Deleted temporary request file: {gcs_uri}")
        except Exception as e:
            logger.warning(
                f"Could not delete request file {gcs_uri}, it may have been deleted already. Error: {e}"
            )
