"""Chirp transcriber implementation using Google Cloud Speech-to-Text v2."""

import os
import logging
from typing import Optional
from google.cloud import speech_v2
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPIError
from transcription_benchmark.transcription.base_transcriber import BaseTranscriber
from transcription_benchmark.utils.gcs import list_gcs_files
from transcription_benchmark.utils.retry import retry_with_backoff
from transcription_benchmark.utils.constants import (
    DEFAULT_REGION,
    MIN_SPEAKER_COUNT,
    MAX_SPEAKER_COUNT,
)

logger = logging.getLogger(__name__)


class ChirpTranscriber(BaseTranscriber):
    """Transcriber for Google's Chirp model using Speech-to-Text v2 API."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        region: str = DEFAULT_REGION,
        min_speaker_count: int = MIN_SPEAKER_COUNT,
        max_speaker_count: int = MAX_SPEAKER_COUNT,
    ):
        """
        Initialize Chirp transcriber.

        Args:
            project_id: Google Cloud project ID (defaults to GOOGLE_CLOUD_PROJECT env var)
            region: Multi-regional endpoint for Speech-to-Text v2
            min_speaker_count: Minimum number of speakers for diarization
            max_speaker_count: Maximum number of speakers for diarization

        Raises:
            ValueError: If project_id is not provided and not in environment
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT must be set in environment or passed as argument")

        self.region = region
        self.min_speaker_count = min_speaker_count
        self.max_speaker_count = max_speaker_count
        self.client = speech_v2.SpeechClient(
            client_options=ClientOptions(api_endpoint=f"{self.region}-speech.googleapis.com")
        )
        logger.info(f"Initialized ChirpTranscriber for project {self.project_id} in region {self.region}")

    @retry_with_backoff(max_retries=3, exceptions=(GoogleAPIError,))
    def transcribe(self, gcs_audio_folder: str, gcs_output_folder: str, audio_extension: str = ".wav") -> Optional[str]:
        """
        Transcribe audio files using Chirp model.

        Args:
            gcs_audio_folder: GCS URI of folder containing audio files
            gcs_output_folder: GCS URI where transcription results should be saved
            audio_extension: Audio file extension to filter (default: .wav)

        Returns:
            GCS path to folder containing transcription results, or None if no files found

        Raises:
            GoogleAPIError: If API call fails after retries
            ValueError: If no audio files are found
        """
        logger.info(f"Starting Chirp transcription for folder: {gcs_audio_folder}")

        files_to_transcribe = list_gcs_files(gcs_audio_folder, file_extension=audio_extension)
        if not files_to_transcribe:
            logger.warning(f"No audio files with extension {audio_extension} found in {gcs_audio_folder}")
            raise ValueError(f"No audio files found in {gcs_audio_folder}")

        logger.info(f"Found {len(files_to_transcribe)} audio files to transcribe")

        config = speech_v2.RecognitionConfig(
            auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
            language_codes=["auto"],
            model="chirp_3",
            features=speech_v2.RecognitionFeatures(
                diarization_config=speech_v2.SpeakerDiarizationConfig(
                    min_speaker_count=self.min_speaker_count,
                    max_speaker_count=self.max_speaker_count
                )
            ),
        )

        files_metadata = [speech_v2.BatchRecognizeFileMetadata(uri=uri) for uri in files_to_transcribe]

        request = speech_v2.BatchRecognizeRequest(
            recognizer=f"projects/{self.project_id}/locations/{self.region}/recognizers/_",
            config=config,
            files=files_metadata,
            recognition_output_config=speech_v2.RecognitionOutputConfig(
                gcs_output_config=speech_v2.GcsOutputConfig(uri=gcs_output_folder),
            ),
        )

        logger.info("Submitting Chirp batch recognition request...")
        operation = self.client.batch_recognize(request=request)
        logger.info(f"Chirp batch job submitted. Operation name: {operation.operation.name}")
        logger.info("Waiting for Chirp transcription job to complete...")
        response = operation.result()

        logger.info("Chirp transcription job completed successfully")
        return gcs_output_folder
