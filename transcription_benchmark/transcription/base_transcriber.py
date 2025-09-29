"""Base class for transcription services."""

from abc import ABC, abstractmethod


class BaseTranscriber(ABC):
    """Abstract base class for transcription services."""

    @abstractmethod
    def transcribe(self, gcs_audio_folder: str, gcs_output_folder: str) -> str:
        """
        Transcribes all audio files from a GCS folder.

        Args:
            gcs_audio_folder (str): The GCS URI of the folder containing audio files.
            gcs_output_folder (str): The GCS URI where transcription results should be saved.

        Returns:
            str: The GCS path to the folder containing the transcription results.
        """
        pass
