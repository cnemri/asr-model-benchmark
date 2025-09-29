"""Base class for transcription result parsers."""

from abc import ABC, abstractmethod


class BaseParser(ABC):
    """Abstract base class for transcription result parsers."""

    @abstractmethod
    def parse(self, input_path: str) -> dict:
        """
        Parses transcription results from a given path.

        Args:
            input_path (str): The local or GCS path to the results file/folder.

        Returns:
            dict: A dictionary mapping a numerical index to the full transcript.
        """
        pass
