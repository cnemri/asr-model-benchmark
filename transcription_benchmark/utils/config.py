"""Configuration management for the transcription benchmark system."""

import os
import yaml
from typing import Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GoogleCloudConfig:
    """Google Cloud credentials configuration."""
    project_id: str
    gemini_api_key: str


@dataclass
class GCSConfig:
    """Google Cloud Storage configuration."""
    bucket: str
    audio_folder: str
    groundtruth_folder: str
    output_base: str

    @property
    def audio_uri(self) -> str:
        return f"gs://{self.bucket}/{self.audio_folder}"

    @property
    def groundtruth_uri(self) -> str:
        return f"gs://{self.bucket}/{self.groundtruth_folder}"

    @property
    def output_base_uri(self) -> str:
        return f"gs://{self.bucket}/{self.output_base}"


@dataclass
class ChirpConfig:
    """Chirp model configuration."""
    region: str
    model: str
    min_speaker_count: int
    max_speaker_count: int
    output_folder: str


@dataclass
class GeminiModelConfig:
    """Individual Gemini model configuration."""
    model: str
    thinking: bool


@dataclass
class GeminiConfig:
    """Gemini models configuration."""
    location: str
    output_folder_prefix: str
    models: dict[str, GeminiModelConfig]


@dataclass
class AudioConfig:
    """Audio format configuration."""
    format: str
    mime_type: str


@dataclass
class RetryConfig:
    """Retry logic configuration."""
    max_retries: int
    initial_delay: float
    backoff_factor: float


@dataclass
class BatchJobConfig:
    """Batch job polling configuration."""
    poll_interval: int


@dataclass
class OutputConfig:
    """Output and results configuration."""
    results_dir: str
    save_csv: bool
    save_json: bool


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str
    log_file: Optional[str]


@dataclass
class Config:
    """Main configuration container."""
    google_cloud: GoogleCloudConfig
    gcs: GCSConfig
    chirp: ChirpConfig
    gemini: GeminiConfig
    audio: AudioConfig
    retry: RetryConfig
    batch_job: BatchJobConfig
    output: OutputConfig
    logging: LoggingConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Parsed configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)

    # Parse Google Cloud credentials
    google_cloud = GoogleCloudConfig(**data['google_cloud'])

    # Parse GCS config
    gcs = GCSConfig(**data['gcs'])

    # Parse Chirp config
    chirp = ChirpConfig(**data['chirp'])

    # Parse Gemini config
    gemini_models = {
        name: GeminiModelConfig(**model_data)
        for name, model_data in data['gemini']['models'].items()
    }
    gemini = GeminiConfig(
        location=data['gemini']['location'],
        output_folder_prefix=data['gemini']['output_folder_prefix'],
        models=gemini_models
    )

    # Parse other configs
    audio = AudioConfig(**data['audio'])
    retry = RetryConfig(**data['retry'])
    batch_job = BatchJobConfig(**data['batch_job'])
    output = OutputConfig(**data['output'])
    logging_config = LoggingConfig(**data['logging'])

    return Config(
        google_cloud=google_cloud,
        gcs=gcs,
        chirp=chirp,
        gemini=gemini,
        audio=audio,
        retry=retry,
        batch_job=batch_job,
        output=output,
        logging=logging_config
    )