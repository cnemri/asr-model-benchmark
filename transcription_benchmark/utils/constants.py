"""Constants used throughout the transcription benchmark system."""

# API Configuration
DEFAULT_REGION = "us"
DEFAULT_LOCATION = "us-central1"

# Retry Configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0
BACKOFF_FACTOR = 2.0

# Speaker Diarization
MIN_SPEAKER_COUNT = 1
MAX_SPEAKER_COUNT = 6

# Batch Job Polling
BATCH_JOB_POLL_INTERVAL = 30  # seconds

# Thinking Budget Configuration
UNLIMITED_THINKING_BUDGET = -1
PRO_MODEL_LIMITED_THINKING_BUDGET = 128
FLASH_MODEL_LIMITED_THINKING_BUDGET = 0

# Audio Configuration
SUPPORTED_AUDIO_FORMATS = [".wav", ".mp3", ".flac"]
DEFAULT_AUDIO_FORMAT = ".wav"
DEFAULT_AUDIO_MIME_TYPE = "audio/wav"

# File Extensions
JSON_EXTENSION = ".json"
TXT_EXTENSION = ".txt"
JSONL_EXTENSION = ".jsonl"

# Result Output
WER_PERCENTAGE_FORMAT = "%.2f%%"
RESULTS_OUTPUT_DIR = "results"