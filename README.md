# ASR Benchmarking System

A production-ready benchmarking system for evaluating and comparing Automatic Speech Recognition (ASR) models. Built to compare Google Cloud's speech-to-text services (Chirp 3 and Gemini family) on French language audio from the [Mozilla Common Voice dataset](https://commonvoice.mozilla.org/en/datasets).

## Overview

This project provides a complete pipeline to:
- Transcribe audio files using multiple ASR models in parallel
- Parse and normalize outputs from different model formats
- Calculate Word Error Rate (WER) and detailed error metrics
- Generate comprehensive analysis reports (CSV/JSON)

The system is designed for reproducibility, scalability, and ease of extension to new models or languages.

### Supported Models

- **Chirp 3**: Google Cloud Speech-to-Text v2 with speaker diarization
- **Gemini Models** (6 configurations):
  - `gemini-2.5-pro` (with/without thinking)
  - `gemini-2.5-flash` (with/without thinking)
  - `gemini-2.5-flash-lite` (with/without thinking)

## Architecture

### Pipeline Stages

1. **Transcription Phase** (Parallel)
   - Submits batch transcription jobs to all models concurrently
   - Chirp uses Speech-to-Text v2 batch API
   - Gemini uses batch prediction API with structured JSON output

2. **Parsing Phase**
   - Extracts ground truth from TSV files
   - Parses Chirp JSON responses
   - Parses Gemini JSONL batch results

3. **Analysis Phase**
   - Calculates WER using jiwer library
   - Applies text normalization (lowercase, punctuation removal, etc.)
   - Outputs per-file, average, and median WER

### Project Structure

```
transcription_benchmark/
    transcription/
        base_transcriber.py    # Abstract transcriber interface
        chirp.py               # Chirp 3 batch transcription
        gemini.py              # Gemini batch transcription with thinking config
    parsing/
        base_parser.py         # Abstract parser interface
        groundtruth.py         # Parses TSV ground truth files
        chirp.py               # Parses Chirp JSON outputs
        gemini.py              # Parses Gemini JSONL batch results
    analysis/
        wer.py                 # WER calculation with jiwer
    utils/
        config.py              # Configuration loading
        constants.py           # Project constants
        gcs.py                 # GCS utilities
        logging.py             # Logging setup
        retry.py               # Retry logic
    main.py                    # Main orchestration script
```

## Prerequisites

- Python 3.12+
- Google Cloud Project with:
  - Speech-to-Text API enabled
  - Vertex AI API enabled
  - Storage API enabled
- GCS bucket with audio files and ground truth transcripts
- Service account with appropriate permissions

## Setup

### 1. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install .
```

### 2. Configure Application

Copy the example configuration and edit it:

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` with your settings:

```yaml
# Google Cloud credentials
google_cloud:
  project_id: "your-google-cloud-project-id"
  gemini_api_key: "your-gemini-api-key-here"

gcs:
  bucket: "your-gcs-bucket-name"
  audio_folder: "transcription-dataset/MP3/"
  groundtruth_folder: "transcription-dataset/TXT/"
  output_base: "transcription_results/"
```

**Note**: `config.yaml` is gitignored and should never be committed. Always use `config.yaml.example` as a template.

### 3. Prepare Data

#### Using Mozilla Common Voice Dataset

This project uses audio samples from the [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) French dataset.

**Sample Selection Process:**
- Source: Common Voice French - latest delta segment
- Filtered: Only **validated** recordings (out of ~400 validated samples)
- Sample size: **15 audio clips** selected for benchmarking
- Quality: Human-verified transcriptions for accurate evaluation

**To use your own dataset:**

1. **Download Common Voice:**
   - Visit [Common Voice Datasets](https://commonvoice.mozilla.org/en/datasets)
   - Select **French (Français)** language
   - Download and extract

2. **Organize data:**
   ```
   dataset/
   ├── MP3/              # Audio files
   │   ├── common_voice_fr_*.mp3
   └── TXT/              # Ground truth transcripts
       ├── common_voice_fr_*.txt
   ```

   Each `.txt` file contains one sentence:
   ```
   Il devient le troisième établissement d'enseignement supérieur de Victoria.
   ```

3. **Upload to GCS:**
   ```bash
   uv run python scripts/upload_dataset.py
   ```

**Data Format:**
- Audio: `.mp3` or `.wav`
- Ground truth: Plain text `.txt`, one sentence per file
- Naming: `common_voice_fr_12345.mp3` ↔ `common_voice_fr_12345.txt`

## Usage

### Run Complete Benchmark

```bash
# Using uv (recommended)
uv run python -m transcription_benchmark.main

# Or with standard python
python -m transcription_benchmark.main
```

This will:
1. Load configuration from `config.yaml`
2. Transcribe all audio files using all configured models (parallel)
3. Parse all transcription outputs
4. Calculate WER with detailed metrics
5. Save results to `results/` directory

### Expected Output

```
--- Word Error Rate (WER) Results ---
                    chirp_wer  pro_thinking_wer  pro_no_thinking_wer  ...
recording_0           12.50%            10.25%               11.00%  ...
recording_1           15.30%            13.50%               14.20%  ...
...
AVERAGE              13.90%            11.88%               12.60%  ...
MEDIAN               13.50%            11.50%               12.30%  ...
```

## Configuration

All configuration is managed through `config.yaml`. See the `config.yaml.example` file for a complete list of available options.

## Project Structure

```
.
├── config.yaml.example          # Configuration template
├── scripts/
│   └── upload_dataset.py        # Upload local data to GCS
├── transcription_benchmark/
│   ├── main.py                  # Main orchestration pipeline
│   ├── analysis/
│   │   └── wer.py              # WER calculation and metrics
│   ├── parsing/
│   │   ├── chirp.py            # Chirp result parser
│   │   ├── gemini.py           # Gemini result parser
│   │   └── groundtruth.py      # Ground truth parser
│   ├── transcription/
│   │   ├── chirp.py            # Chirp transcriber
│   │   └── gemini.py           # Gemini transcriber
│   └── utils/
│       ├── config.py           # YAML config loader
│       ├── constants.py        # System constants
│       ├── gcs.py              # GCS utilities
│       ├── logging.py          # Logging setup
│       └── retry.py            # Retry with backoff
├── dataset/                     # Local dataset (gitignored)
│   ├── MP3/
│   └── TXT/
└── results/                     # Output results (gitignored)
    ├── wer_results_*.csv
    └── wer_results_*.json
```

## Dependencies

### Core Dependencies
- `google-cloud-speech>=2.33.0` - Speech-to-Text API (Chirp)
- `google-cloud-storage>=3.4.0` - GCS operations
- `google-genai>=1.39.1` - Gemini batch API
- `jiwer>=4.0.0` - WER calculation
- `pandas>=2.3.2` - Results aggregation
- `pyyaml>=6.0.0` - Configuration management
- `python-dotenv>=1.1.1` - Environment variables

### Development Dependencies (optional)
- `pytest>=8.0.0` - Testing framework
- `black>=24.0.0` - Code formatting
- `ruff>=0.3.0` - Linting
- `mypy>=1.8.0` - Type checking

## Dataset Attribution

This project uses audio samples from the [Mozilla Common Voice](https://commonvoice.mozilla.org/) dataset:

- **Dataset**: Common Voice Corpus (French)
- **License**: CC0 (Public Domain)
- **URL**: https://commonvoice.mozilla.org/en/datasets
- **Citation**:
  ```
  @article{ardila2020common,
    title={Common Voice: A Massively-Multilingual Speech Corpus},
    author={Ardila, Rosana and Branson, Megan and Davis, Kelly and others},
    journal={Proceedings of the 12th Language Resources and Evaluation Conference},
    pages={4218--4222},
    year={2020}
  }
  ```

## Contributing

Contributions are welcome! Key areas:
- Additional ASR model integrations
- Support for more languages
- Performance optimizations
- Documentation improvements

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Mozilla Common Voice community for providing the dataset
- Google Cloud for ASR APIs
- Open source contributors of jiwer, pandas, and other dependencies

## Resources

- [Common Voice Dataset](https://commonvoice.mozilla.org/en/datasets)
- [Google Cloud Speech-to-Text](https://cloud.google.com/speech-to-text)
- [Google Gemini](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini)

## Troubleshooting

### Authentication Issues

Ensure you have Google Cloud credentials configured:

```bash
gcloud auth application-default login
```

### Import Errors

Make sure the project root is in your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
```

### GCS Permission Errors

Verify your service account has:
- `storage.objects.get`
- `storage.objects.create`
- `speech.operations.get`
- `aiplatform.batchPredictionJobs.create`
