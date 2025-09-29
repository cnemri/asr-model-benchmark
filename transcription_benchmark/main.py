"""Main orchestration script for transcription benchmarking pipeline."""

import os
import sys
import logging
import concurrent.futures
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from transcription_benchmark.utils.logging import setup_logger
from transcription_benchmark.utils.config import load_config
from transcription_benchmark.transcription.chirp import ChirpTranscriber
from transcription_benchmark.transcription.gemini import GeminiTranscriber
from transcription_benchmark.parsing.groundtruth import GroundTruthParser
from transcription_benchmark.parsing.chirp import ChirpParser
from transcription_benchmark.parsing.gemini import GeminiParser
from transcription_benchmark.analysis.wer import calculate_wer, save_results

logger = logging.getLogger(__name__)


def main():
    """Main function to run the transcription and evaluation pipeline."""
    # Load configuration
    try:
        config = load_config("config.yaml")
    except FileNotFoundError:
        print("ERROR: config.yaml not found. Please create it from config.yaml.example")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}")
        sys.exit(1)

    # Set environment variables from config (for backward compatibility)
    os.environ['GOOGLE_CLOUD_PROJECT'] = config.google_cloud.project_id
    os.environ['GEMINI_API_KEY'] = config.google_cloud.gemini_api_key

    # Setup logging
    log_level = getattr(logging, config.logging.level.upper(), logging.INFO)
    setup_logger(level=log_level, log_file=config.logging.log_file)

    logger.info("="*60)
    logger.info("Starting Transcription Benchmark Pipeline")
    logger.info("="*60)

    # --- 1. Transcription Phase (Parallel) ---
    logger.info("\n--- Starting Transcription Phase ---")

    chirp_output_path = None
    gemini_output_paths = {}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit Chirp job
        logger.info("Submitting Chirp transcription job...")
        chirp_transcriber = ChirpTranscriber(
            region=config.chirp.region,
            min_speaker_count=config.chirp.min_speaker_count,
            max_speaker_count=config.chirp.max_speaker_count,
        )
        chirp_output_folder = f"{config.gcs.output_base_uri}{config.chirp.output_folder}"
        chirp_future = executor.submit(
            chirp_transcriber.transcribe,
            config.gcs.audio_uri,
            chirp_output_folder,
            config.audio.format
        )

        # Submit Gemini jobs
        logger.info(f"Submitting {len(config.gemini.models)} Gemini transcription jobs...")
        gemini_futures = {}
        for name, model_config in config.gemini.models.items():
            gemini_transcriber = GeminiTranscriber(
                model=model_config.model,
                thinking=model_config.thinking,
                location=config.gemini.location,
                audio_mime_type=config.audio.mime_type,
            )
            output_folder = f"{config.gcs.output_base_uri}{config.gemini.output_folder_prefix}{name}/"
            future = executor.submit(
                gemini_transcriber.transcribe,
                config.gcs.audio_uri,
                output_folder,
                config.audio.format
            )
            gemini_futures[future] = name

        # Collect results
        logger.info("Waiting for Chirp transcription to complete...")
        try:
            chirp_output_path = chirp_future.result()
            logger.info("✓ Chirp transcription completed successfully")
        except Exception as e:
            logger.error(f"✗ Chirp transcription failed: {e}")

        logger.info("Waiting for Gemini transcriptions to complete...")
        for future in concurrent.futures.as_completed(gemini_futures):
            name = gemini_futures[future]
            try:
                result_path = future.result()
                gemini_output_paths[name] = result_path
                logger.info(f"✓ Gemini transcription '{name}' completed successfully")
            except Exception as exc:
                logger.error(f"✗ Gemini transcription '{name}' failed: {exc}")

        logger.info("All transcription jobs completed")

    # --- 2. Parsing Phase ---
    logger.info("\n--- Starting Parsing Phase ---")

    # Parse ground truth
    logger.info("Parsing ground truth data...")
    gt_parser = GroundTruthParser()
    try:
        ground_truth_data = gt_parser.parse(config.gcs.groundtruth_uri)
        logger.info(f"✓ Parsed {len(ground_truth_data)} ground truth files")
    except Exception as e:
        logger.error(f"✗ Failed to parse ground truth: {e}")
        sys.exit(1)

    # Parse Chirp results
    all_hypotheses = {}
    if chirp_output_path:
        logger.info("Parsing Chirp results...")
        try:
            chirp_parser = ChirpParser()
            chirp_transcripts = chirp_parser.parse(chirp_output_path)
            all_hypotheses["chirp"] = chirp_transcripts
            logger.info(f"✓ Parsed {len(chirp_transcripts)} Chirp transcripts")
        except Exception as e:
            logger.error(f"✗ Failed to parse Chirp results: {e}")

    # Parse Gemini results
    for name, path in gemini_output_paths.items():
        logger.info(f"Parsing Gemini results for '{name}'...")
        try:
            parser = GeminiParser()
            gemini_transcripts = parser.parse(path)
            all_hypotheses[name] = gemini_transcripts
            logger.info(f"✓ Parsed {len(gemini_transcripts)} Gemini '{name}' transcripts")
        except Exception as e:
            logger.error(f"✗ Failed to parse Gemini '{name}' results: {e}")

    if not all_hypotheses:
        logger.error("No transcription results were successfully parsed. Exiting.")
        sys.exit(1)

    # --- 3. Analysis Phase ---
    logger.info("\n--- Starting Analysis Phase ---")

    try:
        wer_results = calculate_wer(
            ground_truth_data,
            all_hypotheses,
            include_detailed_metrics=True
        )

        # Display results
        logger.info("\n" + "="*60)
        logger.info("WORD ERROR RATE (WER) RESULTS")
        logger.info("="*60)
        # Only show WER columns for cleaner output
        wer_cols = [col for col in wer_results.columns if col.endswith('_wer')]
        print("\n" + (wer_results[wer_cols] * 100).to_string(float_format="%.2f%%"))

        # Save results
        if config.output.save_csv or config.output.save_json:
            logger.info(f"\nSaving results to {config.output.results_dir}/...")
            saved_files = save_results(
                wer_results,
                output_dir=config.output.results_dir,
                save_csv=config.output.save_csv,
                save_json=config.output.save_json
            )
            for format_type, file_path in saved_files.items():
                logger.info(f"✓ Saved {format_type.upper()}: {file_path}")

        logger.info("\n" + "="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60 + "\n")

    except Exception as e:
        logger.error(f"Analysis phase failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()