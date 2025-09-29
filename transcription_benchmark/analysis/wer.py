"""Word Error Rate (WER) calculation and detailed metrics analysis."""

import os
import json
import logging
from pathlib import Path
from typing import Optional
import jiwer
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_wer(
    ground_truth: dict[str, str],
    hypotheses: dict[str, dict[str, str]],
    include_detailed_metrics: bool = True
) -> pd.DataFrame:
    """
    Calculate WER and optionally detailed metrics for transcription hypotheses.

    Args:
        ground_truth: Dictionary mapping file keys to ground truth transcripts
        hypotheses: Dictionary where keys are model names and values are
                   dictionaries mapping file keys to hypothesis transcripts
        include_detailed_metrics: If True, include insertions, deletions, substitutions

    Returns:
        DataFrame with WER and optionally detailed metrics for each file and model

    Raises:
        ValueError: If ground_truth or hypotheses are empty
    """
    if not ground_truth:
        raise ValueError("Ground truth dictionary is empty")
    if not hypotheses:
        raise ValueError("Hypotheses dictionary is empty")

    logger.info(f"Calculating WER for {len(ground_truth)} files across {len(hypotheses)} models")

    results = []
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords()
    ])

    for key, reference in ground_truth.items():
        row = {"file": key}

        for model_name, hypothesis_set in hypotheses.items():
            hypothesis = hypothesis_set.get(key)
            if hypothesis:
                if include_detailed_metrics:
                    # Use process_words to get all metrics at once
                    output = jiwer.process_words(
                        reference,
                        hypothesis,
                        reference_transform=transformation,
                        hypothesis_transform=transformation
                    )
                    row[f"{model_name}_wer"] = output.wer
                    row[f"{model_name}_insertions"] = output.insertions
                    row[f"{model_name}_deletions"] = output.deletions
                    row[f"{model_name}_substitutions"] = output.substitutions
                    row[f"{model_name}_hits"] = output.hits
                else:
                    # Just calculate WER
                    wer_value = jiwer.wer(
                        reference,
                        hypothesis,
                        reference_transform=transformation,
                        hypothesis_transform=transformation
                    )
                    row[f"{model_name}_wer"] = wer_value
            else:
                logger.warning(f"No hypothesis found for file {key} in model {model_name}")

        results.append(row)

    df = pd.DataFrame(results)

    # Calculate and append summary statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    avg_row = df[numeric_cols].mean().to_frame().T
    avg_row['file'] = 'AVERAGE'

    median_row = df[numeric_cols].median().to_frame().T
    median_row['file'] = 'MEDIAN'

    df = pd.concat([df, avg_row, median_row], ignore_index=True)

    logger.info("WER calculation completed")
    return df.set_index('file')


def save_results(
    df: pd.DataFrame,
    output_dir: str = "results",
    save_csv: bool = True,
    save_json: bool = True,
    timestamp: Optional[str] = None
) -> dict[str, str]:
    """
    Save WER results to CSV and/or JSON files.

    Args:
        df: DataFrame containing WER results
        output_dir: Directory to save results (created if doesn't exist)
        save_csv: Whether to save as CSV
        save_json: Whether to save as JSON
        timestamp: Optional timestamp string for filename (auto-generated if None)

    Returns:
        Dictionary mapping format to file path

    Raises:
        ValueError: If neither save_csv nor save_json is True
    """
    if not save_csv and not save_json:
        raise ValueError("At least one of save_csv or save_json must be True")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    saved_files = {}

    if save_csv:
        csv_path = output_path / f"wer_results_{timestamp}.csv"
        df.to_csv(csv_path)
        saved_files['csv'] = str(csv_path)
        logger.info(f"Saved WER results to CSV: {csv_path}")

    if save_json:
        json_path = output_path / f"wer_results_{timestamp}.json"
        df.to_json(json_path, orient='index', indent=2)
        saved_files['json'] = str(json_path)
        logger.info(f"Saved WER results to JSON: {json_path}")

    return saved_files
