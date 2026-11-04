"""Train and save the selected text-classification pipeline."""

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib

from triage.baseline import load_records
from triage.text_model import fit_text_model


MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"


def train_and_save(
    train_path: Path,
    output_dir: Path,
) -> dict:
    """Fit the selected pipeline and save a new artifact directory."""

    # Read the exact bytes represented by the training-file hash.
    training_bytes = train_path.read_bytes()

    records = load_records(train_path)

    # Detect a file change between hashing and loading.
    if train_path.read_bytes() != training_bytes:
        raise ValueError(
            "Training file changed while it was being read."
        )

    model = fit_text_model(records)

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    metadata = {
        "artifact_format_version": 1,
        "experiment": "word_12",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training": {
            "path": str(train_path),
            "sha256": hashlib.sha256(training_bytes).hexdigest(),
            "record_count": len(records),
            "incident_ids": sorted(
                record["incident_id"]
                for record in records
            ),
            "scenario_families": sorted({
                record["scenario_family"]
                for record in records
            }),
        },
        "input_features": ["error_message"],
        "labels": classifier.classes_.tolist(),
        "configuration": {
            "analyzer": vectorizer.analyzer,
            "ngram_range": list(vectorizer.ngram_range),
            "lowercase": vectorizer.lowercase,
            "min_df": vectorizer.min_df,
            "C": classifier.C,
            "solver": classifier.solver,
            "max_iter": classifier.max_iter,
        },
        "vocabulary_size": len(vectorizer.vocabulary_),
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": version("scikit-learn"),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "joblib": version("joblib"),
        },
        "limitations": [
            "Trained on synthetic, template-based records.",
            "Predictions are triage suggestions, not verified root causes.",
            "Real-world performance is unknown.",
        ],
    }

    # Refuse to reuse an existing directory.
    # This prevents accidental replacement of an earlier artifact.
    output_dir.mkdir(parents=True, exist_ok=False)

    joblib.dump(
        model,
        output_dir / MODEL_FILENAME,
    )

    (output_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and save the selected word_12 pipeline."
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/train.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/word_12_v1"),
    )

    args = parser.parse_args()

    try:
        metadata = train_and_save(
            args.train,
            args.output_dir,
        )
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.exit(
            status=2,
            message=f"Training failed: {error}\n",
        )

    print(f"Experiment: {metadata['experiment']}")
    print(
        f"Training records: "
        f"{metadata['training']['record_count']}"
    )
    print(f"Vocabulary size: {metadata['vocabulary_size']}")
    print(f"Model: {args.output_dir / MODEL_FILENAME}")
    print(f"Metadata: {args.output_dir / METADATA_FILENAME}")


if __name__ == "__main__":
    main()