"""Evaluate a saved model on a held-out dataset without retraining."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from triage.baseline import load_records, write_predictions
from triage.labels import VALID_CATEGORIES
from triage.predict import load_artifact
from triage.text_model import (
    collect_errors,
    evaluate_text_model,
    write_errors,
)
from triage.train import METADATA_FILENAME, MODEL_FILENAME


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hash of a file's contents."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_held_out(
    records: list[dict],
    metadata: dict,
) -> None:
    """Reject overlap with the artifact's recorded training data."""

    training = metadata.get("training")

    if not isinstance(training, dict):
        raise ValueError("Training membership metadata is missing.")

    checks = (
        ("incident_id", "incident_ids"),
        ("scenario_family", "scenario_families"),
    )

    for record_field, metadata_field in checks:
        training_values = training.get(metadata_field)

        if (
            not isinstance(training_values, list)
            or not training_values
            or not all(
                isinstance(value, str) and value.strip()
                for value in training_values
            )
        ):
            raise ValueError(
                f"Invalid training membership: {metadata_field}"
            )

        evaluation_values = {
            record[record_field]
            for record in records
        }

        overlap = evaluation_values & set(training_values)

        if overlap:
            raise ValueError(
                f"Training/evaluation overlap in "
                f"'{record_field}': {sorted(overlap)}"
            )


def evaluate_saved_model(
    artifact_dir: Path,
    test_path: Path,
    output_dir: Path,
) -> dict:
    """Evaluate a trusted saved artifact and write a new report set."""

    if output_dir.exists():
        raise FileExistsError(
            f"Evaluation directory already exists: {output_dir}. "
            "Existing evaluation results will not be overwritten."
        )

    source_paths = {
        "model": artifact_dir / MODEL_FILENAME,
        "metadata": artifact_dir / METADATA_FILENAME,
        "test_dataset": test_path,
    }

    hashes_before = {
        name: sha256_file(path)
        for name, path in source_paths.items()
    }

    model, metadata = load_artifact(artifact_dir)
    records = load_records(test_path)

    ensure_held_out(records, metadata)

    observed_categories = {
        record["failure_category"]
        for record in records
    }

    if observed_categories != VALID_CATEGORIES:
        raise ValueError(
            "This project's final evaluation requires "
            "all expected categories."
        )

    # This function only predicts and scores; it does not call fit().
    metrics, predictions = evaluate_text_model(model, records)
    errors = collect_errors(records, predictions)

    hashes_after = {
        name: sha256_file(path)
        for name, path in source_paths.items()
    }

    if hashes_after != hashes_before:
        raise ValueError(
            "An input file changed during evaluation. "
            "No evaluation report was written."
        )

    report = {
        "evaluation_kind": "held_out_test",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": metadata["experiment"],
        "artifact_created_at_utc": metadata.get("created_at_utc"),
        "sources": {
            name: {
                "path": str(path),
                "sha256": hashes_before[name],
            }
            for name, path in source_paths.items()
        },
        "training": {
            "record_count": metadata["training"]["record_count"],
            "dataset_sha256": metadata["training"]["sha256"],
        },
        "evaluation_environment": metadata["environment"],
        "checks": {
            "dataset_validation_passed": True,
            "no_training_incident_overlap": True,
            "no_training_family_overlap": True,
            "all_expected_categories_present": True,
            "input_files_unchanged_during_evaluation": True,
        },
        "evaluation_family_count": len({
            record["scenario_family"]
            for record in records
        }),
        "evaluation": metrics,
        "limitations": [
            "Evaluation uses synthetic, template-based records.",
            "Record count is not the number of independent scenarios.",
            "Training-membership checks rely on artifact metadata.",
            "Real-world performance is unknown.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=False)

    (output_dir / "test_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    write_predictions(
        records,
        predictions,
        output_dir / "test_predictions.csv",
    )

    write_errors(
        errors,
        output_dir / "test_errors.csv",
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trusted saved artifact on held-out data "
            "without retraining."
        )
    )

    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/word_12_v1"),
    )

    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/processed/test.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/final_test/word_12_v1"),
    )

    args = parser.parse_args()

    try:
        report = evaluate_saved_model(
            args.artifact_dir,
            args.test,
            args.output_dir,
        )
    except Exception as error:
        parser.exit(
            status=2,
            message=f"Evaluation failed: {error}\n",
        )

    metrics = report["evaluation"]

    print(f"Experiment: {report['experiment']}")
    print(f"Test records: {metrics['record_count']}")
    print(f"Test families: {report['evaluation_family_count']}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Test errors: {metrics['error_count']}")
    print("Training/test membership overlap: none")
    print(f"Reports: {args.output_dir}")


if __name__ == "__main__":
    main()