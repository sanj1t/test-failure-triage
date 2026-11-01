"""Train a dummy baseline and evaluate development predictions."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from triage.labels import VALID_CATEGORIES
from triage.validate_data import validate_dataset


LABELS = sorted(VALID_CATEGORIES)


def load_records(path: Path) -> list[dict[str, str]]:
    """Load a validated dataset with stripped field values."""

    validation = validate_dataset(path)

    if not validation["is_valid"]:
        raise ValueError(
            f"Dataset failed validation: {path}. "
            "Run triage.validate_data for details."
        )

    with path.open(
        encoding="utf-8",
        newline="",
    ) as csv_file:
        return [
            {
                key: value.strip()
                for key, value in record.items()
            }
            for record in csv.DictReader(csv_file, strict=True)
        ]


def ensure_partition_isolation(
    train_records: list[dict],
    development_records: list[dict],
) -> None:
    """Reject overlapping records or scenario families."""

    for field in ("incident_id", "scenario_family"):
        train_values = {
            record[field]
            for record in train_records
        }

        development_values = {
            record[field]
            for record in development_records
        }

        overlap = train_values & development_values

        if overlap:
            raise ValueError(
                f"Partition overlap in '{field}': "
                f"{sorted(overlap)}"
            )


def fit_baseline(train_records: list[dict]) -> DummyClassifier:
    """Fit a most-frequent classifier using training labels only."""

    if not train_records:
        raise ValueError("Training records must not be empty.")

    targets = [
        record["failure_category"]
        for record in train_records
    ]

    # DummyClassifier ignores these placeholder feature values.
    features = [[0] for _ in train_records]

    model = DummyClassifier(strategy="most_frequent")
    model.fit(features, targets)

    return model


def evaluate_baseline(
    model: DummyClassifier,
    records: list[dict],
) -> tuple[dict, list[str]]:
    """Evaluate a fitted baseline without refitting it."""

    if not records:
        raise ValueError("Evaluation records must not be empty.")

    targets = [
        record["failure_category"]
        for record in records
    ]

    features = [[0] for _ in records]
    predictions = model.predict(features).tolist()

    metrics = {
        "record_count": len(records),
        "accuracy": float(
            accuracy_score(targets, predictions)
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "labels": LABELS,
        "classification_report": classification_report(
            targets,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            targets,
            predictions,
            labels=LABELS,
        ).tolist(),
        "prediction_counts": dict(
            sorted(Counter(predictions).items())
        ),
    }

    return metrics, predictions


def write_predictions(
    records: list[dict],
    predictions: list[str],
    path: Path,
) -> None:
    """Save predictions alongside their reference labels."""

    if len(records) != len(predictions):
        raise ValueError(
            "Record and prediction counts must match."
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "incident_id",
                "scenario_family",
                "true_category",
                "predicted_category",
                "is_correct",
            ],
        )

        writer.writeheader()

        for record, prediction in zip(
            records,
            predictions,
            strict=True,
        ):
            writer.writerow(
                {
                    "incident_id": record["incident_id"],
                    "scenario_family": record["scenario_family"],
                    "true_category": record["failure_category"],
                    "predicted_category": prediction,
                    "is_correct": (
                        record["failure_category"] == prediction
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a dummy classifier on training data "
            "and evaluate development data."
        )
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/train.csv"),
    )

    parser.add_argument(
        "--development",
        type=Path,
        default=Path("data/processed/development.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/baseline"),
    )

    args = parser.parse_args()

    metrics_path = args.output_dir / "development_metrics.json"
    predictions_path = (
        args.output_dir / "development_predictions.csv"
    )

    input_paths = {
        args.train.resolve(),
        args.development.resolve(),
    }

    if any(
        path.resolve() in input_paths
        for path in (metrics_path, predictions_path)
    ):
        parser.error("An output path would overwrite an input file.")

    try:
        train_records = load_records(args.train)
        development_records = load_records(args.development)

        ensure_partition_isolation(
            train_records,
            development_records,
        )

        model = fit_baseline(train_records)

        metrics, predictions = evaluate_baseline(
            model,
            development_records,
        )

        report = {
            "model": "DummyClassifier",
            "strategy": "most_frequent",
            "sklearn_version": sklearn.__version__,
            "train_path": str(args.train),
            "development_path": str(args.development),
            "training_record_count": len(train_records),
            "training_category_counts": dict(
                sorted(
                    Counter(
                        record["failure_category"]
                        for record in train_records
                    ).items()
                )
            ),
            "evaluation": metrics,
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)

        metrics_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        write_predictions(
            development_records,
            predictions,
            predictions_path,
        )

    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.exit(
            status=2,
            message=f"Baseline run failed: {error}\n",
        )

    print(f"Training records: {len(train_records)}")
    print(f"Development records: {len(development_records)}")
    print("Strategy: most_frequent")
    print(f"Development accuracy: {metrics['accuracy']:.4f}")
    print(f"Development macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()