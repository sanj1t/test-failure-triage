"""Train and evaluate a text-only failure classifier."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from triage.baseline import (
    LABELS,
    ensure_partition_isolation,
    load_records,
    write_predictions,
)


ERROR_FIELDNAMES = [
    "incident_id",
    "scenario_family",
    "error_message",
    "true_category",
    "predicted_category",
]


def build_pipeline() -> Pipeline:
    """Create an unfitted text-classification pipeline."""

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                ),
            ),
        ]
    )


def fit_text_model(records: list[dict]) -> Pipeline:
    """Fit vectorization and classification on training data only."""

    if not records:
        raise ValueError("Training records must not be empty.")

    targets = [
        record["failure_category"]
        for record in records
    ]

    if set(targets) != set(LABELS):
        raise ValueError(
            "Training data must contain all expected categories."
        )

    texts = [
        record["error_message"]
        for record in records
    ]

    model = build_pipeline()
    model.fit(texts, targets)

    return model


def evaluate_text_model(
    model: Pipeline,
    records: list[dict],
) -> tuple[dict, list[str]]:
    """Evaluate a fitted model without changing its parameters."""

    if not records:
        raise ValueError("Evaluation records must not be empty.")

    texts = [
        record["error_message"]
        for record in records
    ]

    targets = [
        record["failure_category"]
        for record in records
    ]

    predictions = model.predict(texts).tolist()

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
        "error_count": sum(
            target != prediction
            for target, prediction in zip(
                targets,
                predictions,
                strict=True,
            )
        ),
    }

    return metrics, predictions


def collect_errors(
    records: list[dict],
    predictions: list[str],
) -> list[dict]:
    """Return incorrectly classified records for manual inspection."""

    if len(records) != len(predictions):
        raise ValueError(
            "Record and prediction counts must match."
        )

    errors = []

    for record, prediction in zip(
        records,
        predictions,
        strict=True,
    ):
        if record["failure_category"] == prediction:
            continue

        errors.append(
            {
                "incident_id": record["incident_id"],
                "scenario_family": record["scenario_family"],
                "error_message": record["error_message"],
                "true_category": record["failure_category"],
                "predicted_category": prediction,
            }
        )

    return errors


def write_errors(errors: list[dict], path: Path) -> None:
    """Write an error CSV, including a header if there are no errors."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=ERROR_FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(errors)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train TF-IDF plus logistic regression "
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
        default=Path("reports/text_logreg"),
    )

    args = parser.parse_args()

    metrics_path = args.output_dir / "development_metrics.json"
    predictions_path = (
        args.output_dir / "development_predictions.csv"
    )
    errors_path = args.output_dir / "development_errors.csv"

    input_paths = {
        args.train.resolve(),
        args.development.resolve(),
    }

    output_paths = (
        metrics_path,
        predictions_path,
        errors_path,
    )

    if any(
        path.resolve() in input_paths
        for path in output_paths
    ):
        parser.error("An output path would overwrite an input file.")

    try:
        train_records = load_records(args.train)
        development_records = load_records(args.development)

        ensure_partition_isolation(
            train_records,
            development_records,
        )

        model = fit_text_model(train_records)

        metrics, predictions = evaluate_text_model(
            model,
            development_records,
        )

        errors = collect_errors(
            development_records,
            predictions,
        )

        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]

        report = {
            "model": "TF-IDF + LogisticRegression",
            "sklearn_version": sklearn.__version__,
            "train_path": str(args.train),
            "development_path": str(args.development),
            "training_record_count": len(train_records),
            "input_features": ["error_message"],
            "configuration": {
                "ngram_range": list(vectorizer.ngram_range),
                "min_df": vectorizer.min_df,
                "lowercase": vectorizer.lowercase,
                "C": classifier.C,
                "solver": classifier.solver,
                "max_iter": classifier.max_iter,
            },
            "vocabulary_size": len(vectorizer.vocabulary_),
            "solver_iterations": classifier.n_iter_.tolist(),
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

        write_errors(errors, errors_path)

    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.exit(
            status=2,
            message=f"Text-model run failed: {error}\n",
        )

    print(f"Training records: {len(train_records)}")
    print(f"Development records: {len(development_records)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"Development accuracy: {metrics['accuracy']:.4f}")
    print(f"Development macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Development errors: {metrics['error_count']}")
    print(f"Metrics: {metrics_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Errors: {errors_path}")


if __name__ == "__main__":
    main()