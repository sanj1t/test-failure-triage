"""Compare a small, fixed set of text representations."""

import argparse
import csv
import json
from pathlib import Path

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from triage.baseline import (
    LABELS,
    ensure_partition_isolation,
    load_records,
    write_predictions,
)
from triage.text_model import (
    collect_errors,
    evaluate_text_model,
    write_errors,
)


EXPERIMENTS = {
    "word_12": {
        "analyzer": "word",
        "ngram_range": (1, 2),
    },
    "word_1": {
        "analyzer": "word",
        "ngram_range": (1, 1),
    },
    "char_35": {
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
    },
}


def build_candidate(name: str) -> Pipeline:
    """Build a candidate without fitting any data."""

    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}")

    settings = EXPERIMENTS[name]

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    min_df=1,
                    analyzer=settings["analyzer"],
                    ngram_range=settings["ngram_range"],
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


def run_comparison(
    train_records: list[dict],
    development_records: list[dict],
    output_dir: Path,
) -> list[dict]:
    """Fit candidates on train and evaluate development only."""

    if not train_records or not development_records:
        raise ValueError("Both partitions must contain records.")

    ensure_partition_isolation(
        train_records,
        development_records,
    )

    training_labels = [
        record["failure_category"]
        for record in train_records
    ]

    if set(training_labels) != set(LABELS):
        raise ValueError(
            "Training data must contain all expected categories."
        )

    training_texts = [
        record["error_message"]
        for record in train_records
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = []

    for name in EXPERIMENTS:
        model = build_candidate(name)
        model.fit(training_texts, training_labels)

        metrics, predictions = evaluate_text_model(
            model,
            development_records,
        )

        errors = collect_errors(
            development_records,
            predictions,
        )

        experiment_dir = output_dir / name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        settings = EXPERIMENTS[name]

        report = {
            "experiment": name,
            "sklearn_version": sklearn.__version__,
            "configuration": {
                "analyzer": settings["analyzer"],
                "ngram_range": list(settings["ngram_range"]),
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "min_df": 1,
                "lowercase": True,
            },
            "training_record_count": len(train_records),
            "vocabulary_size": len(
                model.named_steps["tfidf"].vocabulary_
            ),
            "evaluation": metrics,
        }

        (experiment_dir / "development_metrics.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        write_predictions(
            development_records,
            predictions,
            experiment_dir / "development_predictions.csv",
        )

        write_errors(
            errors,
            experiment_dir / "development_errors.csv",
        )

        comparison.append(
            {
                "experiment": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "error_count": metrics["error_count"],
                "vocabulary_size": report["vocabulary_size"],
            }
        )

    with (output_dir / "comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "experiment",
                "accuracy",
                "macro_f1",
                "error_count",
                "vocabulary_size",
            ],
        )

        writer.writeheader()
        writer.writerows(comparison)

    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare three fixed TF-IDF configurations."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/model_comparison"),
    )

    args = parser.parse_args()

    train_path = Path("data/processed/train.csv")
    development_path = Path("data/processed/development.csv")

    output_dir = args.output_dir.resolve()
    input_paths = {
        train_path.resolve(),
        development_path.resolve(),
    }

    possible_outputs = {
        output_dir / "comparison.csv",
    }

    for name in EXPERIMENTS:
        for filename in (
            "development_metrics.json",
            "development_predictions.csv",
            "development_errors.csv",
        ):
            possible_outputs.add(output_dir / name / filename)

    if any(path.resolve() in input_paths for path in possible_outputs):
        parser.error("An output path would overwrite an input file.")

    try:
        train_records = load_records(train_path)
        development_records = load_records(development_path)

        comparison = run_comparison(
            train_records,
            development_records,
            args.output_dir,
        )

    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.exit(
            status=2,
            message=f"Comparison failed: {error}\n",
        )

    print(
        f"{'Experiment':16} "
        f"{'Accuracy':>10} "
        f"{'Macro-F1':>10} "
        f"{'Errors':>8} "
        f"{'Features':>10}"
    )

    for result in comparison:
        print(
            f"{result['experiment']:16} "
            f"{result['accuracy']:10.4f} "
            f"{result['macro_f1']:10.4f} "
            f"{result['error_count']:8d} "
            f"{result['vocabulary_size']:10d}"
        )

    print(f"\nReports: {args.output_dir}")


if __name__ == "__main__":
    main()