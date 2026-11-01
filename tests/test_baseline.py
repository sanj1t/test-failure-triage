import csv

import pytest

from triage.baseline import (
    ensure_partition_isolation,
    evaluate_baseline,
    fit_baseline,
    load_records,
    write_predictions,
)
from triage.generate_data import generate_records, write_records
from triage.split_data import split_records


def make_partitions():
    records = generate_records(samples_per_family=2)
    return split_records(records)


def test_baseline_has_expected_balanced_data_scores():
    partitions = make_partitions()
    model = fit_baseline(partitions["train"])

    metrics, predictions = evaluate_baseline(
        model,
        partitions["development"],
    )

    assert metrics["accuracy"] == pytest.approx(0.25)
    assert metrics["macro_f1"] == pytest.approx(0.10)
    assert len(set(predictions)) == 1


def test_training_majority_controls_predictions():
    train_records = [
        {"failure_category": "timeout"},
        {"failure_category": "timeout"},
        {"failure_category": "communication"},
    ]

    evaluation_records = [
        {"failure_category": "communication"},
        {"failure_category": "configuration"},
    ]

    model = fit_baseline(train_records)

    _, predictions = evaluate_baseline(
        model,
        evaluation_records,
    )

    assert predictions == ["timeout", "timeout"]


def test_evaluation_does_not_refit_model():
    model = fit_baseline(
        [
            {"failure_category": "timeout"},
            {"failure_category": "timeout"},
            {"failure_category": "communication"},
        ]
    )

    classes_before = model.classes_.copy()
    priors_before = model.class_prior_.copy()

    evaluate_baseline(
        model,
        [{"failure_category": "communication"}],
    )

    assert (model.classes_ == classes_before).all()
    assert (model.class_prior_ == priors_before).all()


def test_confusion_matrix_counts_all_evaluation_records():
    partitions = make_partitions()
    model = fit_baseline(partitions["train"])

    metrics, _ = evaluate_baseline(
        model,
        partitions["development"],
    )

    matrix = metrics["confusion_matrix"]

    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)
    assert sum(sum(row) for row in matrix) == 8


def test_overlapping_family_is_rejected():
    train = [
        {
            "incident_id": "INC-001",
            "scenario_family": "shared_family",
        }
    ]

    development = [
        {
            "incident_id": "INC-002",
            "scenario_family": "shared_family",
        }
    ]

    with pytest.raises(ValueError, match="scenario_family"):
        ensure_partition_isolation(train, development)


def test_overlapping_incident_id_is_rejected():
    train = [
        {
            "incident_id": "INC-001",
            "scenario_family": "family_a",
        }
    ]

    development = [
        {
            "incident_id": "INC-001",
            "scenario_family": "family_b",
        }
    ]

    with pytest.raises(ValueError, match="incident_id"):
        ensure_partition_isolation(train, development)


def test_loader_rejects_invalid_dataset(tmp_path):
    records = generate_records(samples_per_family=1)
    records[0]["duration_ms"] = -1
    path = tmp_path / "invalid.csv"

    write_records(records, path)

    with pytest.raises(ValueError, match="failed validation"):
        load_records(path)


def test_predictions_are_saved_with_reference_labels(tmp_path):
    partitions = make_partitions()
    records = partitions["development"]
    model = fit_baseline(partitions["train"])
    _, predictions = evaluate_baseline(model, records)

    path = tmp_path / "predictions.csv"
    write_predictions(records, predictions, path)

    with path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == len(records)
    assert rows[0]["true_category"] == records[0]["failure_category"]
    assert rows[0]["predicted_category"] == predictions[0]


def test_empty_training_data_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        fit_baseline([])


def test_empty_evaluation_data_is_rejected():
    partitions = make_partitions()
    model = fit_baseline(partitions["train"])

    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_baseline(model, [])