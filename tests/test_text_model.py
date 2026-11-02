import csv

import numpy as np
import pytest

from triage.generate_data import generate_records
from triage.split_data import split_records
from triage.text_model import (
    build_pipeline,
    collect_errors,
    evaluate_text_model,
    fit_text_model,
    write_errors,
)


def make_partitions():
    records = generate_records(samples_per_family=2)
    return split_records(records)


def test_pipeline_has_expected_steps():
    model = build_pipeline()

    assert list(model.named_steps) == [
        "tfidf",
        "classifier",
    ]

    assert model.named_steps["tfidf"].ngram_range == (1, 2)


def test_fitted_model_predicts_one_label_per_record():
    partitions = make_partitions()
    model = fit_text_model(partitions["train"])

    metrics, predictions = evaluate_text_model(
        model,
        partitions["development"],
    )

    assert len(predictions) == len(partitions["development"])
    assert metrics["record_count"] == len(predictions)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_development_text_does_not_change_vocabulary():
    partitions = make_partitions()
    model = fit_text_model(partitions["train"])

    vectorizer = model.named_steps["tfidf"]
    vocabulary_before = vectorizer.vocabulary_.copy()
    idf_before = vectorizer.idf_.copy()

    development = [
        {
            **partitions["development"][0],
            "error_message": "uniquedevelopmenttoken",
        }
    ]

    evaluate_text_model(model, development)

    assert "uniquedevelopmenttoken" not in vectorizer.vocabulary_
    assert vectorizer.vocabulary_ == vocabulary_before
    np.testing.assert_array_equal(
        vectorizer.idf_,
        idf_before,
    )


def test_evaluation_does_not_refit_classifier():
    partitions = make_partitions()
    model = fit_text_model(partitions["train"])

    classifier = model.named_steps["classifier"]
    coefficients_before = classifier.coef_.copy()
    intercept_before = classifier.intercept_.copy()

    evaluate_text_model(
        model,
        partitions["development"],
    )

    np.testing.assert_array_equal(
        classifier.coef_,
        coefficients_before,
    )
    np.testing.assert_array_equal(
        classifier.intercept_,
        intercept_before,
    )


def test_confusion_matrix_counts_every_record():
    partitions = make_partitions()
    model = fit_text_model(partitions["train"])

    metrics, _ = evaluate_text_model(
        model,
        partitions["development"],
    )

    matrix = metrics["confusion_matrix"]

    assert len(matrix) == 4
    assert all(len(row) == 4 for row in matrix)
    assert sum(sum(row) for row in matrix) == 8


def test_collect_errors_keeps_only_incorrect_predictions():
    records = [
        {
            "incident_id": "INC-001",
            "scenario_family": "family_a",
            "error_message": "Example one",
            "failure_category": "timeout",
        },
        {
            "incident_id": "INC-002",
            "scenario_family": "family_b",
            "error_message": "Example two",
            "failure_category": "communication",
        },
    ]

    errors = collect_errors(
        records,
        ["timeout", "configuration"],
    )

    assert len(errors) == 1
    assert errors[0]["incident_id"] == "INC-002"
    assert errors[0]["true_category"] == "communication"
    assert errors[0]["predicted_category"] == "configuration"


def test_collect_errors_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="counts must match"):
        collect_errors([], ["timeout"])


def test_empty_error_report_still_has_header(tmp_path):
    path = tmp_path / "errors.csv"

    write_errors([], path)

    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        assert reader.fieldnames == [
            "incident_id",
            "scenario_family",
            "error_message",
            "true_category",
            "predicted_category",
        ]

        assert list(reader) == []


def test_empty_training_data_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        fit_text_model([])


def test_empty_evaluation_data_is_rejected():
    partitions = make_partitions()
    model = fit_text_model(partitions["train"])

    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_text_model(model, [])


def test_training_requires_all_categories():
    records = [
        {
            "error_message": "Operation exceeded its deadline",
            "failure_category": "timeout",
        },
        {
            "error_message": "Received incomplete payload",
            "failure_category": "communication",
        },
    ]

    with pytest.raises(ValueError, match="all expected categories"):
        fit_text_model(records)