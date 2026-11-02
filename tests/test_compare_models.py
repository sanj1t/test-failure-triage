import csv
import json

import pytest

from triage.compare_models import (
    EXPERIMENTS,
    build_candidate,
    run_comparison,
)
from triage.generate_data import generate_records
from triage.split_data import split_records


def test_candidate_configurations_match_experiment_definitions():
    for name, settings in EXPERIMENTS.items():
        model = build_candidate(name)
        vectorizer = model.named_steps["tfidf"]
        classifier = model.named_steps["classifier"]

        assert vectorizer.analyzer == settings["analyzer"]
        assert vectorizer.ngram_range == settings["ngram_range"]
        assert classifier.C == 1.0
        assert classifier.max_iter == 1000


def test_unknown_candidate_is_rejected():
    with pytest.raises(ValueError, match="Unknown experiment"):
        build_candidate("not_an_experiment")


def test_comparison_writes_reports_for_every_candidate(tmp_path):
    records = generate_records(samples_per_family=2)
    partitions = split_records(records)

    results = run_comparison(
        partitions["train"],
        partitions["development"],
        tmp_path,
    )

    assert {
        result["experiment"]
        for result in results
    } == set(EXPERIMENTS)

    for result in results:
        name = result["experiment"]
        experiment_dir = tmp_path / name

        report = json.loads(
            (experiment_dir / "development_metrics.json").read_text(
                encoding="utf-8"
            )
        )

        assert report["evaluation"]["record_count"] == 8

        with (
            experiment_dir / "development_predictions.csv"
        ).open(encoding="utf-8", newline="") as csv_file:
            assert len(list(csv.DictReader(csv_file))) == 8

        assert (
            experiment_dir / "development_errors.csv"
        ).exists()

    assert (tmp_path / "comparison.csv").exists()


def test_comparison_rejects_overlapping_partitions(tmp_path):
    records = generate_records(samples_per_family=2)
    partitions = split_records(records)

    with pytest.raises(ValueError, match="Partition overlap"):
        run_comparison(
            partitions["train"],
            partitions["train"],
            tmp_path,
        )