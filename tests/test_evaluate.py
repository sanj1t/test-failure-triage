import json

import pytest

from triage.evaluate import (
    ensure_held_out,
    evaluate_saved_model,
    sha256_file,
)
from triage.generate_data import generate_records, write_records
from triage.split_data import split_records
from triage.train import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    train_and_save,
)


@pytest.fixture
def evaluation_setup(tmp_path):
    records = generate_records(samples_per_family=2)
    partitions = split_records(records)

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    artifact_dir = tmp_path / "artifact"
    output_dir = tmp_path / "evaluation"

    write_records(partitions["train"], train_path)
    write_records(partitions["test"], test_path)

    metadata = train_and_save(train_path, artifact_dir)

    return {
        "partitions": partitions,
        "train_path": train_path,
        "test_path": test_path,
        "artifact_dir": artifact_dir,
        "output_dir": output_dir,
        "metadata": metadata,
    }


def test_final_evaluation_writes_reports(evaluation_setup):
    setup = evaluation_setup

    report = evaluate_saved_model(
        setup["artifact_dir"],
        setup["test_path"],
        setup["output_dir"],
    )

    assert report["evaluation_kind"] == "held_out_test"
    assert report["evaluation"]["record_count"] == 8
    assert report["evaluation_family_count"] == 4
    assert all(report["checks"].values())

    for filename in (
        "test_metrics.json",
        "test_predictions.csv",
        "test_errors.csv",
    ):
        assert (setup["output_dir"] / filename).exists()

    saved_report = json.loads(
        (setup["output_dir"] / "test_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert saved_report == report


def test_evaluation_preserves_artifact_files(evaluation_setup):
    setup = evaluation_setup

    paths = [
        setup["artifact_dir"] / MODEL_FILENAME,
        setup["artifact_dir"] / METADATA_FILENAME,
    ]

    before = [path.read_bytes() for path in paths]

    evaluate_saved_model(
        setup["artifact_dir"],
        setup["test_path"],
        setup["output_dir"],
    )

    after = [path.read_bytes() for path in paths]

    assert after == before


def test_training_incident_overlap_is_rejected(evaluation_setup):
    setup = evaluation_setup
    records = [
        dict(record)
        for record in setup["partitions"]["test"]
    ]

    records[0]["incident_id"] = (
        setup["metadata"]["training"]["incident_ids"][0]
    )

    with pytest.raises(ValueError, match="incident_id"):
        ensure_held_out(records, setup["metadata"])


def test_training_family_overlap_is_rejected(evaluation_setup):
    setup = evaluation_setup
    records = [
        dict(record)
        for record in setup["partitions"]["test"]
    ]

    records[0]["scenario_family"] = (
        setup["metadata"]["training"]["scenario_families"][0]
    )

    with pytest.raises(ValueError, match="scenario_family"):
        ensure_held_out(records, setup["metadata"])


def test_missing_membership_metadata_is_rejected(evaluation_setup):
    setup = evaluation_setup

    with pytest.raises(ValueError, match="membership"):
        ensure_held_out(
            setup["partitions"]["test"],
            {"training": {}},
        )


def test_existing_evaluation_is_not_overwritten(evaluation_setup):
    setup = evaluation_setup

    evaluate_saved_model(
        setup["artifact_dir"],
        setup["test_path"],
        setup["output_dir"],
    )

    metrics_path = setup["output_dir"] / "test_metrics.json"
    original_bytes = metrics_path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_saved_model(
            setup["artifact_dir"],
            setup["test_path"],
            setup["output_dir"],
        )

    assert metrics_path.read_bytes() == original_bytes


def test_invalid_dataset_creates_no_report(evaluation_setup):
    setup = evaluation_setup
    records = [
        dict(record)
        for record in setup["partitions"]["test"]
    ]

    records[0]["duration_ms"] = -1
    write_records(records, setup["test_path"])

    with pytest.raises(ValueError, match="failed validation"):
        evaluate_saved_model(
            setup["artifact_dir"],
            setup["test_path"],
            setup["output_dir"],
        )

    assert not setup["output_dir"].exists()


def test_report_hashes_match_evaluated_files(evaluation_setup):
    setup = evaluation_setup

    report = evaluate_saved_model(
        setup["artifact_dir"],
        setup["test_path"],
        setup["output_dir"],
    )

    assert report["sources"]["model"]["sha256"] == sha256_file(
        setup["artifact_dir"] / MODEL_FILENAME
    )

    assert report["sources"]["metadata"]["sha256"] == sha256_file(
        setup["artifact_dir"] / METADATA_FILENAME
    )

    assert report["sources"]["test_dataset"]["sha256"] == sha256_file(
        setup["test_path"]
    )