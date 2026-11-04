import hashlib
import json

import joblib
import pytest

from triage.generate_data import generate_records, write_records
from triage.split_data import split_records
from triage.text_model import fit_text_model
from triage.train import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    train_and_save,
)


def make_training_file(tmp_path):
    records = generate_records(samples_per_family=2)
    partitions = split_records(records)

    train_path = tmp_path / "train.csv"
    write_records(partitions["train"], train_path)

    return train_path, partitions


def test_training_creates_model_and_metadata(tmp_path):
    train_path, _ = make_training_file(tmp_path)
    output_dir = tmp_path / "artifact"

    metadata = train_and_save(train_path, output_dir)

    assert (output_dir / MODEL_FILENAME).exists()
    assert (output_dir / METADATA_FILENAME).exists()

    saved_metadata = json.loads(
        (output_dir / METADATA_FILENAME).read_text(
            encoding="utf-8"
        )
    )

    assert saved_metadata == metadata
    assert metadata["experiment"] == "word_12"
    assert metadata["training"]["record_count"] == 16
    assert metadata["configuration"]["ngram_range"] == [1, 2]


def test_loaded_pipeline_preserves_predictions(tmp_path):
    train_path, partitions = make_training_file(tmp_path)
    output_dir = tmp_path / "artifact"

    train_and_save(train_path, output_dir)

    # This file is trusted: the test itself just created it.
    loaded_model = joblib.load(output_dir / MODEL_FILENAME)

    original_model = fit_text_model(partitions["train"])

    messages = [
        record["error_message"]
        for record in partitions["development"]
    ]

    assert (
        loaded_model.predict(messages).tolist()
        == original_model.predict(messages).tolist()
    )

    assert (
        loaded_model.named_steps["tfidf"].vocabulary_
        == original_model.named_steps["tfidf"].vocabulary_
    )


def test_metadata_identifies_training_data(tmp_path):
    train_path, partitions = make_training_file(tmp_path)
    output_dir = tmp_path / "artifact"

    metadata = train_and_save(train_path, output_dir)

    assert metadata["training"]["sha256"] == hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()

    assert set(metadata["training"]["incident_ids"]) == {
        record["incident_id"]
        for record in partitions["train"]
    }

    assert set(metadata["training"]["scenario_families"]) == {
        record["scenario_family"]
        for record in partitions["train"]
    }


def test_existing_artifact_directory_is_not_overwritten(tmp_path):
    train_path, _ = make_training_file(tmp_path)
    output_dir = tmp_path / "artifact"

    train_and_save(train_path, output_dir)

    model_bytes_before = (
        output_dir / MODEL_FILENAME
    ).read_bytes()

    with pytest.raises(FileExistsError):
        train_and_save(train_path, output_dir)

    assert (
        output_dir / MODEL_FILENAME
    ).read_bytes() == model_bytes_before


def test_invalid_training_data_does_not_create_artifact(tmp_path):
    records = generate_records(samples_per_family=1)
    records[0]["failure_category"] = "unknown"

    train_path = tmp_path / "invalid.csv"
    output_dir = tmp_path / "artifact"

    write_records(records, train_path)

    with pytest.raises(ValueError, match="failed validation"):
        train_and_save(train_path, output_dir)

    assert not output_dir.exists()