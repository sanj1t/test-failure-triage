import json
import subprocess
import sys

import numpy as np
import pytest

from triage.generate_data import generate_records, write_records
from triage.labels import VALID_CATEGORIES
from triage.predict import (
    MAX_MESSAGE_LENGTH,
    load_artifact,
    predict_message,
    validate_message,
)
from triage.split_data import split_records
from triage.train import METADATA_FILENAME, train_and_save


@pytest.fixture
def artifact_dir(tmp_path):
    records = generate_records(samples_per_family=2)
    partitions = split_records(records)

    train_path = tmp_path / "train.csv"
    write_records(partitions["train"], train_path)

    output_dir = tmp_path / "artifact"
    train_and_save(train_path, output_dir)

    return output_dir


def test_message_is_stripped():
    assert validate_message("  transfer failed  ") == "transfer failed"


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_blank_message_is_rejected(message):
    with pytest.raises(ValueError, match="must not be empty"):
        validate_message(message)


def test_non_string_message_is_rejected():
    with pytest.raises(ValueError, match="must be a string"):
        validate_message(None)


def test_overlong_message_is_rejected():
    message = "x" * (MAX_MESSAGE_LENGTH + 1)

    with pytest.raises(ValueError, match="must not exceed"):
        validate_message(message)


def test_loaded_artifact_produces_valid_prediction(artifact_dir):
    model, metadata = load_artifact(artifact_dir)

    result = predict_message(
        model,
        "Received data failed the integrity check",
    )

    assert metadata["experiment"] == "word_12"
    assert result["predicted_category"] in VALID_CATEGORIES
    assert result["requires_human_review"] is True


def test_prediction_does_not_refit_model(artifact_dir):
    model, _ = load_artifact(artifact_dir)

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    vocabulary_before = vectorizer.vocabulary_.copy()
    idf_before = vectorizer.idf_.copy()
    coefficients_before = classifier.coef_.copy()

    predict_message(model, "uniquepredictiontoken")

    assert vectorizer.vocabulary_ == vocabulary_before
    np.testing.assert_array_equal(vectorizer.idf_, idf_before)
    np.testing.assert_array_equal(
        classifier.coef_,
        coefficients_before,
    )


def test_environment_mismatch_is_rejected(artifact_dir):
    metadata_path = artifact_dir / METADATA_FILENAME
    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    metadata["environment"]["scikit-learn"] = "0.0.0"

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Environment mismatch"):
        load_artifact(artifact_dir)


def test_missing_artifact_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "missing")


def test_cli_prints_json(artifact_dir):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "triage.predict",
            "--artifact-dir",
            str(artifact_dir),
            "--message",
            "Received data failed the integrity check",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr

    result = json.loads(completed.stdout)

    assert result["predicted_category"] in VALID_CATEGORIES
    assert result["experiment"] == "word_12"


def test_cli_rejects_blank_message():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "triage.predict",
            "--message",
            "   ",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "must not be empty" in completed.stderr