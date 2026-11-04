"""Predict a failure category using a trusted local artifact."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from triage.labels import VALID_CATEGORIES
from triage.train import METADATA_FILENAME, MODEL_FILENAME


MAX_MESSAGE_LENGTH = 10_000


def validate_message(message: str) -> str:
    """Validate and normalize a single input message."""

    if not isinstance(message, str):
        raise ValueError("Message must be a string.")

    message = message.strip()

    if not message:
        raise ValueError("Message must not be empty.")

    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            f"Message must not exceed {MAX_MESSAGE_LENGTH} characters."
        )

    return message


def load_artifact(artifact_dir: Path) -> tuple[Pipeline, dict]:
    """Load a trusted artifact after basic compatibility checks.

    These checks do not make untrusted joblib files safe to load.
    """

    metadata_path = artifact_dir / METADATA_FILENAME
    model_path = artifact_dir / MODEL_FILENAME

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if not isinstance(metadata, dict):
        raise ValueError("Artifact metadata must be a JSON object.")

    if metadata.get("artifact_format_version") != 1:
        raise ValueError("Unsupported artifact format version.")

    if metadata.get("experiment") != "word_12":
        raise ValueError("Artifact is not the selected word_12 model.")

    environment = metadata.get("environment")

    if not isinstance(environment, dict):
        raise ValueError("Artifact environment metadata is missing.")

    current_environment = {
        "python": platform.python_version(),
        "scikit-learn": version("scikit-learn"),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "joblib": version("joblib"),
    }

    for package, current_version in current_environment.items():
        saved_version = environment.get(package)

        if saved_version != current_version:
            raise ValueError(
                f"Environment mismatch for {package}: "
                f"artifact={saved_version!r}, "
                f"current={current_version!r}. "
                "Use the artifact's training environment."
            )

    labels = metadata.get("labels")

    if (
        not isinstance(labels, list)
        or not all(isinstance(label, str) for label in labels)
        or len(labels) != len(VALID_CATEGORIES)
        or set(labels) != VALID_CATEGORIES
    ):
        raise ValueError("Artifact labels do not match project categories.")

    # Only load artifacts from a trusted source.
    model = joblib.load(model_path)

    if not isinstance(model, Pipeline):
        raise ValueError("Artifact does not contain a Pipeline.")

    if list(model.named_steps) != ["tfidf", "classifier"]:
        raise ValueError("Artifact has unexpected pipeline steps.")

    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "classes_"):
        raise ValueError("Artifact classifier is not fitted.")

    if classifier.classes_.tolist() != labels:
        raise ValueError("Model labels do not match artifact metadata.")

    return model, metadata


def predict_message(
    model: Pipeline,
    message: str,
) -> dict:
    """Predict one category without fitting or modifying the model."""

    message = validate_message(message)

    prediction = str(model.predict([message])[0])

    if prediction not in VALID_CATEGORIES:
        raise ValueError(
            f"Model returned an unsupported category: {prediction}"
        )

    return {
        "predicted_category": prediction,
        "requires_human_review": True,
        "notice": (
            "Synthetic-data triage suggestion, "
            "not a verified root-cause diagnosis."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict using a trusted local failure-triage artifact."
        )
    )

    parser.add_argument(
        "--message",
        required=True,
        help="Failure message to classify.",
    )

    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/word_12_v1"),
        help="Directory containing a trusted model and metadata.",
    )

    args = parser.parse_args()

    try:
        # Reject bad input before loading the artifact.
        message = validate_message(args.message)

        model, metadata = load_artifact(args.artifact_dir)
        result = predict_message(model, message)

    except Exception as error:
        # CLI boundary: provide a failure status, not a false prediction.
        parser.exit(
            status=2,
            message=f"Prediction failed: {error}\n",
        )

    result["experiment"] = metadata["experiment"]

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()