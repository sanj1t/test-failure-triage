# Test-Failure Triage

A small machine-learning project that suggests failure categories
from engineering test-failure records.

## Intended user

An engineer reviewing failed automated tests.

## Prediction task

Given a failure message and structured test metadata, predict one of:

- configuration
- timeout
- communication
- resource_exhaustion

The output is a triage suggestion, not a verified root-cause diagnosis.

## Planned input fields

- incident_id
- scenario_family
- timestamp
- board
- subsystem
- test_name
- error_message
- duration_ms
- retry_count
- failure_category

`failure_category` is the training target.

`incident_id`, `scenario_family`, and `timestamp` will not be
used as input features in the initial model.

`scenario_family` will be used to keep related synthetic examples
together when splitting the dataset.

## Data policy

The initial dataset will be synthetic and reproducible.

No employer logs, customer data, credentials, or proprietary
documents will be included.

Results on synthetic data do not establish real-world performance.

## Current status

- Project skeleton
- Shared failure-category definitions
- Initial unit tests

## Development setup

Requires Python 3.12 or newer.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Generate the synthetic dataset

From the project root:

```bash
python -m triage.generate_data
```

Default output:

- File: `data/raw/test_failures.csv`
- Records: 480
- Categories: 4
- Scenario families: 16
- Samples per family: 30
- Seed: 42

Custom generation:

```bash
python -m triage.generate_data \
  --samples-per-family 40 \
  --seed 99 \
  --output data/raw/test_failures_custom.csv
```

Generation replaces an existing file at the selected output path.

### Limitations

Labels are assigned by synthetic scenario design.

The dataset is balanced and template-based; it does not represent
real-world failure frequencies or establish production performance.

Related examples share a scenario family. Later dataset splitting
must keep each family entirely within one partition.

The model must not use `scenario_family`, `incident_id`, or
`timestamp` as input features.

## Validate a dataset

```bash
python -m triage.validate_data
```

Custom input and report:

```bash
python -m triage.validate_data \
  --input data/raw/test_failures.csv \
  --report reports/data_quality.json
```

The validator reports problems without modifying the dataset.

Checks include:

- Required, unexpected, and duplicate columns
- Empty datasets and missing values
- Duplicate incident IDs
- Unknown categories
- Invalid or negative integer fields
- Invalid timestamps and missing timezone information
- Inconsistent category assignments within a scenario family
- Extra or missing CSV cells

Exit codes:

- 0: validation passed
- 1: validation failed
- 2: command or file-processing error

A validation pass checks structural and basic consistency rules.
It does not establish label correctness, realism, or model quality.

## Split the dataset

```bash
python -m triage.split_data
```

Outputs:

- `data/processed/train.csv`
- `data/processed/development.csv`
- `data/processed/test.csv`
- `data/processed/split_manifest.json`

The split keeps each scenario family entirely within one partition.

For each category:

- One family is reserved for development.
- One family is reserved for testing.
- Remaining families are assigned to training.

With the default dataset, this produces:

| Partition | Records | Families |
|---|---:|---:|
| Train | 240 | 8 |
| Development | 120 | 4 |
| Test | 120 | 4 |

The default split seed is 42.

Training fits model parameters and preprocessing.
Development supports model selection.
Testing is reserved for the final evaluation.

This split evaluates unseen synthetic scenario families.
It does not establish performance on real engineering failures.

## Dummy baseline

Install project dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the baseline:

```bash
python -m triage.baseline
```

The baseline uses `DummyClassifier(strategy="most_frequent")`.

It fits category frequencies using training labels and always
predicts the selected most-frequent training category.

It does not learn from failure messages or structured features.

Development results on the default split:

| Metric | Value |
|---|---:|
| Accuracy | 0.25 |
| Macro-F1 | 0.10 |

Generated outputs:

- `reports/baseline/development_metrics.json`
- `reports/baseline/development_predictions.csv`

The test partition is not evaluated during this step.

The baseline establishes a reference result, not a useful
failure-triage capability.

## Text classifier: TF-IDF + logistic regression

Run:

```bash
python -m triage.text_model
```

The model uses only `error_message` as its input column.

Pipeline:

1. TF-IDF over words and two-word sequences.
2. Logistic regression.

The vectorizer and classifier are fitted on training data only.
Development data is transformed and evaluated without refitting.

Outputs:

- `reports/text_logreg/development_metrics.json`
- `reports/text_logreg/development_predictions.csv`
- `reports/text_logreg/development_errors.csv`

The test partition remains unevaluated during model selection.

Results apply to a small synthetic, template-based dataset.
They do not establish performance on real engineering failures.

### Initial text-model development results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Most-frequent baseline | 0.2500 | 0.1000 |
| Word TF-IDF (1–2 grams) + logistic regression | 0.6917 | 0.6677 |

The text model correctly classified 83 of 120 development records.

The vectorizer learned 259 features from training data.

Development contains four held-out synthetic scenario families.
Results do not establish real-world performance.

The test partition has not been evaluated.

## Controlled representation comparison

```bash
python -m triage.compare_models
```

Compares three fixed configurations:

- Word TF-IDF, 1–2 grams + logistic regression
- Word TF-IDF, unigrams only + logistic regression
- Within-word character TF-IDF, 3–5 grams + logistic regression

Classifier settings are held constant.

All candidates fit on training data and are evaluated on development
data only. The test partition remains unevaluated.

Primary comparison metric: development macro-F1, supported by
per-category recall and error inspection.

### Development comparison results

| Experiment | Accuracy | Macro-F1 | Errors | Features |
|---|---:|---:|---:|---:|
| Word TF-IDF, 1–2 grams | 0.6917 | 0.6677 | 37 | 259 |
| Word TF-IDF, unigrams | 0.6833 | 0.6426 | 38 | 84 |
| Character TF-IDF, 3–5 grams | 0.6917 | 0.6144 | 37 | 1140 |

Provisional selection: word TF-IDF with 1–2 grams and logistic
regression.

Reason: highest development macro-F1 and joint-highest accuracy
among the three fixed configurations.

Character features did not justify replacing the original model:
they increased the feature count, did not improve accuracy,
and produced lower macro-F1.

These results describe four synthetic development scenario families.
They do not establish real-world performance or statistical
superiority across other datasets.

The test partition remains unevaluated.

## Model-selection decision

Selected configuration: word TF-IDF with 1–2 grams followed by
logistic regression.

Development accuracy: 0.6917  
Development macro-F1: 0.6677

See [Model selection and error analysis](docs/model_selection.md)
for the controlled comparison, category-level results, representative
errors, and limitations.

The reserved test partition has not yet been evaluated.

## Train and save the selected pipeline

```bash
python -m triage.train
```

Default artifact:

- `artifacts/word_12_v1/model.joblib`
- `artifacts/word_12_v1/metadata.json`

The artifact contains the fitted TF-IDF vectorizer and classifier.

Training uses the existing training partition only.
Development and test records are not used to fit this artifact.

Metadata records:

- Training-file hash
- Training incident IDs and scenario families
- Model configuration
- Python and dependency versions
- Model limitations

The command refuses to reuse an existing artifact directory.

Only load joblib artifacts created by this project or obtained
from a trusted, verified source. Loading untrusted artifacts
can execute code.

Use the matching training environment when loading the model.

The installed-package snapshot is recorded in
`requirements-macos-py314.txt`.

## Predict a failure category

From the project root:

```bash
python -m triage.predict \
  --message "Received data failed the integrity check"
```

The command loads the saved artifact at `artifacts/word_12_v1`.
It does not retrain the model.

To select another trusted local artifact:

```bash
python -m triage.predict \
  --artifact-dir artifacts/word_12_v1 \
  --message "Transfer stopped unexpectedly"
```

Behavior:

- Rejects blank messages.
- Limits messages to 10,000 characters.
- Checks recorded environment versions before loading.
- Prints a JSON prediction.
- Returns exit status 2 on input or artifact errors.

Only load trusted joblib artifacts.

Predictions are synthetic-data triage suggestions, not verified
root-cause diagnoses. Human review is required.

The review flag is informational; this CLI does not enforce an
approval workflow.

The model has no abstention mechanism: valid nonempty text may receive
a category even when it is unrelated to the training domain.

## Final held-out evaluation

Run only after freezing the selected model:

```bash
python -m triage.evaluate
```

Default inputs:

- Artifact: `artifacts/word_12_v1`
- Held-out dataset: `data/processed/test.csv`

Outputs:

- `reports/final_test/word_12_v1/test_metrics.json`
- `reports/final_test/word_12_v1/test_predictions.csv`
- `reports/final_test/word_12_v1/test_errors.csv`

The command:

- Loads the saved pipeline without retraining.
- Validates the evaluation dataset.
- Rejects overlap with recorded training IDs or scenario families.
- Records model, metadata, and dataset SHA-256 hashes.
- Refuses to overwrite an existing evaluation directory.

Training-membership checks depend on artifact metadata.
They do not independently prove that test data was never used
during development.

The final test contains 120 synthetic records from four scenario
families. Results do not establish real-world performance.

Any later model changes informed by these test results must be
documented as further development.