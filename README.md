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