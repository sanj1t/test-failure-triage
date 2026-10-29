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