# Model Card: Synthetic Test-Failure Triage

## Summary

This project classifies synthetic engineering test-failure messages
into four categories using word TF-IDF and logistic regression.

The model is an educational portfolio artifact demonstrating:

- Reproducible synthetic-data generation
- Dataset validation
- Scenario-family-aware partitioning
- Baseline evaluation
- Controlled model comparison
- Model persistence and provenance
- A prediction CLI
- Held-out evaluation without retraining

It is not a production-ready failure-diagnosis system.

## Model identity

- Experiment: `word_12`
- Artifact directory: `artifacts/word_12_v1`
- Artifact format version: 1
- Training timestamp: `2026-11-04T17:57:24.742814+00:00`
- Input feature: `error_message`
- Output: one predicted failure category

Categories:

- communication
- configuration
- resource_exhaustion
- timeout

Predictions are triage suggestions, not verified root causes.

## Intended use

Educational experimentation and demonstration of a small,
reproducible text-classification workflow.

An engineer can submit a failure message through the CLI and
inspect the suggested category.

Every prediction should receive human review.

## Out-of-scope use

The model must not be presented as validated for:

- Production failure routing
- Safety-critical decisions
- Automotive safety assessment
- Verified root-cause diagnosis
- Automatic remediation
- Autonomous ticket assignment or incident closure

The project does not demonstrate measured user productivity
improvements or production business impact.

## Dataset

The dataset contains 480 synthetic records across 16 scenario families.

There are four families per category and 30 records per family.

Each record includes:

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

Labels are assigned by scenario design, not verified through
real incident investigation.

The selected model uses only `error_message` as its input column.

The message itself includes subsystem and context wording.

## Partitioning

Split seed: 42.

Each scenario family belongs entirely to one partition.

| Partition | Records | Families | Records per category |
|---|---:|---:|---:|
| Train | 240 | 8 | 60 |
| Development | 120 | 4 | 30 |
| Test | 120 | 4 | 30 |

The artifact was fitted on the 240 training records only.

Development data was used for model selection.

The test partition was evaluated after the selected configuration
and artifact were fixed.

### Test scenario families

- cfg_parameter_mismatch
- comm_link_interruption
- res_queue_capacity
- time_recovery_wait

## Model configuration

Text representation:

- TfidfVectorizer
- analyzer: word
- ngram_range: (1, 2)
- lowercase: true
- min_df: 1
- Learned vocabulary size: 259

Classifier:

- LogisticRegression
- C: 1.0
- solver: lbfgs
- max_iter: 1000

Other estimator settings use the recorded scikit-learn version's
defaults.

The saved pipeline includes both the fitted vectorizer and classifier.

## Development-based model selection

Primary selection metric: development macro-F1.

| Experiment | Accuracy | Macro-F1 | Errors | Features |
|---|---:|---:|---:|---:|
| Most-frequent dummy baseline | 0.2500 | 0.1000 | 90 | N/A |
| Word TF-IDF, 1–2 grams | 0.6917 | 0.6677 | 37 | 259 |
| Word TF-IDF, unigrams | 0.6833 | 0.6426 | 38 | 84 |
| Within-word character TF-IDF, 3–5 grams | 0.6917 | 0.6144 | 37 | 1140 |

The word 1–2 gram configuration was retained because it achieved
the highest development macro-F1 and joint-highest accuracy.

No statistical significance claim is made.

Detailed development analysis is recorded in `model_selection.md`.

## Final held-out test results

The saved artifact was loaded and evaluated without retraining.

| Metric | Result |
|---|---:|
| Records | 120 |
| Scenario families | 4 |
| Correct predictions | 91 |
| Incorrect predictions | 29 |
| Accuracy | 0.7583 |
| Macro-F1 | 0.7504 |

The higher test score relative to development describes performance
on different synthetic families. It does not indicate that the model
improved between evaluations.

### Per-category metrics

Each category has 30 test records.

| Category | Precision | Recall | F1 | Correct |
|---|---:|---:|---:|---:|
| communication | 0.5294 | 0.6000 | 0.5625 | 18/30 |
| configuration | 0.9333 | 0.4667 | 0.6222 | 14/30 |
| resource_exhaustion | 1.0000 | 1.0000 | 1.0000 | 30/30 |
| timeout | 0.7073 | 0.9667 | 0.8169 | 29/30 |

### Confusion matrix

Rows are actual categories.
Columns are predicted categories.

| Actual category | Communication | Configuration | Resource exhaustion | Timeout |
|---|---:|---:|---:|---:|
| communication | 18 | 0 | 0 | 12 |
| configuration | 16 | 14 | 0 | 0 |
| resource_exhaustion | 0 | 0 | 30 | 0 |
| timeout | 0 | 1 | 0 | 29 |

### Main observed weaknesses

- Configuration recall is below 0.5.
- Sixteen configuration examples were predicted as communication.
- Twelve communication examples were predicted as timeout.
- Perfect resource-exhaustion performance applies only to the
  single held-out resource scenario family.

The confusion matrix establishes where errors occurred. It does not
by itself establish which words caused each prediction.

## Evaluation safeguards

The recorded final evaluation checks all passed:

- Dataset validation
- No overlap with recorded training incident IDs
- No overlap with recorded training scenario families
- All expected categories present
- Model, metadata, and test files unchanged during evaluation

Training-membership checks rely on artifact metadata.

They do not independently prove that test data was never used
during development.

## Provenance

Training dataset SHA-256:

```text
01059dd49480fb71b53e4c0d8c619a3f7d98a30b564b467534425c6d4cf433f6
```

Saved model SHA-256:

```text
991769e012e23821fe4a62b17f7f45b9d99f96b08bb5d581da96d0d1c2fe5c31
```

Artifact metadata SHA-256:

```text
a4ccd0c900bde147180885489250027265f17d2fed99b5a4e206d10d60bb8526
```

Test dataset SHA-256:

```text
677baa82b6d5730ac96c5ac92e0cdf940d0f86ef963e69cbffc54377702dfe40
```

Hashes identify file contents. They do not establish safety,
authenticity, or model quality.

## Recorded training environment

| Component | Version |
|---|---|
| Python | 3.14.4 |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| joblib | 1.6.0 |

An installed-package snapshot is stored in:

`requirements-macos-py314.txt`

The prediction loader currently requires exact matches for its
recorded Python and package-version checks.

## Prediction interface

From the project root:

```bash
python -m triage.predict \
  --message "Received data failed the integrity check"
```

Observed example output:

```json
{
  "predicted_category": "communication",
  "requires_human_review": true,
  "notice": "Synthetic-data triage suggestion, not a verified root-cause diagnosis.",
  "experiment": "word_12"
}
```

The CLI:

- Rejects blank input
- Limits messages to 10,000 characters
- Loads the saved pipeline without retraining
- Checks recorded environment versions
- Returns JSON
- Returns exit status 2 on input or artifact errors

The human-review flag is informational.
The CLI does not enforce a review or approval workflow.

## Security and operational limitations

- Load only trusted joblib artifacts.
- Metadata validation is not a security sandbox.
- The model has no abstention mechanism.
- Unrelated but valid text may receive a category.
- Confidence calibration has not been evaluated.
- Real-world robustness has not been evaluated.
- Production latency and throughput have not been benchmarked.
- There is no authentication, hosted API, or production monitoring
  in this small CLI project.

## Data and evaluation limitations

- All records are synthetic and template-based.
- Categories are artificially balanced.
- Each development/test category contains only one scenario family.
- Variations of a family are not independent real-world incidents.
- Failure messages may not uniquely establish their assigned category.
- Incidental subsystem and context wording may influence predictions.
- The generator's coverage limits what evaluation can demonstrate.
- Real-world generalization is unknown.

## Verification

At completion of final-evaluation implementation:

```bash
python -m pytest -q
```

Observed result: 84 passed.

Tests verify specific implementation behavior and safeguards.
They do not establish model accuracy on unseen real-world data.

## Reproduction and report preservation

See the README for setup, data generation, splitting, training,
prediction, and evaluation commands.

Artifact training refuses to reuse an existing artifact directory.

Final evaluation refuses to reuse an existing evaluation directory.

Preserve the first final-evaluation report.
Do not delete it merely to replace an unfavorable score.

If future changes are informed by the held-out test results,
document those changes as further development and obtain a new
independent evaluation set before making fresh held-out claims.

## Future work

Potential future experiments, not implemented claims:

- Acquire appropriately permitted and de-identified real failure data.
- Expand the number and diversity of scenario families.
- Review category definitions and ambiguous cases with domain experts.
- Evaluate robustness to incidental context and subsystem wording.
- Investigate abstention and probability calibration.
- Evaluate whether additional structured inputs provide useful signals.

Any experiment informed by the current test errors must be
distinguished from this frozen evaluation.