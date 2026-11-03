# Model Selection and Development Error Analysis

## Decision

Select word TF-IDF with individual words and two-word sequences,
followed by logistic regression.

Experiment name: `word_12`.

This is a development-based selection among three fixed
configurations. It is not a claim of production readiness.

## Prediction task

Predict a synthetic engineering test-failure category from
the `error_message` field.

Categories:

- communication
- configuration
- resource_exhaustion
- timeout

Predictions are triage suggestions, not verified root-cause diagnoses.

## Dataset and split

The dataset contains 480 synthetic records from 16 scenario families.

| Partition | Records | Families |
|---|---:|---:|
| Train | 240 | 8 |
| Development | 120 | 4 |
| Test | 120 | 4 |

Each scenario family belongs to exactly one partition.

Each partition contains all four categories.

The split seed is 42.

The test partition has not been evaluated during model selection.

## Selection criterion

Primary metric: development macro-F1.

Supporting evidence:

- Accuracy
- Per-category precision, recall, and F1
- Confusion matrix
- Inspection of misclassified messages
- Representation complexity

No broad hyperparameter search was performed.

## Baseline

A most-frequent DummyClassifier produced:

| Metric | Development result |
|---|---:|
| Accuracy | 0.2500 |
| Macro-F1 | 0.1000 |

The baseline ignores failure-message features.

## Controlled comparison

All three candidates used logistic regression with the same settings.

| Experiment | Representation | Accuracy | Macro-F1 | Errors | Features |
|---|---|---:|---:|---:|---:|
| word_12 | Word TF-IDF, 1–2 grams | 0.6917 | 0.6677 | 37 | 259 |
| word_1 | Word TF-IDF, unigrams | 0.6833 | 0.6426 | 38 | 84 |
| char_35 | Within-word character TF-IDF, 3–5 grams | 0.6917 | 0.6144 | 37 | 1140 |

## Selected configuration

Input column: `error_message`

TF-IDF settings:

- analyzer: word
- ngram_range: (1, 2)
- lowercase: true
- min_df: 1

Classifier settings:

- LogisticRegression
- C: 1.0
- solver: lbfgs
- max_iter: 1000
- Other parameters: installed scikit-learn defaults

The generated metrics report records the scikit-learn version.

Vectorization and classification are fitted on training data only.

Development evaluation does not refit either component.

## Why retain word_12?

- Highest development macro-F1 among the three candidates.
- Joint-highest development accuracy.
- Highest configuration recall among the candidates.
- Fewer features than the character-based alternative.

The unigram-only candidate slightly reduced accuracy and macro-F1.

The character-based candidate improved communication and timeout
recall but failed to identify any configuration examples.

Its equal overall accuracy therefore did not indicate equivalent
category-level behavior.

## Selected-model category performance

Each category has 30 development examples.

| Category | Precision | Recall | F1 |
|---|---:|---:|---:|
| communication | 1.0000 | 0.5667 | 0.7234 |
| configuration | 1.0000 | 0.3000 | 0.4615 |
| resource_exhaustion | 0.6818 | 1.0000 | 0.8108 |
| timeout | 0.5400 | 0.9000 | 0.6750 |

A precision of 1.0 for communication or configuration does not
mean those categories are reliably detected: recall is much lower.

## Selected-model confusion matrix

Rows are actual categories.
Columns are predicted categories.

Column order:

1. communication
2. configuration
3. resource_exhaustion
4. timeout

| Actual category | Communication | Configuration | Resource exhaustion | Timeout |
|---|---:|---:|---:|---:|
| communication | 17 | 0 | 0 | 13 |
| configuration | 0 | 9 | 11 | 10 |
| resource_exhaustion | 0 | 0 | 30 | 0 |
| timeout | 0 | 0 | 3 | 27 |

## Error concentration

| Scenario family | Errors |
|---|---:|
| cfg_profile_rejected | 21 |
| comm_partial_transfer | 13 |
| time_completion_wait | 3 |
| res_buffer_unavailable | 0 |

## Representative errors

### Communication predicted as timeout

Message:

> i2c: Transfer stopped before the expected byte count after a reset.

Assigned label: communication  
Prediction: timeout

Interpretation:

The text describes an incomplete transfer, but does not establish
a unique underlying cause. The category comes from the synthetic
scenario definition.

### Configuration predicted as resource exhaustion

Message:

> i2c: Requested operating mode is not accepted during startup.

Assigned label: configuration  
Prediction: resource_exhaustion

Interpretation:

The development family uses rejected-mode/profile wording.
Training configuration families concern initial states and mappings.
Vocabulary differences are a plausible difficulty, but this example
alone does not prove which feature caused the prediction.

### Configuration predicted as timeout

Message:

> spi: Selected operating profile was rejected during repeated execution.

Assigned label: configuration  
Prediction: timeout

Interpretation:

Other examples from the same scenario family receive a different
incorrect category. Incidental context or subsystem wording may
influence predictions, but that hypothesis has not been isolated
in a controlled perturbation experiment.

### Timeout predicted as resource exhaustion

Message:

> uart: Operation remained pending beyond the expected interval after a recovery attempt.

Assigned label: timeout  
Prediction: resource_exhaustion

Interpretation:

The model does not consistently map all deadline-related wording
to the timeout category.

## Evaluation limitations

- All records are synthetic and template-based.
- Development contains only four scenario families.
- Thirty variations of a family are not thirty independent
  failure situations.
- Category frequencies are artificially balanced.
- Labels are assigned by scenario design, not verified diagnosis.
- Some messages are ambiguous without additional context.
- The model may use incidental subsystem and context wording.
- Performance differences have not been established as statistically
  significant.
- Real-world accuracy and reliability are unknown.

## Freeze and next step

Carry `word_12` into the packaging and final-evaluation stage.

Do not change the split, rewrite development examples, or repeatedly
tune against the reserved test partition.

For the first final evaluation, fit on the existing training partition
only and evaluate the frozen pipeline on the test partition once.

If later changes are informed by those test results, disclose that
the test set has become part of development rather than continuing
to describe it as untouched evidence.

## Reproduction

```bash
python -m triage.generate_data
python -m triage.validate_data
python -m triage.split_data
python -m triage.baseline
python -m triage.text_model
python -m triage.compare_models
python -m pytest -q
```

Run from the project root with the project environment activated.