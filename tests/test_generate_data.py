import csv
from collections import Counter

import pytest

from triage.generate_data import (
    FIELDNAMES,
    SCENARIOS,
    generate_records,
    write_records,
)
from triage.labels import VALID_CATEGORIES


def test_record_count_matches_requested_samples():
    records = generate_records(samples_per_family=3)

    assert len(records) == len(SCENARIOS) * 3


def test_same_seed_produces_identical_records():
    first = generate_records(samples_per_family=3, seed=42)
    second = generate_records(samples_per_family=3, seed=42)

    assert first == second


def test_different_seed_changes_records():
    first = generate_records(samples_per_family=3, seed=42)
    second = generate_records(samples_per_family=3, seed=99)

    assert first != second


def test_generated_records_follow_basic_contract():
    records = generate_records(samples_per_family=3)

    for record in records:
        assert set(record) == set(FIELDNAMES)
        assert record["failure_category"] in VALID_CATEGORIES
        assert record["error_message"].strip()
        assert record["duration_ms"] >= 0
        assert record["retry_count"] >= 0


def test_incident_ids_are_unique():
    records = generate_records(samples_per_family=3)

    ids = [record["incident_id"] for record in records]

    assert len(ids) == len(set(ids))


def test_each_family_has_requested_number_of_samples():
    records = generate_records(samples_per_family=3)

    counts = Counter(
        record["scenario_family"]
        for record in records
    )

    assert set(counts) == {
        scenario.family
        for scenario in SCENARIOS
    }

    assert all(count == 3 for count in counts.values())


def test_scenario_families_are_unique_and_cover_categories():
    families = [scenario.family for scenario in SCENARIOS]

    assert len(families) == len(set(families))

    counts = Counter(
        scenario.category.value
        for scenario in SCENARIOS
    )

    assert set(counts) == VALID_CATEGORIES
    assert all(count == 4 for count in counts.values())


def test_nonpositive_sample_count_is_rejected():
    for value in (0, -1):
        with pytest.raises(ValueError):
            generate_records(samples_per_family=value)


def test_write_records_creates_readable_csv(tmp_path):
    records = generate_records(samples_per_family=2)
    output_path = tmp_path / "nested" / "test_failures.csv"

    write_records(records, output_path)

    with output_path.open(
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

        assert reader.fieldnames == FIELDNAMES

    assert len(rows) == len(records)
    assert rows[0]["incident_id"] == records[0]["incident_id"]
    assert int(rows[0]["duration_ms"]) == records[0]["duration_ms"]