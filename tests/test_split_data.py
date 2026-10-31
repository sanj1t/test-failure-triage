from collections import Counter

import pytest

from triage.generate_data import generate_records
from triage.labels import VALID_CATEGORIES
from triage.split_data import (
    PARTITIONS,
    split_records,
    summarize_partitions,
)


def test_default_data_has_expected_partition_sizes():
    records = generate_records()
    partitions = split_records(records)

    assert len(partitions["train"]) == 240
    assert len(partitions["development"]) == 120
    assert len(partitions["test"]) == 120


def test_scenario_families_do_not_overlap():
    records = generate_records(samples_per_family=2)
    partitions = split_records(records)

    families = {
        name: {
            record["scenario_family"]
            for record in partitions[name]
        }
        for name in PARTITIONS
    }

    assert families["train"].isdisjoint(
        families["development"]
    )
    assert families["train"].isdisjoint(
        families["test"]
    )
    assert families["development"].isdisjoint(
        families["test"]
    )


def test_every_record_appears_exactly_once():
    records = generate_records(samples_per_family=2)
    partitions = split_records(records)

    original_ids = Counter(
        record["incident_id"]
        for record in records
    )

    partitioned_ids = Counter(
        record["incident_id"]
        for name in PARTITIONS
        for record in partitions[name]
    )

    assert partitioned_ids == original_ids


def test_every_partition_contains_all_categories():
    records = generate_records(samples_per_family=2)
    partitions = split_records(records)

    for name in PARTITIONS:
        categories = {
            record["failure_category"]
            for record in partitions[name]
        }

        assert categories == VALID_CATEGORIES


def test_same_seed_produces_identical_partitions():
    records = generate_records(samples_per_family=2)

    first = split_records(records, seed=42)
    second = split_records(records, seed=42)

    assert first == second


def test_too_few_families_is_rejected():
    records = generate_records(samples_per_family=2)

    configuration_families = sorted({
        record["scenario_family"]
        for record in records
        if record["failure_category"] == "configuration"
    })

    removed_families = set(configuration_families[:2])

    reduced_records = [
        record
        for record in records
        if record["scenario_family"] not in removed_families
    ]

    with pytest.raises(
        ValueError,
        match="at least three scenario families",
    ):
        split_records(reduced_records)


def test_summary_matches_partition_contents():
    records = generate_records(samples_per_family=2)
    partitions = split_records(records)

    summary = summarize_partitions(partitions)

    assert summary["train"]["record_count"] == 16
    assert summary["train"]["family_count"] == 8

    assert summary["development"]["record_count"] == 8
    assert summary["development"]["family_count"] == 4

    assert summary["test"]["record_count"] == 8
    assert summary["test"]["family_count"] == 4


def test_family_with_multiple_categories_is_rejected():
    records = generate_records(samples_per_family=2)

    first = records[0]

    other = next(
        record
        for record in records
        if record["failure_category"] != first["failure_category"]
    )

    other["scenario_family"] = first["scenario_family"]

    with pytest.raises(
        ValueError,
        match="multiple categories",
    ):
        split_records(records)