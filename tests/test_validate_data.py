import csv

import pytest

from triage.generate_data import (
    FIELDNAMES,
    generate_records,
    write_records,
)
from triage.validate_data import validate_dataset


def issue_codes(report):
    return {
        issue["code"]
        for issue in report["issues"]
    }


def test_generated_dataset_passes_validation(tmp_path):
    records = generate_records(samples_per_family=2)
    input_path = tmp_path / "valid.csv"

    write_records(records, input_path)

    report = validate_dataset(input_path)

    assert report["is_valid"] is True
    assert report["records_checked"] == 32
    assert report["unique_incident_ids"] == 32
    assert report["scenario_family_count"] == 16
    assert report["issue_count"] == 0


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_code"),
    [
        ("error_message", "   ", "blank_value"),
        ("failure_category", "unknown", "unknown_category"),
        ("duration_ms", "-10", "negative_value"),
        ("retry_count", "1.5", "invalid_integer"),
        ("timestamp", "not-a-date", "invalid_timestamp"),
        (
            "timestamp",
            "2026-01-01T00:00:00",
            "missing_timezone",
        ),
    ],
)
def test_invalid_values_are_reported(
    tmp_path,
    field,
    bad_value,
    expected_code,
):
    records = generate_records(samples_per_family=1)
    records[0][field] = bad_value
    input_path = tmp_path / "invalid.csv"

    write_records(records, input_path)

    report = validate_dataset(input_path)

    assert report["is_valid"] is False
    assert expected_code in issue_codes(report)


def test_duplicate_incident_id_is_reported(tmp_path):
    records = generate_records(samples_per_family=1)
    records[1]["incident_id"] = records[0]["incident_id"]
    input_path = tmp_path / "duplicates.csv"

    write_records(records, input_path)

    report = validate_dataset(input_path)

    assert "duplicate_incident_id" in issue_codes(report)


def test_family_cannot_have_multiple_categories(tmp_path):
    records = generate_records(samples_per_family=1)

    first = records[0]
    other = next(
        record
        for record in records
        if record["failure_category"] != first["failure_category"]
    )

    other["scenario_family"] = first["scenario_family"]
    input_path = tmp_path / "mixed_family.csv"

    write_records(records, input_path)

    report = validate_dataset(input_path)

    assert "inconsistent_family_category" in issue_codes(report)


def test_missing_column_is_reported(tmp_path):
    input_path = tmp_path / "missing_column.csv"

    input_path.write_text(
        "incident_id,failure_category\nINC-001,timeout\n",
        encoding="utf-8",
    )

    report = validate_dataset(input_path)

    assert "missing_columns" in issue_codes(report)
    assert report["records_checked"] == 0


def test_empty_dataset_is_reported(tmp_path):
    input_path = tmp_path / "empty.csv"

    write_records([], input_path)

    report = validate_dataset(input_path)

    assert "empty_dataset" in issue_codes(report)


def test_extra_csv_cell_is_reported(tmp_path):
    input_path = tmp_path / "extra_cell.csv"
    record = generate_records(samples_per_family=1)[0]

    with input_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(FIELDNAMES)
        writer.writerow(
            [record[field] for field in FIELDNAMES]
            + ["unexpected_cell"]
        )

    report = validate_dataset(input_path)

    assert "extra_cells" in issue_codes(report)


def test_missing_csv_cell_is_reported(tmp_path):
    input_path = tmp_path / "missing_cell.csv"
    record = generate_records(samples_per_family=1)[0]

    with input_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(FIELDNAMES)
        writer.writerow(
            [record[field] for field in FIELDNAMES[:-1]]
        )

    report = validate_dataset(input_path)

    assert "missing_value" in issue_codes(report)


def test_duplicate_header_is_reported(tmp_path):
    input_path = tmp_path / "duplicate_header.csv"

    with input_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(FIELDNAMES + ["incident_id"])

    report = validate_dataset(input_path)

    assert "duplicate_headers" in issue_codes(report)