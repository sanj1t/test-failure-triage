"""Validate a test-failure CSV and write a JSON quality report."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from triage.generate_data import FIELDNAMES
from triage.labels import VALID_CATEGORIES


def validate_dataset(input_path: Path) -> dict:
    """Inspect a CSV without modifying the input file."""

    issues = []
    category_counts = Counter()
    family_counts = Counter()

    seen_ids = set()
    family_categories = {}
    total_records = 0

    def add_issue(
        code: str,
        message: str,
        record_number: int | None = None,
    ) -> None:
        issues.append(
            {
                "code": code,
                "record_number": record_number,
                "message": message,
            }
        )

    with input_path.open(
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file, strict=True)
        headers = reader.fieldnames or []

        required = set(FIELDNAMES)
        actual = set(headers)

        missing_columns = sorted(required - actual)
        unexpected_columns = sorted(actual - required)

        duplicate_headers = sorted(
            name
            for name, count in Counter(headers).items()
            if count > 1
        )

        if missing_columns:
            add_issue(
                "missing_columns",
                f"Missing columns: {missing_columns}",
            )

        if unexpected_columns:
            add_issue(
                "unexpected_columns",
                f"Unexpected columns: {unexpected_columns}",
            )

        if duplicate_headers:
            add_issue(
                "duplicate_headers",
                f"Duplicate headers: {duplicate_headers}",
            )

        schema_is_valid = not (
            missing_columns
            or unexpected_columns
            or duplicate_headers
        )

        if schema_is_valid:
            for record_number, row in enumerate(reader, start=1):
                total_records += 1

                if None in row:
                    add_issue(
                        "extra_cells",
                        "Record contains more cells than the header.",
                        record_number,
                    )

                values = {}

                for field in FIELDNAMES:
                    raw_value = row.get(field)

                    if raw_value is None:
                        add_issue(
                            "missing_value",
                            f"Missing cell for '{field}'.",
                            record_number,
                        )
                        continue

                    value = raw_value.strip()

                    if not value:
                        add_issue(
                            "blank_value",
                            f"Blank value for '{field}'.",
                            record_number,
                        )
                        continue

                    values[field] = value

                incident_id = values.get("incident_id")

                if incident_id is not None:
                    if incident_id in seen_ids:
                        add_issue(
                            "duplicate_incident_id",
                            f"Duplicate incident ID: {incident_id}",
                            record_number,
                        )

                    seen_ids.add(incident_id)

                category = values.get("failure_category")

                if category is not None:
                    category_counts[category] += 1

                    if category not in VALID_CATEGORIES:
                        add_issue(
                            "unknown_category",
                            f"Unsupported category: {category}",
                            record_number,
                        )

                for field in ("duration_ms", "retry_count"):
                    value = values.get(field)

                    if value is None:
                        continue

                    try:
                        number = int(value)
                    except ValueError:
                        add_issue(
                            "invalid_integer",
                            f"'{field}' must be an integer: {value!r}",
                            record_number,
                        )
                        continue

                    if number < 0:
                        add_issue(
                            "negative_value",
                            f"'{field}' must not be negative.",
                            record_number,
                        )

                timestamp = values.get("timestamp")

                if timestamp is not None:
                    try:
                        parsed = datetime.fromisoformat(timestamp)
                    except ValueError:
                        add_issue(
                            "invalid_timestamp",
                            f"Invalid ISO timestamp: {timestamp!r}",
                            record_number,
                        )
                    else:
                        if parsed.utcoffset() is None:
                            add_issue(
                                "missing_timezone",
                                "Timestamp must include timezone information.",
                                record_number,
                            )

                family = values.get("scenario_family")

                if family is not None:
                    family_counts[family] += 1

                    if category in VALID_CATEGORIES:
                        if family not in family_categories:
                            family_categories[family] = category
                        elif family_categories[family] != category:
                            add_issue(
                                "inconsistent_family_category",
                                (
                                    f"Family '{family}' has more than "
                                    "one failure category."
                                ),
                                record_number,
                            )

            if total_records == 0:
                add_issue(
                    "empty_dataset",
                    "Dataset contains no records.",
                )

    return {
        "input_path": str(input_path),
        "is_valid": not issues,
        "records_checked": total_records,
        "unique_incident_ids": len(seen_ids),
        "scenario_family_count": len(family_counts),
        "category_counts": dict(sorted(category_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "issue_count": len(issues),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a test-failure CSV."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/test_failures.csv"),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/data_quality.json"),
    )

    args = parser.parse_args()

    if args.input.resolve() == args.report.resolve():
        parser.error("Input and report paths must be different.")

    try:
        report = validate_dataset(args.input)
    except (OSError, UnicodeError, csv.Error) as error:
        parser.exit(
            status=2,
            message=f"Unable to read dataset: {error}\n",
        )

    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)

        args.report.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        parser.exit(
            status=2,
            message=f"Unable to write report: {error}\n",
        )

    print(f"Records checked: {report['records_checked']}")
    print(f"Validation issues: {report['issue_count']}")
    print(f"Report: {args.report}")

    if not report["is_valid"]:
        print("FAIL: dataset did not pass validation.")
        raise SystemExit(1)

    print("PASS: dataset passed validation.")


if __name__ == "__main__":
    main()