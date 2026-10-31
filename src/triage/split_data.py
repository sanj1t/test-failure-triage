"""Split validated data by scenario family."""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from triage.generate_data import write_records
from triage.labels import VALID_CATEGORIES
from triage.validate_data import validate_dataset


PARTITIONS = ("train", "development", "test")


def split_records(
    records: list[dict],
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Keep each family entirely within one partition.

    For each category:
    - Reserve one family for development.
    - Reserve one family for test.
    - Assign the remaining families to training.

    Input records are expected to have passed dataset validation.
    """

    families_by_category = {
        category: set()
        for category in sorted(VALID_CATEGORIES)
    }

    family_categories = {}

    for record in records:
        category = record["failure_category"]
        family = record["scenario_family"]

        if category not in VALID_CATEGORIES:
            raise ValueError(f"Unsupported category: {category}")

        if (
            family in family_categories
            and family_categories[family] != category
        ):
            raise ValueError(
                f"Family '{family}' has multiple categories."
            )

        family_categories[family] = category
        families_by_category[category].add(family)

    rng = random.Random(seed)
    family_partition = {}

    for category in sorted(VALID_CATEGORIES):
        families = sorted(families_by_category[category])

        if len(families) < 3:
            raise ValueError(
                f"Category '{category}' requires at least "
                "three scenario families."
            )

        rng.shuffle(families)

        family_partition[families[0]] = "development"
        family_partition[families[1]] = "test"

        for family in families[2:]:
            family_partition[family] = "train"

    partitions = {
        name: []
        for name in PARTITIONS
    }

    for record in records:
        partition = family_partition[record["scenario_family"]]
        partitions[partition].append(record)

    return partitions


def summarize_partitions(
    partitions: dict[str, list[dict]],
) -> dict:
    """Describe partition sizes, categories, and families."""

    summary = {}

    for name in PARTITIONS:
        records = partitions[name]

        families = sorted({
            record["scenario_family"]
            for record in records
        })

        counts = Counter(
            record["failure_category"]
            for record in records
        )

        summary[name] = {
            "record_count": len(records),
            "family_count": len(families),
            "category_counts": dict(sorted(counts.items())),
            "scenario_families": families,
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split validated data by scenario family."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/test_failures.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    output_paths = [
        args.output_dir / f"{name}.csv"
        for name in PARTITIONS
    ]
    output_paths.append(args.output_dir / "split_manifest.json")

    if args.input.resolve() in {
        path.resolve()
        for path in output_paths
    }:
        parser.error("An output path would overwrite the input file.")

    try:
        validation = validate_dataset(args.input)

        if not validation["is_valid"]:
            parser.exit(
                status=1,
                message=(
                    "Dataset failed validation. Run "
                    "triage.validate_data for details.\n"
                ),
            )

        with args.input.open(
            encoding="utf-8",
            newline="",
        ) as csv_file:
            records = list(csv.DictReader(csv_file, strict=True))

        # Match the validator's whitespace handling.
        records = [
            {
                key: value.strip()
                for key, value in record.items()
            }
            for record in records
        ]

        partitions = split_records(records, seed=args.seed)
        summary = summarize_partitions(partitions)

        args.output_dir.mkdir(parents=True, exist_ok=True)

        for name in PARTITIONS:
            write_records(
                partitions[name],
                args.output_dir / f"{name}.csv",
            )

        manifest = {
            "input_path": str(args.input),
            "seed": args.seed,
            "strategy": (
                "Per category: one development family, "
                "one test family, remaining families for training."
            ),
            "partitions": summary,
        }

        manifest_path = args.output_dir / "split_manifest.json"

        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        parser.exit(
            status=2,
            message=f"Unable to split dataset: {error}\n",
        )

    for name in PARTITIONS:
        details = summary[name]

        print(
            f"{name}: "
            f"{details['record_count']} records, "
            f"{details['family_count']} families"
        )

    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()