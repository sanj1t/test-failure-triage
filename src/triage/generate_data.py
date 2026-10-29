"""Generate synthetic engineering test-failure records.

This dataset is for learning and pipeline development.
Its labels are assigned by scenario design, not verified diagnosis.
"""

import argparse
import csv
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from triage.labels import FailureCategory


FIELDNAMES = [
    "incident_id",
    "scenario_family",
    "timestamp",
    "board",
    "subsystem",
    "test_name",
    "error_message",
    "duration_ms",
    "retry_count",
    "failure_category",
]

BOARDS = ("board_a", "board_b", "board_c")
SUBSYSTEMS = ("uart", "spi", "i2c", "can")

TEST_NAMES = (
    "initialization",
    "data_transfer",
    "recovery",
    "repeated_operation",
)

CONTEXTS = (
    "during startup",
    "after a reset",
    "during repeated execution",
    "after a recovery attempt",
)

START_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Scenario:
    family: str
    category: FailureCategory
    messages: tuple[str, ...]


SCENARIOS = (
    Scenario(
        family="cfg_parameter_mismatch",
        category=FailureCategory.CONFIGURATION,
        messages=(
            "Observed value differs from the expected setting",
            "Parameter readback does not match the requested value",
        ),
    ),
    Scenario(
        family="cfg_profile_rejected",
        category=FailureCategory.CONFIGURATION,
        messages=(
            "Selected operating profile was rejected",
            "Requested operating mode is not accepted",
        ),
    ),
    Scenario(
        family="cfg_initial_state",
        category=FailureCategory.CONFIGURATION,
        messages=(
            "Initial state differs from the expected setup",
            "Startup check found an unexpected device state",
        ),
    ),
    Scenario(
        family="cfg_mapping_conflict",
        category=FailureCategory.CONFIGURATION,
        messages=(
            "Requested channel mapping conflicts with the active setup",
            "Assigned interface mapping is inconsistent with the test setup",
        ),
    ),
    Scenario(
        family="time_completion_wait",
        category=FailureCategory.TIMEOUT,
        messages=(
            "Completion event was not observed within the allowed interval",
            "Operation remained pending beyond the expected interval",
        ),
    ),
    Scenario(
        family="time_startup_wait",
        category=FailureCategory.TIMEOUT,
        messages=(
            "Device readiness was not observed before the deadline",
            "Startup did not reach the ready state in the permitted interval",
        ),
    ),
    Scenario(
        family="time_recovery_wait",
        category=FailureCategory.TIMEOUT,
        messages=(
            "Recovery sequence did not finish before the deadline",
            "Recovery remained active after the permitted wait",
        ),
    ),
    Scenario(
        family="time_callback_wait",
        category=FailureCategory.TIMEOUT,
        messages=(
            "Expected callback did not arrive within the allowed interval",
            "No completion notification was received before the deadline",
        ),
    ),
    Scenario(
        family="comm_partial_transfer",
        category=FailureCategory.COMMUNICATION,
        messages=(
            "Transfer stopped before the expected byte count",
            "Received payload is shorter than the requested transfer",
        ),
    ),
    Scenario(
        family="comm_integrity_check",
        category=FailureCategory.COMMUNICATION,
        messages=(
            "Received data failed the integrity check",
            "Payload content differs from the transmitted sequence",
        ),
    ),
    Scenario(
        family="comm_peer_response",
        category=FailureCategory.COMMUNICATION,
        messages=(
            "Peer response could not be decoded",
            "Received response does not match the expected frame structure",
        ),
    ),
    Scenario(
        family="comm_link_interruption",
        category=FailureCategory.COMMUNICATION,
        messages=(
            "Connection was interrupted before the exchange completed",
            "Data exchange stopped after the link became unavailable",
        ),
    ),
    Scenario(
        family="res_buffer_unavailable",
        category=FailureCategory.RESOURCE_EXHAUSTION,
        messages=(
            "No free buffer was available for the next operation",
            "Buffer request could not be satisfied",
        ),
    ),
    Scenario(
        family="res_queue_capacity",
        category=FailureCategory.RESOURCE_EXHAUSTION,
        messages=(
            "Pending work could not be added because the queue was full",
            "Work submission exceeded the available queue capacity",
        ),
    ),
    Scenario(
        family="res_allocation_failure",
        category=FailureCategory.RESOURCE_EXHAUSTION,
        messages=(
            "Required allocation could not be completed",
            "Operation could not reserve the requested memory",
        ),
    ),
    Scenario(
        family="res_handle_limit",
        category=FailureCategory.RESOURCE_EXHAUSTION,
        messages=(
            "No additional operation handle was available",
            "Request could not obtain a free descriptor",
        ),
    ),
)


def generate_records(
    samples_per_family: int = 30,
    seed: int = 42,
) -> list[dict[str, str | int]]:
    """Generate records without reading or writing files."""

    if samples_per_family < 1:
        raise ValueError("samples_per_family must be at least 1")

    rng = random.Random(seed)
    records = []

    for scenario in SCENARIOS:
        for _ in range(samples_per_family):
            board = rng.choice(BOARDS)
            subsystem = rng.choice(SUBSYSTEMS)
            test_name = rng.choice(TEST_NAMES)
            symptom = rng.choice(scenario.messages)
            context = rng.choice(CONTEXTS)

            record = {
                "scenario_family": scenario.family,
                "board": board,
                "subsystem": subsystem,
                "test_name": test_name,
                "error_message": f"{subsystem}: {symptom} {context}.",
                "duration_ms": rng.randint(50, 5000),
                "retry_count": rng.randint(0, 3),
                "failure_category": scenario.category.value,
            }

            records.append(record)

    rng.shuffle(records)

    for index, record in enumerate(records, start=1):
        timestamp = START_TIME + timedelta(minutes=index - 1)

        record["incident_id"] = f"INC-{index:06d}"
        record["timestamp"] = timestamp.isoformat()

    return records


def write_records(
    records: list[dict[str, str | int]],
    output_path: Path,
) -> None:
    """Write records to CSV, replacing an existing output file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic test-failure dataset."
    )

    parser.add_argument(
        "--samples-per-family",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/test_failures.csv"),
    )

    args = parser.parse_args()

    if args.samples_per_family < 1:
        parser.error("--samples-per-family must be at least 1")

    records = generate_records(
        samples_per_family=args.samples_per_family,
        seed=args.seed,
    )

    write_records(records, args.output)

    counts = Counter(
        record["failure_category"]
        for record in records
    )

    print(f"Wrote {len(records)} records to {args.output}")

    for category, count in sorted(counts.items()):
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()