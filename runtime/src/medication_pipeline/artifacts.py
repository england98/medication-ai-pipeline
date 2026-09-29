"""Allocate a new experiment directory without reusing earlier results."""

import re
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path

from medication_contracts.records import write_json

KST = timezone(timedelta(hours=9), "KST")


def validate_test_name(value: str) -> str:
    if not 1 <= len(value) <= 64 or re.fullmatch(r"\w[\w-]*", value) is None:
        raise ValueError("test_name must be 1-64 letters, digits, underscores or hyphens, starting with a letter, digit or underscore")
    return value


def create_test_directory(parent: Path, test_name: str) -> Path:
    test_name = validate_test_name(test_name)
    started_at = datetime.now(KST)
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    parent = Path(parent).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    for attempt in count(1):
        # Keep the timestamp last, including when two runs start in the same second.
        purpose = test_name if attempt == 1 else f"{test_name}-{attempt:02d}"
        directory = parent / f"{purpose}_{timestamp}"
        try:
            directory.mkdir()
        except FileExistsError:
            continue
        write_json(directory / "test-run.json", {
            "schema_version": "1.0", "test_name": test_name,
            "started_at": started_at.isoformat(timespec="milliseconds"),
            "timezone": "Asia/Seoul", "run_dir": str(directory),
        })
        return directory
