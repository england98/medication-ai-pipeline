"""Persistence utilities for a single pipeline owner and immutable exchange records."""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

_LOG = logging.getLogger(__name__)
# Only Windows access/sharing/lock violations may be transient here.
_RETRYABLE_WINERRORS = {5, 32, 33}
_REPLACE_RETRY_DELAYS = (0.02, 0.04, 0.08, 0.16, 0.32, 0.32)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_id(kind: str) -> str:
    return f"{kind}-{uuid4().hex}"


def key_path(directory: Path, identifier: str) -> Path:
    return directory / (hashlib.sha256(identifier.encode()).hexdigest() + ".json")


def as_data(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _replacement_event(path, temporary, state, attempt, started, error, delay=None):
    """Best-effort append only: diagnostics must not mask the original failure."""
    event = {
        "schema_version": "1.0", "at": utc_now(), "operation": "os.replace", "state": state,
        "path": str(path.absolute()), "temporary_path": str(temporary.absolute()),
        "pid": os.getpid(), "attempt": attempt, "max_attempts": len(_REPLACE_RETRY_DELAYS) + 1,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
        "next_delay_ms": round(delay * 1000) if delay is not None else None,
        "winerror": getattr(error, "winerror", None), "errno": error.errno,
        "error": f"{type(error).__name__}: {error}",
    }
    try:
        append_json(path.parent / "persistence-events.jsonl", event)
    except OSError as exc:
        _LOG.warning("Could not append persistence diagnostics for %s: %s", path, exc)


def _replace_with_retry(temporary: Path, path: Path) -> None:
    started = time.monotonic()
    last_error = None
    for index in range(len(_REPLACE_RETRY_DELAYS) + 1):
        attempt = index + 1
        try:
            os.replace(temporary, path)
        except OSError as exc:
            retry = getattr(exc, "winerror", None) in _RETRYABLE_WINERRORS and index < len(_REPLACE_RETRY_DELAYS)
            if not retry:
                _replacement_event(path, temporary, "FAILED", attempt, started, exc)
                _LOG.error("JSON replacement failed after %s attempt(s); pending data retained: %s", attempt, temporary)
                raise
            last_error = exc
            delay = _REPLACE_RETRY_DELAYS[index]
            _replacement_event(path, temporary, "RETRY", attempt, started, exc, delay)
            if index == 0:
                _LOG.warning("Windows denied JSON replacement; retrying %s (WinError %s)", path, exc.winerror)
            time.sleep(delay)
        else:
            if last_error is not None:
                _replacement_event(path, temporary, "RECOVERED", attempt, started, last_error)
            return


def write_json(path: Path, value, *, immutable: bool = True) -> None:
    value = as_data(value)
    if immutable and path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Immutable record conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                         + "\n", encoding="utf-8")
    # Retry only this operation, never append a session-history row twice or
    # truncate/delete the last complete snapshot to bypass a reader's lock.
    _replace_with_retry(temporary, path)


def append_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(as_data(value), ensure_ascii=False, allow_nan=False) + "\n")
