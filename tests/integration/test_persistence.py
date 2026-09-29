import errno
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from medication_contracts import records
from medication_pipeline.pipeline import run_pipeline
from test_pipeline import ScriptedPerception, TestBackend, make_video


def windows_error(code):
    error = PermissionError(errno.EACCES, "simulated Windows file conflict")
    error.winerror = code
    return error


def events(directory):
    return [json.loads(line) for line in (directory / "persistence-events.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("code", [5, 32, 33])
def test_transient_windows_conflict_keeps_old_snapshot_until_replaced(tmp_path, monkeypatch, code):
    target = tmp_path / "session.json"
    records.write_json(target, {"revision": 1})
    original_replace = records.os.replace
    attempts = []
    sleeps = []

    def replace(source, destination):
        attempts.append(source)
        assert json.loads(target.read_text()) == {"revision": 1}
        if len(attempts) <= 2:
            raise windows_error(code)
        original_replace(source, destination)

    monkeypatch.setattr(records.os, "replace", replace)
    monkeypatch.setattr(records.time, "sleep", sleeps.append)
    records.write_json(target, {"revision": 2}, immutable=False)
    assert json.loads(target.read_text()) == {"revision": 2}
    assert len(attempts) == 3 and len(set(attempts)) == 1
    assert sleeps == list(records._REPLACE_RETRY_DELAYS[:2])
    assert [row["state"] for row in events(tmp_path)] == ["RETRY", "RETRY", "RECOVERED"]
    assert not list(tmp_path.glob("*.tmp"))


def test_persistent_denial_stops_at_limit_and_retains_both_versions(tmp_path, monkeypatch):
    target = tmp_path / "session.json"
    records.write_json(target, {"revision": 1})
    failure = windows_error(5)
    attempts = []
    sleeps = []

    def replace(source, destination):
        attempts.append(source)
        raise failure

    monkeypatch.setattr(records.os, "replace", replace)
    monkeypatch.setattr(records.time, "sleep", sleeps.append)
    with pytest.raises(PermissionError) as caught:
        records.write_json(target, {"revision": 2}, immutable=False)
    assert caught.value is failure
    assert len(attempts) == len(records._REPLACE_RETRY_DELAYS) + 1
    assert sleeps == list(records._REPLACE_RETRY_DELAYS)
    assert json.loads(target.read_text()) == {"revision": 1}
    assert json.loads(attempts[-1].read_text()) == {"revision": 2}
    final = events(tmp_path)[-1]
    assert final["state"] == "FAILED" and final["winerror"] == 5
    assert final["attempt"] == len(attempts)
    assert Path(final["temporary_path"]).exists()


@pytest.mark.parametrize("diagnostics_fail", [False, True])
def test_other_io_errors_are_not_retried_or_masked(tmp_path, monkeypatch, diagnostics_fail):
    target = tmp_path / "session.json"
    records.write_json(target, {"revision": 1})
    failure = OSError(errno.ENOSPC, "simulated disk full")
    attempts = []
    sleeps = []

    def replace(source, destination):
        attempts.append(source)
        raise failure

    def append(*args):
        raise PermissionError("diagnostic log also locked")

    monkeypatch.setattr(records.os, "replace", replace)
    monkeypatch.setattr(records.time, "sleep", sleeps.append)
    if diagnostics_fail:
        monkeypatch.setattr(records, "append_json", append)
    with pytest.raises(OSError) as caught:
        records.write_json(target, {"revision": 2}, immutable=False)
    assert caught.value is failure
    assert len(attempts) == 1 and not sleeps
    assert json.loads(target.read_text()) == {"revision": 1}
    assert json.loads(attempts[0].read_text()) == {"revision": 2}


def test_immutable_records_still_reject_conflicting_content(tmp_path):
    target = tmp_path / "request.json"
    records.write_json(target, {"request_id": "one"})
    records.write_json(target, {"request_id": "one"})
    with pytest.raises(ValueError, match="Immutable record conflict"):
        records.write_json(target, {"request_id": "different"})
    assert json.loads(target.read_text()) == {"request_id": "one"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("filename", ["session.json", "session-view.json"])
def test_pipeline_continues_after_denial_without_duplicate_history(config, monkeypatch, filename):
    make_video(config.source)
    original_replace = records.os.replace
    conflicts = []

    def replace(source, destination):
        if destination.name == filename and destination.exists() and len(conflicts) < 2:
            conflicts.append(str(destination))
            raise windows_error(5)
        original_replace(source, destination)

    monkeypatch.setattr(records.os, "replace", replace)
    monkeypatch.setattr(records.time, "sleep", lambda delay: None)
    result = run_pipeline(config, perception=ScriptedPerception(), backend=TestBackend("UNKNOWN"))
    directory = Path(result["run_dir"])
    assert len(conflicts) == 2
    assert result["frames_analyzed"] == 22
    assert result["requests"] == result["results"] == 2
    assert result["processing_errors"] == 0
    history = [json.loads(line) for line in (directory / "session/session-history.jsonl").read_text().splitlines()]
    assert [row["session_revision"] for row in history] == list(range(1, 24))
    assert json.loads((directory / "session/session.json").read_text()) == history[-1]
    assert json.loads((directory / "session/session-view.json").read_text()) == result["session"]
    assert not list((directory / "session").glob("*.tmp"))
    assert [row["state"] for row in events(directory / "session")] == ["RETRY", "RETRY", "RECOVERED"]


@contextmanager
def deny_delete_handle(path):
    """Hold a real Windows reader that shares read/write, but not replacement."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wintypes.HANDLE]
    close.restype = wintypes.BOOL
    handle = create(str(path), 0x80000000, 0x1 | 0x2, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        close(handle)


@pytest.mark.skipif(os.name != "nt", reason="Requires actual Windows file-sharing semantics")
def test_actual_windows_reader_lock_recovers_when_handle_closes(tmp_path, monkeypatch):
    target = tmp_path / "session.json"
    records.write_json(target, {"revision": 1})
    blocked = threading.Event()
    errors = []
    original_event = records._replacement_event

    def event(*args, **kwargs):
        original_event(*args, **kwargs)
        if args[2] == "RETRY":
            blocked.set()

    def write():
        try:
            records.write_json(target, {"revision": 2}, immutable=False)
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(records, "_replacement_event", event)
    writer = threading.Thread(target=write, daemon=True)
    try:
        with deny_delete_handle(target):
            writer.start()
            assert blocked.wait(5), "Actual os.replace did not report a Windows lock conflict"
            assert json.loads(target.read_text()) == {"revision": 1}
    finally:
        if writer.ident is not None:
            writer.join(5)
    assert not writer.is_alive() and not errors
    assert json.loads(target.read_text()) == {"revision": 2}
    history = events(tmp_path)
    assert history[0]["winerror"] in {5, 32, 33}
    assert history[-1]["state"] == "RECOVERED"
