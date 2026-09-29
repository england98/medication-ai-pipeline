import pytest
from authentication_tracking import TargetTracker, operator_authenticate
from authentication_tracking.tracking import select_person
from conftest import frame
from medication_contracts import AuthenticationResult
from session_video import SessionManager


def test_operator_auth_requires_explicit_binding(config, session):
    with pytest.raises(ValueError, match="operator_confirmed"):
        operator_authenticate(config.occurrence, frame(session.context, 0, 0), acknowledged=False)


def test_session_gated_by_matching_auth(tmp_path, config, session):
    auth = operator_authenticate(config.occurrence, frame(session.context, 0, 900), acknowledged=True)
    manager = SessionManager(tmp_path)
    started = manager.start(config.occurrence, auth)
    assert started.source_time_origin_ms == 900
    assert manager.start(config.occurrence, auth) == started
    assert manager.view().session_result is None
    with pytest.raises(ValueError):
        SessionManager(tmp_path / "other").start(config.occurrence, auth.model_copy(update={"user_id": "wrong"}))


def test_no_silent_target_switch_after_disappearance(session):
    tracker = TargetTracker([0.1, 0.1, 0.6, 0.9])
    assert tracker.update(session.context, frame(session.context, 0, 0), [[0.1, 0.1, 0.6, 0.9]]).tracking_status == "TRACKED"
    assert tracker.update(session.context, frame(session.context, 1, 100), []).tracking_status == "UNRESOLVED"
    assert tracker.update(session.context, frame(session.context, 2, 200), [[0.1, 0.1, 0.6, 0.9]]).tracking_status == "UNRESOLVED"


def test_ambiguous_people_are_unresolved(session):
    tracker = TargetTracker([0.1, 0.1, 0.6, 0.9])
    result = tracker.update(session.context, frame(session.context, 0, 0), [[0.1, 0.1, 0.6, 0.9], [0.2, 0.1, 0.7, 0.9]])
    assert result.tracking_status == "UNRESOLVED"
    with pytest.raises(ValueError):
        select_person([[0.1, 0.1, 0.6, 0.9], [0.7, 0.1, 0.9, 0.9]])


def test_failed_auth_cannot_carry_identity():
    with pytest.raises(ValueError):
        AuthenticationResult(schema_version="1.0", auth_id="a", scheduled_occurrence_id="o",
            auth_status="FAILED", user_id="u", stream_id=None, target_track_id=None, frame=None,
            processed_at="2026-09-27T00:00:00.000Z", reason="mismatch")
