"""Authentication-gated lifecycle; only a matching completion closes a session."""

from pathlib import Path

from medication_contracts import (
    AuthenticationResult,
    CompletionRecord,
    ScheduledOccurrence,
    SessionClosure,
    SessionContext,
    SessionKey,
    SessionResultView,
    TrackingUpdate,
)
from medication_contracts.records import append_json, new_id, utc_now, write_json


class SessionManager:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.session = None
        self.completion = None

    def _persist(self):
        append_json(self.directory / "session-history.jsonl", self.session)
        write_json(self.directory / "session.json", self.session, immutable=False)
        write_json(self.directory / "session-view.json", self.view(), immutable=False)

    def start(self, occurrence: ScheduledOccurrence, auth: AuthenticationResult):
        if self.session is not None:
            if self.session.auth_id == auth.auth_id:
                if self.auth != auth or self.occurrence != occurrence:
                    raise ValueError("Conflicting duplicate authentication")
                return self.session.model_copy(deep=True)
            raise ValueError("A session already exists")
        if auth.auth_status != "SUCCEEDED" or auth.user_id != occurrence.user_id:
            raise ValueError("Successful authentication of the scheduled user is required")
        if auth.scheduled_occurrence_id != occurrence.scheduled_occurrence_id:
            raise ValueError("Authentication occurrence mismatch")
        self.auth, self.occurrence = auth.model_copy(deep=True), occurrence.model_copy(deep=True)
        self.session = SessionContext(
            schema_version="1.0", context=SessionKey(
                session_id=new_id("session"), user_id=auth.user_id,
                scheduled_occurrence_id=occurrence.scheduled_occurrence_id,
                stream_id=auth.stream_id, target_track_id=auth.target_track_id),
            session_revision=1, auth_id=auth.auth_id, scheduled_at=occurrence.scheduled_at,
            started_at=max(utc_now(), auth.processed_at), source_time_origin_ms=auth.frame.source_ms,
            session_status="ACTIVE", tracking=None, completion_id=None, closed_at=None)
        write_json(self.directory / "authentication.json", auth)
        write_json(self.directory / "occurrence.json", occurrence)
        self._persist()
        return self.session.model_copy(deep=True)

    def update(self, tracking: TrackingUpdate):
        session = self.session
        if tracking.context != session.context:
            raise ValueError("Tracking context mismatch")
        frame = tracking.frame
        if frame.stream_id != session.context.stream_id or (
            frame.session_ms != frame.source_ms - session.source_time_origin_ms
        ):
            raise ValueError("Tracking time/stream mismatch")
        previous = session.tracking
        if previous is not None and frame.frame_index <= previous.frame.frame_index:
            if frame.frame_index == previous.frame.frame_index and tracking != previous:
                raise ValueError("Conflicting duplicate tracking update")
            return session.model_copy(deep=True)
        if previous is not None and frame.source_ms <= previous.frame.source_ms:
            raise ValueError("Tracking source timestamps must increase")
        if session.session_status == "CLOSED":
            return session.model_copy(deep=True)
        self.session = session.model_copy(update={"tracking": tracking.model_copy(deep=True),
                                                "session_revision": session.session_revision + 1})
        self._persist()
        return self.session.model_copy(deep=True)

    def close(self, completion: CompletionRecord):
        completion = CompletionRecord.model_validate_json(completion.model_dump_json())
        if completion.context != self.session.context:
            raise ValueError("Completion target/session mismatch")
        if self.completion is not None:
            if completion != self.completion:
                raise ValueError("Conflicting completion for session")
            if self.session.session_status == "CLOSED":
                return self.session.model_copy(deep=True)
        self.completion = completion.model_copy(deep=True)
        closed_at = max(utc_now(), completion.decided_at, self.session.started_at)
        closure_path = self.directory / "closure.json"
        if closure_path.exists():
            closure = SessionClosure.model_validate_json(closure_path.read_text(encoding="utf-8"))
            if (closure.session_id != self.session.context.session_id
                    or closure.completion_id != completion.completion_id
                    or closure.closed_at < completion.decided_at):
                raise ValueError("Persisted closure does not match completion")
            closed_at = closure.closed_at
        self.session = self.session.model_copy(update={
            "session_status": "CLOSED", "closed_at": closed_at,
            "completion_id": completion.completion_id,
            "session_revision": self.session.session_revision + 1})
        write_json(self.directory / "completion.json", completion)
        write_json(self.directory / "closure.json", SessionClosure(
            schema_version="1.0", session_id=self.session.context.session_id,
            completion_id=completion.completion_id, session_status="CLOSED", closed_at=closed_at))
        self._persist()
        return self.session.model_copy(deep=True)

    def view(self):
        s, c = self.session, self.completion
        return SessionResultView(
            schema_version="1.0", session_id=s.context.session_id, user_id=s.context.user_id,
            scheduled_occurrence_id=s.context.scheduled_occurrence_id,
            session_status=s.session_status, session_result="COMPLETE" if c else None,
            started_at=s.started_at, decided_at=c.decided_at if c else None,
            closed_at=s.closed_at, completion_id=s.completion_id)
