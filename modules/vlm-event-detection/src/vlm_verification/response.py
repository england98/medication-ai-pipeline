"""Parse model observations and derive event results without repairing responses."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .contracts import CHECKS_BY_EVENT, ModelResponse


class ResponseError(ValueError):
    def __init__(self, stage: str, code: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.code = code
        self.message = message


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_checks(response: ModelResponse, candidate_type: str) -> None:
    if candidate_type not in CHECKS_BY_EVENT:
        raise ResponseError("VALIDATION", "INVALID_VALUE", f"unsupported candidate type: {candidate_type}")
    expected = set(CHECKS_BY_EVENT[candidate_type])
    actual = set(response.checks)
    if expected - actual:
        raise ResponseError("VALIDATION", "MISSING_FIELD", f"missing checks: {sorted(expected - actual)}")
    if actual - expected:
        raise ResponseError("VALIDATION", "INVALID_VALUE", f"unexpected checks: {sorted(actual - expected)}")
    first, second = CHECKS_BY_EVENT[candidate_type]
    if candidate_type in {"E01", "E03"}:
        if response.checks[first] == "NO" and response.checks[second] == "YES":
            raise ResponseError(
                "VALIDATION", "CONTRADICTORY_RESPONSE",
                f"{second}=YES contradicts {first}=NO for the same object/action",
            )


def parse_response(raw_output: str, candidate_type: str, input_count: int) -> ModelResponse:
    """Require one complete JSON object and evidence for every required check.

    Markdown wrappers, duplicate keys, missing values, and extra fields are
    rejected; a normal UNKNOWN answer is retained unchanged.
    """
    if type(input_count) is not int or input_count < 1:
        raise ResponseError("VALIDATION", "EVIDENCE_MISMATCH", "actual input_count must be a positive integer")
    if not isinstance(raw_output, str):
        raise ResponseError("PARSE", "INVALID_RESPONSE", "model response must be a JSON string")
    try:
        payload = json.loads(raw_output, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise ResponseError("PARSE", "INVALID_RESPONSE", f"invalid response JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResponseError("PARSE", "INVALID_RESPONSE", "model response must be a JSON object")
    try:
        response = ModelResponse.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors(include_url=False, include_input=False)
        code = "MISSING_FIELD" if any(error["type"] == "missing" for error in errors) else "INVALID_VALUE"
        raise ResponseError("VALIDATION", code, f"invalid model response fields: {errors}") from exc
    _validate_checks(response, candidate_type)
    supported: set[str] = set()
    for item in response.evidence:
        if item.check not in response.checks:
            raise ResponseError("VALIDATION", "EVIDENCE_MISMATCH", f"evidence names an unknown check: {item.check}")
        if any(index >= input_count for index in item.input_indices):
            raise ResponseError("VALIDATION", "EVIDENCE_MISMATCH", f"evidence indices exceed actual input range 0..{input_count - 1}")
        supported.add(item.check)
    missing = set(response.checks) - supported
    if missing:
        raise ResponseError("VALIDATION", "EVIDENCE_MISMATCH", f"checks have no input-grounded evidence: {sorted(missing)}")
    return response


def derive_verification(response: ModelResponse, candidate_type: str) -> str:
    """Apply section 2-4 rules after structural and evidence validation."""
    _validate_checks(response, candidate_type)
    first, second = CHECKS_BY_EVENT[candidate_type]
    values = [response.checks[first]]
    if candidate_type != "E01":
        values.append(response.checks[second])
    if all(value == "YES" for value in values):
        return "CONFIRMED"
    if "NO" in values:
        return "REFUTED"
    return "UNCERTAIN"
