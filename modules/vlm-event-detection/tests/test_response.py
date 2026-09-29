import itertools
import json

import pytest

from vlm_verification.contracts import CHECKS_BY_EVENT
from vlm_verification.response import ResponseError, derive_verification, parse_response


def response_payload(event="E02", first="YES", second="YES"):
    keys = CHECKS_BY_EVENT[event]
    return {
        "checks": dict(zip(keys, [first, second])),
        "object_description": "A small object is visible in the target's hand.",
        "observed_action": "The same object moves toward the mouth.",
        "evidence": [
            {"check": key, "input_indices": [0, 1], "description": "Visible in the supplied images."}
            for key in keys
        ],
    }


@pytest.mark.parametrize("event,first,second", list(itertools.product(
    ["E01", "E02", "E03"], ["YES", "NO", "UNKNOWN"], ["YES", "NO", "UNKNOWN"]
)))
def test_complete_event_truth_tables(event, first, second):
    payload = response_payload(event, first, second)
    if event in {"E01", "E03"} and first == "NO" and second == "YES":
        with pytest.raises(ResponseError) as caught:
            parse_response(json.dumps(payload), event, 2)
        assert (caught.value.stage, caught.value.code) == ("VALIDATION", "CONTRADICTORY_RESPONSE")
        return
    parsed = parse_response(json.dumps(payload), event, 2)
    values = [first] if event == "E01" else [first, second]
    expected = "CONFIRMED" if all(value == "YES" for value in values) else (
        "REFUTED" if "NO" in values else "UNCERTAIN"
    )
    assert derive_verification(parsed, event) == expected
    assert parsed.checks == payload["checks"]


@pytest.mark.parametrize("raw", ["", "not JSON", "{}{}", "```json\n{}\n```", "[]", "null", '{"checks":NaN}', '{"checks":{},"checks":{}}'])
def test_invalid_json_is_not_repaired(raw):
    with pytest.raises(ResponseError) as caught:
        parse_response(raw, "E02", 2)
    assert (caught.value.stage, caught.value.code) == ("PARSE", "INVALID_RESPONSE")


@pytest.mark.parametrize("field", ["checks", "object_description", "observed_action", "evidence"])
def test_missing_required_fields_are_errors_not_unknown(field):
    payload = response_payload()
    del payload[field]
    with pytest.raises(ResponseError) as caught:
        parse_response(json.dumps(payload), "E02", 2)
    assert caught.value.code == "MISSING_FIELD"


def test_missing_required_check_is_error():
    payload = response_payload()
    del payload["checks"]["object_is_medication"]
    with pytest.raises(ResponseError) as caught:
        parse_response(json.dumps(payload), "E02", 2)
    assert caught.value.code == "MISSING_FIELD"


@pytest.mark.parametrize("value", [True, False, 1, None, "yes", "", "UNCERTAIN"])
def test_check_values_are_exact(value):
    payload = response_payload()
    payload["checks"]["object_is_medication"] = value
    with pytest.raises(ResponseError) as caught:
        parse_response(json.dumps(payload), "E02", 2)
    assert caught.value.code == "INVALID_VALUE"


def test_other_event_check_is_not_allowed():
    payload = response_payload()
    payload["checks"]["handling_visible"] = "YES"
    with pytest.raises(ResponseError) as caught:
        parse_response(json.dumps(payload), "E02", 2)
    assert caught.value.code == "INVALID_VALUE"


def test_model_cannot_assign_event_verification_or_ids():
    payload = response_payload()
    payload["verification"] = "CONFIRMED"
    with pytest.raises(ResponseError) as caught:
        parse_response(json.dumps(payload), "E02", 2)
    assert caught.value.code == "INVALID_VALUE"


@pytest.mark.parametrize("evidence_change", ["empty", "missing_check", "wrong_check", "out_of_range"])
def test_each_check_requires_actual_input_evidence_even_unknown(evidence_change):
    payload = response_payload(first="UNKNOWN", second="UNKNOWN")
    if evidence_change == "empty":
        payload["evidence"] = []
    elif evidence_change == "missing_check":
        payload["evidence"] = payload["evidence"][:1]
    elif evidence_change == "wrong_check":
        payload["evidence"][0]["check"] = "handling_visible"
    else:
        payload["evidence"][0]["input_indices"] = [2]
    with pytest.raises(ResponseError) as caught:
        parse_response(json.dumps(payload), "E02", 2)
    assert caught.value.code == "EVIDENCE_MISMATCH"


@pytest.mark.parametrize("indices", [[], [-1], ["0"], [True], [0.0]])
def test_evidence_indices_are_nonnegative_integer_arrays(indices):
    payload = response_payload()
    payload["evidence"][0]["input_indices"] = indices
    with pytest.raises(ResponseError):
        parse_response(json.dumps(payload), "E02", 2)


@pytest.mark.parametrize("field", ["object_description", "observed_action"])
@pytest.mark.parametrize("value", ["", "   ", None, 12])
def test_observation_text_is_required_nonempty_string(field, value):
    payload = response_payload()
    payload[field] = value
    with pytest.raises(ResponseError) as caught:
        parse_response(json.dumps(payload), "E02", 2)
    assert caught.value.code == "INVALID_VALUE"
