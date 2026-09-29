import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from PIL import Image

from vlm_verification.backend import QwenBackend
from vlm_verification.config import ExecutionConfig
from vlm_verification.contracts import VerificationRequest
from vlm_verification.media import LoadedImage
from vlm_verification.prompts import QUESTIONS, build_messages


def make_request(event_type="E02"):
    context = dict(
        session_id="session",
        user_id="user",
        scheduled_occurrence_id="occurrence",
        stream_id="stream",
        target_track_id="target",
    )
    candidate = dict(candidate_id="candidate", candidate_type=event_type, candidate_revision=1)
    frame = dict(
        stream_id="stream", frame_index=3, source_ms=1100, session_ms=100, width=640, height=480
    )
    unavailable = dict(status="UNAVAILABLE", items=[], reason="No detector provided")
    resource = dict(resource_id="config", locator="config.json", media_type="application/json")
    return VerificationRequest.model_validate(
        {
            "schema_version": "1.0",
            "request_id": "request",
            "context": context,
            "event": {
                "context": context,
                "candidate": candidate,
                "created_at": "2026-09-01T00:00:00.000Z",
                "updated_at": "2026-09-01T00:00:00.000Z",
                "action_range": {"start_ms": 100, "end_ms": 100},
                "object_track_ids": [],
                "regions": [
                    {
                        "frame": frame,
                        "region_type": "TARGET",
                        "bbox": [0.2, 0.1, 0.8, 0.9],
                        "object_track_id": None,
                    }
                ],
            },
            "detection": {
                "context": context,
                "candidate": candidate,
                "detector_config": resource,
                "samples": [
                    {
                        "frame": frame,
                        "objects": unavailable,
                        "landmarks": unavailable,
                        "motions": unavailable,
                    }
                ],
            },
            "media": {
                "context": context,
                "candidate": candidate,
                "requested_range": {"start_ms": 100, "end_ms": 100},
                "clip": {
                    "clip_id": "clip",
                    "video": {
                        "resource_id": "video",
                        "locator": "clip.mp4",
                        "media_type": "video/mp4",
                    },
                    "actual_range": {"start_ms": 100, "end_ms": 100},
                    "frames": [{"clip_frame_index": 0, "clip_ms": 0, "source": frame}],
                    "rois": [],
                },
                "unavailable_reason": None,
            },
            "created_at": "2026-09-01T00:00:00.000Z",
            "model_id": "Qwen/Qwen3-VL-2B-Instruct",
            "model_revision": "a" * 40,
            "prompt_version": "vlm-events-v1",
            "execution_config": resource,
        }
    )


def images_for(request):
    return [
        LoadedImage(
            Image.new("RGB", (640, 480)), request.media.clip.frames[0], None, [0.0, 0.0, 1.0, 1.0]
        )
    ]


@pytest.mark.parametrize("event_type", ["E01", "E02", "E03"])
def test_prompt_selects_event_questions_and_grounded_indices(event_type):
    request = make_request(event_type)
    messages = build_messages(request, images_for(request))
    texts = [
        item["text"]
        for message in messages
        for item in message["content"]
        if item["type"] == "text"
    ]
    full_text = "\n".join(texts)
    for check in QUESTIONS[event_type]:
        assert check in full_text
    for other_type, questions in QUESTIONS.items():
        if other_type != event_type:
            assert not any(check in full_text for check in questions)
    metadata = json.loads(texts[-1].removeprefix("Input image metadata: "))
    assert metadata["input_index"] == 0
    assert metadata["source"]["source_ms"] == 1100
    assert metadata["candidate_regions_reference_only"][0]["region_type"] == "TARGET"
    assert "UNKNOWN" in full_text
    assert "same" in full_text.lower()
    assert "request_id" not in full_text
    assert "user_id" not in full_text
    assert "CONFIRMED" not in full_text


def test_prepare_records_processor_observed_dimensions_and_real_template(monkeypatch):
    request = make_request()
    loaded = images_for(request)
    config = ExecutionConfig(model_revision="a" * 40)
    calls = {}

    class Processor:
        image_processor = SimpleNamespace(patch_size=16)
        tokenizer = SimpleNamespace(decode=lambda *a, **k: "native-tokenized-prompt")

        def apply_chat_template(self, messages, **kwargs):
            calls["template"] = (messages, kwargs)
            return "native-template"

        def __call__(self, **kwargs):
            calls["processor"] = kwargs
            return {"image_grid_thw": [[1, 20, 28]], "input_ids": [[7, 8]]}

    monkeypatch.setattr("vlm_verification.backend.load_clip_images", lambda *a, **k: loaded)
    backend = QwenBackend()
    monkeypatch.setattr(backend, "_load_processor", lambda _: Processor())
    prepared = backend.prepare(request, config)
    assert calls["template"][1] == {"tokenize": False, "add_generation_prompt": True}
    assert calls["processor"]["text"] == ["native-template"]
    assert calls["processor"]["images"] == [loaded[0].image]
    assert prepared.record.rendered_prompt == "native-tokenized-prompt"
    assert prepared.record.frames[0].resized_size == [448, 320]
    assert prepared.record.frames[0].input_size == [448, 320]
    assert prepared.record.frames[0].padding == [0, 0, 0, 0]
    assert prepared.record.frames[0].source_bbox == [0, 0, 1, 1]
    assert backend._model is None


def test_unknown_prompt_version_is_rejected():
    request = make_request()
    request.prompt_version = "unregistered"
    with pytest.raises(ValueError, match="prompt_version"):
        build_messages(request, images_for(request))


def test_generate_pins_revision_dtype_and_trims_prompt_tokens(monkeypatch):
    request = make_request()
    config = ExecutionConfig(model_revision="a" * 40)
    calls = {}

    class Tokens:
        shape = (1, 3)

        def __getitem__(self, key):
            if key == 0:
                return [7, 8, 9]
            assert key[0] == slice(None)
            calls["output_slice"] = key[1]
            return [[41, 42]]

    class Batch(dict):
        def to(self, device):
            calls["input_device"] = device
            return self

    class Processor:
        image_processor = SimpleNamespace(patch_size=16)
        tokenizer = SimpleNamespace(decode=lambda *a, **k: "native-tokenized-prompt")

        def apply_chat_template(self, *args, **kwargs):
            return "native-template"

        def __call__(self, **kwargs):
            return Batch(image_grid_thw=[[1, 20, 28]], input_ids=Tokens())

        def batch_decode(self, generated, **kwargs):
            assert generated == [[41, 42]]
            calls["decode"] = kwargs
            return ["raw model response"]

    class Model:
        device = "cpu"

        def to(self, device):
            calls["model_device"] = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            calls["generate"] = kwargs
            return Tokens()

    def from_pretrained(model_id, **kwargs):
        calls["load"] = (model_id, kwargs)
        return Model()

    dtype = object()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            float32=dtype,
            inference_mode=nullcontext,
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=from_pretrained),
        ),
    )
    monkeypatch.setattr(
        "vlm_verification.backend.load_clip_images", lambda *a, **k: images_for(request)
    )
    backend = QwenBackend()
    monkeypatch.setattr(backend, "_load_processor", lambda _: Processor())
    prepared = backend.prepare(request, config)
    assert backend.generate(prepared, config) == "raw model response"
    assert calls["load"][0] == "Qwen/Qwen3-VL-2B-Instruct"
    assert calls["load"][1]["revision"] == "a" * 40
    assert calls["load"][1]["dtype"] is dtype
    assert calls["load"][1]["trust_remote_code"] is False
    assert calls["generate"]["max_new_tokens"] == 768
    assert calls["generate"]["do_sample"] is False
    assert calls["output_slice"] == slice(3, None)
    assert calls["input_device"] == calls["model_device"] == "cpu"


def test_prepared_configuration_cannot_change_before_generate(monkeypatch):
    request = make_request()
    config = ExecutionConfig(model_revision="a" * 40)

    class Processor:
        image_processor = SimpleNamespace(patch_size=16)
        tokenizer = SimpleNamespace(decode=lambda *a, **k: "native-tokenized-prompt")

        def apply_chat_template(self, *args, **kwargs):
            return "template"

        def __call__(self, **kwargs):
            return {"image_grid_thw": [[1, 20, 28]], "input_ids": [[7, 8]]}

    monkeypatch.setattr(
        "vlm_verification.backend.load_clip_images", lambda *a, **k: images_for(request)
    )
    backend = QwenBackend()
    monkeypatch.setattr(backend, "_load_processor", lambda _: Processor())
    prepared = backend.prepare(request, config)
    with pytest.raises(ValueError, match="configuration changed"):
        backend.generate(prepared, config.model_copy(update={"max_new_tokens": 800}))
