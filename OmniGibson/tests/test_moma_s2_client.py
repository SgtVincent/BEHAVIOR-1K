"""CPU tests for the ChatCompletions S2 client transport and response validation.

Sim-free/network-free: the HTTP opener is a fake; no omnigibson/torch import.
"""
import json

import pytest

from omnigibson.moma.s2.client import ChatCompletionsClient

CANS = ["can_of_soda_114", "can_of_soda_115", "can_of_soda_113"]
BUCKET = "trash_can_116"

CFG = {
    "can_names": CANS,
    "bucket_name": BUCKET,
    "allowed_primitives": ["approach_object", "stabilize_and_brake"],
    "primitive_caps": {"approach_object": 300, "stabilize_and_brake": 1800},
}


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self, n: int = -1) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


class FakeOpener:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self._status = status
        self.requests: list = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        return FakeResponse(self._body, self._status)


def _obs():
    return {
        "runner_state": "decision_required",
        "decision_required": True,
        "active_request": None,
        "robot_state": {"arms": {
            "left": {"object_in_hand": None, "constraint_present": False, "release_counter": None, "valid_sdk_hold": False},
            "right": {"object_in_hand": None, "constraint_present": False, "release_counter": None, "valid_sdk_hold": False},
        }},
        "allowed_primitives_schema": {
            "approach_object": {"max_steps": 300, "min_steps": 1, "allowed_objects": CANS, "allowed_stop_conditions": None},
        },
        "task_progress": {},
        "objects": {},
    }


def _response(arguments: str, model: str = "fake-rule", tool_name: str = "execute_primitive"):
    return json.dumps({
        "model": model,
        "choices": [{"message": {"tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": tool_name, "arguments": arguments},
        }]}}],
        "usage": {"total_tokens": 7},
    }).encode()


def _client(opener: FakeOpener, **kwargs):
    return ChatCompletionsClient(
        base_url="http://127.0.0.1:8765",
        model="fake-rule",
        prompt_text="system rule",
        cfg=CFG,
        opener=opener,
        **kwargs,
    )


def test_request_body_keeps_completions_semantics():
    opener = FakeOpener(_response(json.dumps(
        {"primitive": "approach_object", "target_object": CANS[0], "max_steps": 120}
    )))
    client = _client(opener)
    decision = client.decide(_obs(), [])

    request = opener.requests[0]
    assert request.full_url.endswith("/chat/completions")
    body = json.loads(request.data.decode())
    assert body["model"] == "fake-rule"
    assert body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is False
    assert body["messages"][0] == {"role": "system", "content": "system rule"}
    assert body["messages"][-1]["role"] == "user"
    assert body["tools"][0]["function"]["name"] == "execute_primitive"
    assert decision["args"]["primitive"] == "approach_object"
    assert len(decision["request_sha256"]) == 64
    assert decision["usage"] == {"total_tokens": 7}


def test_rejects_model_identity_mismatch():
    opener = FakeOpener(_response("{}", model="other-model"))
    client = _client(opener)
    with pytest.raises(ValueError, match="model_response_identity_mismatch"):
        client.decide(_obs(), [])


def test_rejects_non_200_status():
    opener = FakeOpener(_response("{}"), status=500)
    client = _client(opener)
    with pytest.raises(ValueError, match="model_http_status_500"):
        client.decide(_obs(), [])


def test_rejects_choice_count():
    body = json.dumps({"model": "fake-rule", "choices": []}).encode()
    opener = FakeOpener(body)
    client = _client(opener)
    with pytest.raises(ValueError, match="model_choice_count_invalid"):
        client.decide(_obs(), [])


def test_rejects_multiple_tool_calls():
    body = json.dumps({"model": "fake-rule", "choices": [{"message": {"tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "execute_primitive", "arguments": "{}"}},
        {"id": "b", "type": "function", "function": {"name": "execute_primitive", "arguments": "{}"}},
    ]}}]}).encode()
    opener = FakeOpener(body)
    client = _client(opener)
    with pytest.raises(ValueError, match="exactly_one_tool_call_required"):
        client.decide(_obs(), [])


def test_rejects_wrong_tool_name():
    opener = FakeOpener(_response("{}", tool_name="teleport_object"))
    client = _client(opener)
    with pytest.raises(ValueError, match="unexpected_model_tool"):
        client.decide(_obs(), [])


def test_rejects_duplicate_key_arguments():
    bad = '{"primitive": "approach_object", "primitive": "approach_object"}'
    opener = FakeOpener(_response(bad))
    client = _client(opener)
    with pytest.raises(ValueError, match="duplicate_json_key"):
        client.decide(_obs(), [])


def test_rejects_invalid_tool_arguments():
    bad_args = json.dumps({"primitive": "approach_object", "target_object": BUCKET, "max_steps": 120})
    opener = FakeOpener(_response(bad_args))
    client = _client(opener)
    with pytest.raises(ValueError, match="object_not_allowed"):
        client.decide(_obs(), [])


@pytest.mark.parametrize("url", [
    "https://user:pass@host/x",
    "http://host/path?query=1",
    "http://host/#frag",
    "ftp://host/x",
    "://nohost",
])
def test_rejects_invalid_base_url(url):
    with pytest.raises(ValueError, match="invalid_model_base_url"):
        ChatCompletionsClient(base_url=url, model="m", prompt_text="p", cfg=CFG, opener=FakeOpener(b"{}"))
