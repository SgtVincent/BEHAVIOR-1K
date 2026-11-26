"""Local loopback rule service for the frozen a28 external-S2 boundary.

This is the CPU/network stand-in for the privileged S2 decision boundary: a
deterministic, schema-valid rule selector speaking the OpenAI-compatible
``/v1/chat/completions`` shape captured in the a28 frozen exchanges. It is
**not** an LLM, not a learned policy, and it never touches simulator state:

- it reads only the observation attached to the last user message;
- it chooses one bounded ``execute_primitive`` argument set that passes the
  frozen :mod:`contract.validate` checks against that same observation;
- it cannot repair an initial state, teleport an object, or fill in missing
  scene facts; an unselectable observation surfaces an explicit rejection.

The service binds 127.0.0.1 on an ephemeral port by default. For a formal
canary it is started as a separate, versioned, sha-gated process; this module
only provides the transport and the frozen response shape.

Sim-free: stdlib only. No simulator, torch or cvxpy import.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

from ..config import FROZEN_ALLOWED_PRIMITIVES, FROZEN_PRIMITIVE_CAPS


class RuleSelectionError(ValueError):
    """The observation does not admit a schema-valid bounded primitive."""


def _hand(obs: Dict[str, Any], arm: str) -> Dict[str, Any]:
    return obs.get("robot_state", {}).get("arms", {}).get(arm, {}) or {}


def _clamp_steps(meta: Dict[str, Any], cap: int) -> int:
    lo = int(meta.get("min_steps", 1))
    hi = min(int(meta.get("max_steps", cap)), int(cap))
    default = meta.get("default_max_steps", hi)
    value = int(default) if isinstance(default, int) else hi
    return max(lo, min(hi, value))


def select_rule(observation: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return ``(rule_id, args)`` for the observation; args pass contract.validate.

    The selector is deliberately conservative: a hand that already holds a
    live object tracks it nominally; otherwise it walks the frozen primitive
    order and picks the first schema-valid, non-null target. It never invents
    an object, a stop condition or a grasp outcome.
    """
    schema = observation.get("allowed_primitives_schema")
    if not isinstance(schema, dict) or not schema:
        raise RuleSelectionError("missing_live_primitive_schema")

    allowed = set(cfg.get("allowed_primitives") or FROZEN_ALLOWED_PRIMITIVES)
    caps: Dict[str, int] = dict(cfg.get("primitive_caps") or FROZEN_PRIMITIVE_CAPS)

    # Nominal tracking: an arm already holding an acknowledged live object.
    for arm in ("left", "right"):
        hand = _hand(observation, arm)
        if hand.get("valid_sdk_hold") is True and hand.get("object_in_hand") is not None:
            held = hand["object_in_hand"]
            if "continue_nominal_tracking" in allowed and held in (
                schema.get("continue_nominal_tracking", {}).get("allowed_objects") or []
            ):
                meta = schema["continue_nominal_tracking"]
                stop = (meta.get("allowed_stop_conditions") or [None])[0]
                args: Dict[str, Any] = {
                    "primitive": "continue_nominal_tracking",
                    "target_object": held,
                    "max_steps": _clamp_steps(meta, caps["continue_nominal_tracking"]),
                }
                if stop is not None:
                    args["stop_condition"] = stop
                return f"held_nominal_tracking:{arm}", args

    # Walk the frozen primitive order; first schema-valid non-null target.
    for primitive in FROZEN_ALLOWED_PRIMITIVES:
        if primitive not in allowed or primitive not in schema:
            continue
        meta = schema[primitive]
        if not isinstance(meta, dict):
            continue
        objects = [o for o in (meta.get("allowed_objects") or []) if o is not None]
        if not objects:
            continue
        target = objects[0]
        cap = caps.get(primitive, int(meta.get("max_steps", 1)))
        args = {
            "primitive": primitive,
            "target_object": target,
            "max_steps": _clamp_steps(meta, cap),
        }
        if primitive == "continue_nominal_tracking":
            stop_conditions = meta.get("allowed_stop_conditions") or []
            if not stop_conditions:
                continue
            args["stop_condition"] = stop_conditions[0]
        if primitive == "stabilize_and_brake":
            args["hold_steps"] = 1
        return f"rule_{primitive}:{target}", args

    raise RuleSelectionError("no_schema_valid_primitive_target")


def _observation_from_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    for message in reversed(messages or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuleSelectionError("user_message_not_json") from exc
        observation = payload.get("observation")
        if isinstance(observation, dict):
            return observation
        raise RuleSelectionError("observation_missing_from_user_message")
    raise RuleSelectionError("no_user_message")


class RuleS2Service:
    """Loopback ChatCompletions rule service. Stdlib HTTP, no simulator state."""

    def __init__(self, *, model_name: str, cfg: Dict[str, Any],
                 max_request_bytes: int = 4_194_304) -> None:
        self.model_name = model_name
        self.cfg = cfg
        self.max_request_bytes = int(max_request_bytes)
        self.requests: List[Dict[str, Any]] = []
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # -- selection / response ---------------------------------------------

    def decide(self, observation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        return select_rule(observation, self.cfg)

    def _response(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        rule_id, args = self.decide(observation)
        call_id = "call_" + uuid.uuid4().hex[:24]
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"decision": "execute_primitive", "rule_id": rule_id}),
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "execute_primitive",
                            "arguments": json.dumps(args, ensure_ascii=False, allow_nan=False),
                        },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "x_fake": {"rule_id": rule_id, "args": args},
        }

    # -- transport ----------------------------------------------------------

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def _send(self, status: int, body: Dict[str, Any]) -> None:
                encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/").endswith("/healthz"):
                    self._send(200, {"ok": True, "model": outer.model_name})
                else:
                    self._send(404, {"error": "not_found"})

            def do_POST(self) -> None:  # noqa: N802
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._send(404, {"error": "not_found"})
                    return
                length = int(self.headers.get("Content-Length", 0) or 0)
                if length > outer.max_request_bytes:
                    self._send(413, {"error": "request_too_large"})
                    return
                raw = self.rfile.read(length)
                try:
                    body = json.loads(raw.decode("utf-8"))
                    observation = _observation_from_messages(body.get("messages") or [])
                    outer.requests.append({"model": body.get("model"), "observation": observation})
                    response = outer._response(observation)
                except (RuleSelectionError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    self._send(400, {"error": type(exc).__name__, "detail": str(exc)})
                    return
                self._send(200, response)

        return _Handler

    def start(self, host: str = "127.0.0.1", port: int = 0) -> Tuple[str, int]:
        if self._httpd is not None:
            raise RuntimeError("service_already_started")
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        actual_host, actual_port = self._httpd.server_address[:2]
        return actual_host, actual_port

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def base_url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("service_not_started")
        return f"http://{self._httpd.server_address[0]}:{self._httpd.server_address[1]}/v1"
