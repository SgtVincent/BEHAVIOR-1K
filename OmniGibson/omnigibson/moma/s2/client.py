"""OpenAI-compatible ChatCompletions caller for the privileged S2 boundary.

Ported from the a28 frozen baseline ``s2_runtime.py`` ``Model.decide`` contract:

- No ambient proxy, no redirects; http/https only, no embedded credentials,
  query or fragment.
- The user message carries the sanitized live observation JSON; the body
  carries the live tool schema, ``tool_choice=required``, one tool call only.
- Response identity, choice count, tool name and strict-JSON arguments are
  validated before the decision leaves the boundary.

This module is stdlib-only and does not import the simulator. The HTTP opener
is injectable so the validation path is CPU-testable with a fake transport.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contract import (
    capped_observation,
    digest,
    sanitized,
    strict_json,
    tool_schema,
    validate,
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> Any:
        raise ValueError("model_redirect_disallowed")


def _default_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


class ChatCompletionsClient:
    """Privileged S2 decision client over a local/compatible ChatCompletions API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        prompt_text: str,
        cfg: Dict[str, Any],
        history_turns: int = 4,
        timeout_s: float = 30.0,
        max_response_bytes: int = 1_000_000,
        temperature: float = 0.0,
        api_key_env: Optional[str] = None,
        opener: Optional[urllib.request.OpenerDirector] = None,
        audit_dir: Optional[Path] = None,
    ) -> None:
        url = urllib.parse.urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError("invalid_model_base_url")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("invalid_model_base_url")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.prompt_text = prompt_text
        self.cfg = cfg
        self.history_turns = int(history_turns)
        self.timeout_s = float(timeout_s)
        self.max_response_bytes = int(max_response_bytes)
        self.temperature = float(temperature)
        self.api_key_env = api_key_env
        self.opener = opener if opener is not None else _default_opener()
        self.audit_dir = Path(audit_dir) if audit_dir is not None else None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ValueError("model_api_key_env_missing")
            headers["Authorization"] = "Bearer " + key
        return headers

    def decide(self, observation: Dict[str, Any], recent: List[Dict[str, Any]]) -> Dict[str, Any]:
        capped = capped_observation(observation, self.cfg)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.prompt_text}]
        for turn in list(recent)[-self.history_turns:]:
            messages.extend(turn["messages"])
        messages.append(
            {"role": "user", "content": json.dumps({"observation": sanitized(capped)}, ensure_ascii=False)}
        )
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tool_schema(observation, self.cfg),
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "temperature": self.temperature,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body, allow_nan=False).encode(),
            headers=self._headers(),
        )
        start = time.monotonic()
        with self.opener.open(request, timeout=self.timeout_s) as response:  # type: ignore[union-attr]
            if getattr(response, "status", 200) != 200:
                raise ValueError("model_http_status_" + str(getattr(response, "status", "?")))
            raw = response.read(self.max_response_bytes + 1)
        if len(raw) > self.max_response_bytes:
            raise ValueError("model_response_too_large")

        parsed = strict_json(raw)
        if parsed.get("model") != self.model:
            raise ValueError("model_response_identity_mismatch")

        latency = time.monotonic() - start
        if self.audit_dir is not None:
            _atomic_write_json(
                self.audit_dir / ("api_" + str(time.time_ns()) + ".json"),
                {"request": body, "response": parsed, "latency_s": latency},
            )

        choices = parsed.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("model_choice_count_invalid")
        message = choices[0].get("message") or {}
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("exactly_one_tool_call_required")
        call = calls[0]
        if call.get("type") != "function" or (call.get("function") or {}).get("name") != "execute_primitive":
            raise ValueError("unexpected_model_tool")
        args = strict_json(call["function"]["arguments"])
        validate(args, observation, self.cfg)

        return {
            "args": args,
            "message": message,
            "call_id": call.get("id"),
            "response": parsed,
            "latency_s": latency,
            "request_sha256": digest(body),
            "usage": parsed.get("usage"),
        }
