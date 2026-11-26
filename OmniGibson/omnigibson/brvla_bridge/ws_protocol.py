"""2026 websocket policy protocol boundary (aligned to post1 ``eval.utils.network_utils``).

This module is sim-free and model-free. It defines the request/response
contract, server metadata gate, optional action-chunk cache, and the msgpack
ndarray envelope. Concrete websocket transport, reconnect policy, and simulator
wiring live in the integration lane on the remote 2026 environment.

Wire contract (post1):
  * On connect the server sends a metadata message first.
  * Reset request: ``{"reset": true}``.
  * Observation request: the obs dict, optionally carrying
    ``__action_chunk_size__`` when chunk replay is opted in.
  * Response: ``{"action": <array>, "action_chunk": <optional array>,
    "server_timing": {...}}``.
  * If present, ``action_chunk`` shape is ``(*action.shape[:-1], L, action_dim)``
    and ``action_chunk[..., 0, :]`` must equal ``action`` exactly.
"""
from typing import Any, Dict, Optional

from . import contracts
from ._arr import is_finite, shape_of, to_plain
from .errors import ContractViolation

# Lazy third-party imports live behind these functions so the package has no
# hard install requirement at import time.


def _msgpack():
    try:
        import msgpack  # type: ignore
    except ImportError as exc:  # pragma: no cover - remote env dependency
        raise ContractViolation("msgpack unavailable; install in the remote 2026 env only") from exc
    return msgpack


def _numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:  # pragma: no cover - remote env dependency
        raise ContractViolation("numpy unavailable; install in the remote 2026 env only") from exc
    return np


# ---- request construction --------------------------------------------------

def build_reset_request() -> Dict[str, Any]:
    return {contracts.RESET_KEY: True}


def build_observation_request(obs: Dict[str, Any], action_chunk_size: int = 0) -> Dict[str, Any]:
    request = dict(obs)
    if action_chunk_size and action_chunk_size > 1:
        request[contracts.ACTION_CHUNK_REQUEST_KEY] = int(action_chunk_size)
    return request


# ---- response validation --------------------------------------------------

def _validate_action(action, action_dim: int):
    plain = to_plain(action)
    shape = shape_of(plain)
    if shape not in ((action_dim,), (1, action_dim)):
        raise ContractViolation(f"action shape {list(shape)} != ({action_dim},) or (1,{action_dim})")
    if not is_finite(plain):
        raise ContractViolation("action contains non-finite values")
    return plain


def _row_equal(a, b) -> bool:
    return list(a) == list(b)


def validate_response(response: Dict[str, Any], action_dim: int, chunk_size: int = 0) -> Dict[str, Any]:
    if not isinstance(response, dict):
        raise ContractViolation("response must be a dict")
    if contracts.ACTION_KEY not in response:
        raise ContractViolation("server response missing 'action' key")
    action = _validate_action(response[contracts.ACTION_KEY], action_dim)

    action_chunk = None
    raw_chunk = response.get(contracts.ACTION_CHUNK_KEY)
    if chunk_size > 1 and raw_chunk is not None:
        chunk = to_plain(raw_chunk)
        cshape = shape_of(chunk)
        a_shape = shape_of(action)
        expected = a_shape[:-1] + (chunk_size, action_dim)
        if tuple(cshape) != tuple(expected):
            raise ContractViolation(
                f"action_chunk shape {list(cshape)} != expected {list(expected)}"
            )
        chunk_row0 = chunk[0] if len(a_shape) == 1 else chunk[0][0]
        action_row0 = action if len(a_shape) == 1 else action[0]
        if not _row_equal(chunk_row0, action_row0):
            raise ContractViolation("action_chunk first row must equal action exactly")
        if not is_finite(chunk):
            raise ContractViolation("action_chunk contains non-finite values")
        action_chunk = chunk

    return {
        "action": action,
        "action_chunk": action_chunk,
        "server_timing": response.get(contracts.SERVER_TIMING_KEY),
    }


class ChunkCache:
    """Client-side open-loop chunk replay, mirroring WebsocketClientPolicy."""

    def __init__(self):
        self._chunk = None
        self._index = 0

    def reset(self) -> None:
        self._chunk = None
        self._index = 0

    def store(self, chunk) -> None:
        self._chunk = to_plain(chunk)
        self._index = 1  # row 0 was already returned as the primary action

    def next_action(self):
        if self._chunk is None:
            return None
        if self._index >= len(self._chunk):
            return None
        row = self._chunk[self._index]
        self._index += 1
        return row

    @property
    def active(self) -> bool:
        return self._chunk is not None and self._index < len(self._chunk)


# ---- server metadata gate --------------------------------------------------

def verify_metadata(
    metadata: Dict[str, Any],
    expected_layout: str = contracts.SERVER_METADATA_LAYOUT,
    expected_horizon: int = contracts.HORIZON,
    expected_chunk: int = contracts.CHUNK_SIZE,
    require_history_every_step: bool = True,
) -> Dict[str, Any]:
    if metadata.get("action_layout") != expected_layout:
        raise ContractViolation(f"server layout mismatch: {metadata.get('action_layout')!r}")
    if metadata.get("horizon") != expected_horizon:
        raise ContractViolation(f"horizon mismatch: {metadata.get('horizon')}")
    if metadata.get("chunk_generated") != expected_chunk:
        raise ContractViolation(f"chunk size mismatch: {metadata.get('chunk_generated')}")
    if require_history_every_step and not metadata.get("history_every_control_step"):
        raise ContractViolation("history must be sent every control step")
    return {k: metadata.get(k) for k in contracts.SERVER_METADATA_FIELDS}


# ---- msgpack ndarray envelope (post1 pack_data / unpack_data semantics) ---

def pack_data(obj: Any) -> Any:
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):  # torch tensor duck
        obj = obj.detach().cpu().numpy()
    if hasattr(obj, "dtype") and hasattr(obj, "tobytes"):  # numpy ndarray duck
        dtype = obj.dtype.str
        return {
            contracts.NDARRAY_KEY: True,
            "data": obj.tobytes(),
            "dtype": dtype,
            "shape": list(obj.shape),
        }
    return obj


def unpack_data(obj: Any) -> Any:
    if isinstance(obj, dict) and contracts.NDARRAY_KEY in obj:
        np = _numpy()
        return np.ndarray(buffer=obj["data"], dtype=np.dtype(obj["dtype"]), shape=tuple(obj["shape"]))
    if isinstance(obj, dict) and contracts.NPGENERIC_KEY in obj:
        np = _numpy()
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


class MsgpackWire:
    """Thin msgpack codec with the ndarray envelope. Lazy import on first use."""

    def __init__(self):
        self._msgpack = _msgpack()

    def encode(self, request: Dict[str, Any]) -> bytes:
        return self._msgpack.packb(request, default=pack_data, use_bin_type=True)

    def decode(self, payload: bytes) -> Dict[str, Any]:
        return self._msgpack.unpackb(payload, object_hook=unpack_data, raw=False)
