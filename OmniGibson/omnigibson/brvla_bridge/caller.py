"""Real websocket policy caller and model-input adapter (freeze contracts).

This module wires the protocol boundary (:mod:`ws_protocol`) to an injectable
transport. The concrete websocket client, reconnect policy, and simulator
binding live in the integration lane; here the transport is a duck-typed
interface so the request/response path, chunk replay, metadata gate, and
frozen bundle assembly are exercisable with a fake transport on the CPU
contract tests.

Model loading/forward is NOT performed here. A built bundle is recorded as
input evidence only; a CPU-side construction does not count as a model
forward admission.
"""
from typing import Any, Callable, Dict, List, Optional, Protocol

from . import contracts, schema, ws_protocol
from ._arr import to_plain
from .errors import ContractViolation, Port2026Error


class Transport(Protocol):
    """Ducted transport implemented by the remote websocket client."""

    def healthz(self) -> None: ...

    def connect(self) -> Dict[str, Any]: ...

    def request(self, payload: bytes) -> Any: ...

    def reset(self, request: Dict[str, Any]) -> None: ...


class WsPolicyCaller:
    """Policy caller mirroring post1 ``eval.policies.WebsocketPolicy``.

    Args:
        host/port: frozen server endpoint (recorded; concrete connection remote).
        action_dim: frozen contract action width (23).
        chunk_size: generated chunk width requested when > 1.
        transport: injectable :class:`Transport`; required before ``connect``.
        wire: optional msgpack wire; defaults to :class:`MsgpackWire`.
    """

    def __init__(
        self,
        host: str,
        port: int,
        action_dim: int = contracts.ACTION23,
        chunk_size: int = contracts.CHUNK_SIZE,
        transport: Optional[Transport] = None,
        wire: Optional[ws_protocol.MsgpackWire] = None,
    ):
        self.host = host
        self.port = int(port)
        self._action_dim = action_dim
        self._chunk_size = int(chunk_size)
        self._transport = transport
        self._wire = wire or ws_protocol.MsgpackWire()
        self._cache = ws_protocol.ChunkCache()
        self._metadata: Optional[Dict[str, Any]] = None

    def connect(self) -> Dict[str, Any]:
        if self._transport is None:
            raise Port2026Error("transport not injected; concrete WS client belongs to the integration lane")
        self._transport.healthz()
        metadata = self._transport.connect()
        self._metadata = ws_protocol.verify_metadata(metadata)
        return dict(self._metadata)

    def reset(self) -> None:
        self._cache.reset()
        if self._transport is not None:
            self._transport.reset(ws_protocol.build_reset_request())

    def forward(self, obs: Dict[str, Any], need_new_action: bool = True) -> List[float]:
        if self._transport is None:
            raise Port2026Error("transport not injected")
        if self._metadata is None:
            raise ContractViolation("caller not connected; metadata gate not passed")

        if self._cache.active and not need_new_action:
            row = self._cache.next_action()
            if row is not None:
                return list(row)

        request = ws_protocol.build_observation_request(
            obs, self._chunk_size if self._chunk_size > 1 else 0)
        payload = self._wire.encode(request)
        response = self._transport.request(payload)
        if isinstance(response, (bytes, bytearray)):
            response = self._wire.decode(bytes(response))
        validated = ws_protocol.validate_response(
            response, self._action_dim, self._chunk_size)

        if validated["action_chunk"] is not None:
            self._cache.store(validated["action_chunk"])

        action = validated["action"]
        plain = to_plain(action)
        if len(plain) == self._action_dim:
            return list(plain)
        return list(plain[0])


def _zero_pad_actions() -> List[List[float]]:
    return [[0.0] * contracts.STATE45 for _ in range(contracts.CHUNK_SIZE)]


class ModelInputAdapter:
    """Assembles the frozen inference bundle from the state provider + images.

    The bundle keeps ``robot_type="br2"`` (frozen model identity; not the
    R1Pro sim robot label), ``br_states (4,45)`` finite history, zero action
    pad and pad mask. Validation delegates to :mod:`schema`.
    """

    def __init__(
        self,
        provider,
        instruction: str,
        dataname: Optional[str] = None,
        expected_dataname: Optional[str] = None,
    ):
        self._provider = provider
        self._instruction = instruction
        self._dataname = dataname
        self._expected_dataname = expected_dataname

    def build(self, images: Any, proprio_row: List[float]) -> Dict[str, Any]:
        self._provider.append(proprio_row)
        history = self._provider.history()
        bundle = {
            "dataname": self._dataname,
            "instruction": self._instruction,
            "robot_type": schema.EXPECTED_ROBOT_TYPE,
            "images": images,
            "br_states": history,
            "br_actions": _zero_pad_actions(),
            "br_action_is_pad": [False] * contracts.CHUNK_SIZE,
        }
        return schema.validate_canary_obs(bundle, expected_dataname=self._expected_dataname)
