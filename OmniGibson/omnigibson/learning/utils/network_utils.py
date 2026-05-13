"""
Adapted from https://github.com/Physical-Intelligence/openpi
"""

import asyncio
import functools
import http
import json
import logging
import msgpack
import numpy as np
import time
import torch as th
import traceback
import websockets.asyncio.server as _server
import websockets.sync.client
import websockets
import requests
from copy import deepcopy
from omnigibson.macros import gm
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = ["WebsocketClientPolicy", "WebsocketPolicyServer"]


class WebsocketClientPolicy:
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        allow_reconnect: bool = False,
        expected_task_name: Optional[str] = None,
        expected_task_prompt_sha256: Optional[str] = None,
        expected_server_run_id: Optional[str] = None,
        expected_server_token: Optional[str] = None,
    ) -> None:
        self._uri = f"wss://{host}" if int(port) == 443 else f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = None, None
        self._allow_reconnect = allow_reconnect
        self._last_generated_subtask = None
        self._last_server_fresh_action = False
        self._last_prompt_debug = None
        self._expected_server_identity = {
            "task_name": expected_task_name,
            "task_prompt_sha256": expected_task_prompt_sha256,
            "server_run_id": expected_server_run_id,
            "server_token": expected_server_token,
        }

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _validate_server_identity(self, metadata: Dict, *, source: str) -> None:
        mismatches = []
        for key, expected in self._expected_server_identity.items():
            if expected is None:
                continue
            actual = metadata.get(key)
            if actual != expected:
                mismatches.append(f"{key}: expected={expected!r}, actual={actual!r}")
        if mismatches:
            raise RuntimeError(
                f"Connected to unexpected policy server via {source}. "
                + "; ".join(mismatches)
            )

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        # TODO [Wensi]: use URL parser instead of this
        # Extract host and port for health check
        host_port = self._uri.replace("ws://", "").replace("wss://", "")
        if ":" in host_port:
            host, port = host_port.split(":")
            health_url = f"https://{host}:{port}/healthz" if int(port) == 443 else f"http://{host}:{port}/healthz"
        else:
            health_url = f"http://{host_port}/healthz"

        # First, wait for the health check to pass
        while True:
            try:
                # Important: do NOT honor HTTP proxy env vars for localhost health checks.
                # Some cluster environments set proxies globally, which breaks local requests
                # and causes an infinite wait here.
                response = requests.get(
                    health_url,
                    timeout=2,
                    proxies={"http": None, "https": None},
                )
                if response.ok:
                    if any(value is not None for value in self._expected_server_identity.values()):
                        try:
                            self._validate_server_identity(response.json(), source=f"healthz:{health_url}")
                        except ValueError:
                            raise RuntimeError(f"Health check response from {health_url} is not valid JSON")
                    logger.info("Health check passed, attempting websocket connection...")
                    break
            except RuntimeError:
                raise
            except Exception:
                pass
            logger.info(f"Health check failed, waiting for server at {health_url}...")
            time.sleep(5)

        # Now attempt websocket connection (rest of the code remains the same)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    proxy=None,
                    ping_interval=60,
                    ping_timeout=300,
                )
                metadata = unpackb(conn.recv())
                self._validate_server_identity(metadata, source=f"websocket:{self._uri}")
                logger.info("Connected to server!")
                return conn, metadata
            except (ConnectionRefusedError, websockets.exceptions.InvalidMessage, EOFError) as e:
                logger.info(f"Websocket connection failed ({e}), retrying...")
                time.sleep(5)

    def act(self, obs: Dict) -> th.Tensor:
        if self._ws is None:
            self._ws, self._server_metadata = self._wait_for_server()

        t0 = time.perf_counter()
        data = self._packer.pack(obs)
        pack_s = time.perf_counter() - t0
        while True:
            try:
                t1 = time.perf_counter()
                self._ws.send(data)
                send_s = time.perf_counter() - t1

                t2 = time.perf_counter()
                response = self._ws.recv()
                recv_s = time.perf_counter() - t2
                break
            except websockets.exceptions.ConnectionClosedError:
                if self._allow_reconnect:
                    logger.warning("Connection to server lost, attempting to reconnect...")
                    self._ws, self._server_metadata = self._wait_for_server()
                    continue
                raise
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        t3 = time.perf_counter()
        action_dict = unpackb(response)
        unpack_s = time.perf_counter() - t3
        try:
            action_np = deepcopy(action_dict["action"])
        except KeyError:
            # We try getting action one more time before raising error
            logger.warning("No action received from server, retrying one more time...")
            t1 = time.perf_counter()
            self._ws.send(data)
            send_s = time.perf_counter() - t1

            t2 = time.perf_counter()
            response = self._ws.recv()
            recv_s = time.perf_counter() - t2

            t3 = time.perf_counter()
            action_dict = unpackb(response)
            unpack_s = time.perf_counter() - t3
            action_np = deepcopy(action_dict["action"])

        self._last_client_timing = {
            "pack_ms": pack_s * 1000,
            "send_ms": send_s * 1000,
            "recv_ms": recv_s * 1000,
            "unpack_ms": unpack_s * 1000,
            "rtt_ms": (pack_s + send_s + recv_s + unpack_s) * 1000,
        }
        self._last_server_timing = deepcopy(action_dict.get("server_timing", {}))
        self._last_server_fresh_action = bool(action_dict.get("fresh_action_plan", False))
        if "generated_subtask" in action_dict and action_dict["generated_subtask"] is not None:
            self._last_generated_subtask = action_dict["generated_subtask"]
        self._last_prompt_debug = deepcopy(action_dict.get("prompt_debug"))
        action = th.from_numpy(action_np).to(th.float32)
        return action

    def reset(self) -> None:
        if self._ws is None:
            self._ws, self._server_metadata = self._wait_for_server()

        data = self._packer.pack({"reset": True})
        self._ws.send(data)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def _health_payload(self) -> dict:
        return {"ok": True, **self._metadata}

    def _process_request(self, connection, request) -> Optional[Any]:
        if hasattr(request, "path") and request.path == "/healthz":
            body = json.dumps(self._health_payload(), sort_keys=True).encode("utf-8") + b"\n"
            headers = {"Content-Type": "application/json"}
            if hasattr(connection, "respond"):
                return connection.respond(http.HTTPStatus.OK, body.decode("utf-8"))
            return http.HTTPStatus.OK, headers, body
        return None

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        logger.info(f"Starting websocket server on {self._host}:{self._port}...")
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._process_request,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        # IMPORTANT:
        # Many policies maintain rollout state (e.g., action queues / step counters).
        # If multiple evaluators share a single websocket server, policy state must be
        # isolated per connection, otherwise resets and rollout state will collide.
        policy = self._policy
        if hasattr(self._policy, "spawn_session"):
            try:
                policy = self._policy.spawn_session()
            except Exception:
                logger.warning(
                    "Policy exposes spawn_session() but session creation failed; falling back to shared policy. "
                    "This may break multi-client evaluation.\n" + traceback.format_exc()
                )

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in result:
                    policy.reset()
                    continue

                obs = deepcopy(result)

                infer_time = time.monotonic()
                action = policy.act(obs)
                infer_time = time.monotonic() - infer_time

                action = {
                    "action": action.cpu().numpy(),
                }
                generated_subtask = getattr(policy, "last_generated_subtask", None)
                if generated_subtask is not None:
                    action["generated_subtask"] = generated_subtask
                prompt_debug = getattr(policy, "last_prompt_debug", None)
                if prompt_debug is not None:
                    action["prompt_debug"] = deepcopy(prompt_debug)
                action["fresh_action_plan"] = bool(getattr(policy, "last_policy_inferred", False))
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logger.error(f"Error in connection from {websocket.remote_address}:\n{traceback.format_exc()}")
                if gm.DEBUG:
                    await websocket.send(traceback.format_exc())
                try:
                    # Try new websockets API first
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                except AttributeError:
                    # Fallback for older websockets versions
                    await websocket.close(code=1011, reason="Internal server error")
                raise
"""
Adds NumPy array support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.
"""


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
