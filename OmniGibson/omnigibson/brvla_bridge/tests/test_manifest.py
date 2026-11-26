"""Manifest: private frozen-asset manifest loads from an explicit path/env only."""
import json
import os
import tempfile

from brvla_bridge import manifest
from brvla_bridge.errors import Port2026Error
from brvla_bridge.tests._assert import raises


def _write(tmp, payload):
    fd, path = tempfile.mkstemp(suffix=".json", dir=tmp)
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def test_missing_manifest_path_raises():
    old = os.environ.pop("BRVLA_MANIFEST_PATH", None)
    try:
        with raises(Port2026Error):
            manifest.load_manifest()
    finally:
        if old is not None:
            os.environ["BRVLA_MANIFEST_PATH"] = old


def test_loads_required_sections():
    payload = {
        "checkpoint": {"index_sha256": "deadbeef", "index_keys": 1503},
        "stats": {"stats_full": {"sha256": "abc", "widths": [45]}},
        "training_source": {"dataname": "ds-brvla45"},
        "server_session": {"run_id": "run-x", "prompt_sha256": "cafe"},
        "replay": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, payload)
        m = manifest.load_manifest(path)
        assert m.get("checkpoint", "index_keys") == 1503
        assert m.get("stats", "stats_full", {})["sha256"] == "abc"


def test_missing_section_rejected():
    payload = {"checkpoint": {}, "stats": {}, "training_source": {}, "server_session": {}}
    del payload["server_session"]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, payload)
        with raises(Port2026Error):
            manifest.load_manifest(path)
