"""Private frozen-asset manifest loading.

Checkpoint directories, stats paths/hashes, training-source receipts, prompt
hashes, and replay data refs are **private runtime material**. They are never
hardcoded in public source; the caller supplies a manifest path (env
``BRVLA_MANIFEST_PATH`` or explicit argument) that lives outside the public
worktree. This module only validates structure and fails closed on missing
fields; it does not open weights, stats, or datasets.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .errors import Port2026Error

REQUIRED_TOP = ("checkpoint", "stats", "training_source", "server_session")


@dataclass(frozen=True)
class FrozenManifest:
    checkpoint: Dict[str, Any]
    stats: Dict[str, Any]
    training_source: Dict[str, Any]
    server_session: Dict[str, Any]
    replay: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return getattr(self, section, {}).get(key, default)


def load_manifest(path: Optional[str] = None) -> FrozenManifest:
    path = path or os.environ.get("BRVLA_MANIFEST_PATH")
    if not path:
        raise Port2026Error("BRVLA_MANIFEST_PATH not set; frozen asset refs are private")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    missing = [k for k in REQUIRED_TOP if k not in data]
    if missing:
        raise Port2026Error(f"manifest missing sections: {missing}")
    return FrozenManifest(
        checkpoint=data["checkpoint"],
        stats=data["stats"],
        training_source=data["training_source"],
        server_session=data["server_session"],
        replay=data.get("replay", {}),
        raw=data.get("raw", {}),
    )
