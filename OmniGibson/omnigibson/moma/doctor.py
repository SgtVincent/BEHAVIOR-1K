"""Static preflight doctor for the MoMa runner (no Kit, no sim boot).

The doctor answers one question before the GPU canary: *what is the single
missing piece that blocks a formal run?* It checks, without importing the
simulator:

- frozen effective config validity;
- required / optional Python dependencies (optional simulator/model stacks
  are DEFERRED to the GPU stage, not FAILed here);
- the manifest-injected MP executor plugin (module resolvable, factory present);
- initial-state / asset subset paths on the local cache;
- S2 endpoint shape and API-key env presence.

A check result is ``OK`` / ``MISSING`` / ``DEFERRED`` / ``FAIL``. Only ``FAIL``
and blocking ``MISSING`` entries feed :meth:`Report.blocking`.
"""
from __future__ import annotations

import importlib.util
import os
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RunnerConfig
from .mp_plugin import PluginSpec, classify_plugin


@dataclass
class Check:
    name: str
    status: str
    detail: str
    blocking: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail, "blocking": self.blocking}


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    def blocking(self) -> List[Check]:
        return [c for c in self.checks if c.blocking]

    def missing_pointer(self) -> Optional[Dict[str, str]]:
        blocking = self.blocking()
        if not blocking:
            return None
        first = blocking[0]
        return {"kind": first.name, "detail": first.detail}

    def ready(self) -> bool:
        return not self.blocking()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ready": self.ready(),
            "blocking": [c.to_dict() for c in self.blocking()],
            "missing_pointer": self.missing_pointer(),
            "checks": [c.to_dict() for c in self.checks],
        }


def _spec(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def check_config(cfg: RunnerConfig) -> List[Check]:
    problems = cfg.validate()
    if problems:
        return [Check("config", "FAIL", "; ".join(problems), blocking=True)]
    return [Check("config", "OK", "effective config valid", blocking=False)]


def check_python_deps() -> List[Check]:
    checks: List[Check] = []
    checks.append(Check("dep:numpy", "OK" if _spec("numpy") else "FAIL",
                        "numpy resolvable" if _spec("numpy") else "numpy missing",
                        blocking=not _spec("numpy")))
    for pkg, stage in (("torch", "GPU canary"), ("omnigibson", "GPU canary"),
                       ("warp", "GPU canary"), ("cvxpy", "independent project env")):
        present = _spec(pkg)
        checks.append(Check(
            f"dep:{pkg}",
            "OK" if present else "DEFERRED",
            "present" if present else f"absent; {stage} provisioning owns it",
            blocking=False,
        ))
    return checks


def check_executor_plugin(manifest: Optional[Dict[str, Any]]) -> List[Check]:
    if manifest is None:
        return [Check("executor_plugin", "MISSING", "no executor plugin manifest injected", blocking=True)]
    try:
        spec = PluginSpec.from_manifest(manifest)
    except Exception as exc:  # PluginUnavailable carries the classified reason
        return [Check("executor_plugin", "FAIL", str(exc), blocking=True)]
    result = classify_plugin(spec)
    blocking = result["status"] != "OK" and spec.required
    return [Check("executor_plugin", result["status"], result["detail"], blocking=blocking)]


def check_assets(cfg: RunnerConfig) -> List[Check]:
    checks: List[Check] = []
    for label, path in (("initial_state_npz", cfg.initial_state_npz),
                        ("initial_state_hdf5", cfg.initial_state_hdf5),
                        ("asset_manifest", cfg.asset_manifest_path)):
        if path is None:
            checks.append(Check(f"asset:{label}", "DEFERRED", "path not configured; resolved by local cache manifest",
                                blocking=False))
        else:
            exists = Path(path).is_file()
            checks.append(Check(f"asset:{label}", "OK" if exists else "MISSING",
                                path if exists else f"not found: {path}",
                                blocking=not exists))
    return checks


def check_s2_endpoint(cfg: RunnerConfig) -> List[Check]:
    if cfg.policy_mode != "external_s2":
        return [Check("s2_endpoint", "DEFERRED", "internal policy mode; no S2 endpoint required")]
    url = urllib.parse.urlsplit(cfg.model_base_url)
    if url.scheme not in ("http", "https") or not url.hostname:
        return [Check("s2_endpoint", "FAIL", "base_url scheme/host invalid", blocking=True)]
    if url.username or url.password or url.query or url.fragment:
        return [Check("s2_endpoint", "FAIL", "base_url carries credentials/query/fragment", blocking=True)]
    if cfg.model_api_key_env is not None:
        if not os.environ.get(cfg.model_api_key_env):
            return [Check("s2_api_key", "MISSING", f"env {cfg.model_api_key_env} not set", blocking=True)]
    return [Check("s2_endpoint", "OK", f"{url.scheme}://{url.hostname} endpoint shape valid", blocking=False)]


def run_doctor(cfg: RunnerConfig, manifest: Optional[Dict[str, Any]] = None) -> Report:
    """Assemble the static report. No simulator import, no network call."""
    report = Report()
    for check in check_config(cfg):
        report.add(check)
    for check in check_python_deps():
        report.add(check)
    for check in check_executor_plugin(manifest):
        report.add(check)
    for check in check_assets(cfg):
        report.add(check)
    for check in check_s2_endpoint(cfg):
        report.add(check)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    """Static CLI: print the JSON preflight report; exit 0 ready, 2 blocked."""
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="MoMa static preflight doctor (no Kit, no sim boot)")
    parser.add_argument("--run-id", default="moma_static_doctor")
    parser.add_argument("--manifest", default=None, help="private executor plugin manifest JSON path")
    parser.add_argument("--initial-npz", default=None)
    parser.add_argument("--initial-hdf5", default=None)
    parser.add_argument("--asset-manifest", default=None)
    parser.add_argument("--model-base-url", default=None)
    args = parser.parse_args(argv)

    overrides: Dict[str, Any] = {}
    if args.initial_npz:
        overrides["initial_state_npz"] = args.initial_npz
    if args.initial_hdf5:
        overrides["initial_state_hdf5"] = args.initial_hdf5
    if args.asset_manifest:
        overrides["asset_manifest_path"] = args.asset_manifest
    if args.model_base_url:
        overrides["model_base_url"] = args.model_base_url

    cfg = RunnerConfig(run_id=args.run_id, **overrides)
    manifest: Optional[Dict[str, Any]] = None
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))

    report = run_doctor(cfg, manifest)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ready() else 2
