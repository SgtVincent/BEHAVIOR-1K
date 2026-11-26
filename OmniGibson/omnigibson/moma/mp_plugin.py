"""Manifest-driven executor (MotionPlanner) plugin injection.

The real MP executor behind the a28 bridge is the 2026 cuRobo v2 binding.
That module is integrator-owned and delivered separately; this loader only
defines the pluggable seam:

- a private dependency manifest names the executor module and factory;
- the runner resolves the plugin at ``prepare`` time, never inside ``env.step``;
- a missing plugin is reported as a classified blocker, not a zero-action
  fallback and not a silent cuRobo v1 substitution.

This module itself imports no simulator / cuRobo / torch: the optional
executor module is resolved through :mod:`importlib` and its heavy imports land
only when that module is actually loaded in the GPU-bound stage.
"""
from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class PluginUnavailable(RuntimeError):
    """The executor plugin could not be bound; ``reason`` classifies the cause."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class PluginSpec:
    module: str
    factory: str
    version: str = "unknown"
    required: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, manifest: Dict[str, Any]) -> "PluginSpec":
        block = (manifest or {}).get("moma", {}).get("executor")
        if not isinstance(block, dict):
            raise PluginUnavailable("manifest_invalid", "missing moma.executor block")
        module = block.get("module")
        factory = block.get("factory")
        if not isinstance(module, str) or not module:
            raise PluginUnavailable("manifest_invalid", "executor.module missing")
        if not isinstance(factory, str) or not factory:
            raise PluginUnavailable("manifest_invalid", "executor.factory missing")
        kwargs = block.get("kwargs") or {}
        if not isinstance(kwargs, dict):
            raise PluginUnavailable("manifest_invalid", "executor.kwargs must be an object")
        return cls(
            module=module,
            factory=factory,
            version=str(block.get("version", "unknown")),
            required=bool(block.get("required", True)),
            kwargs=dict(kwargs),
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "factory": self.factory,
            "version": self.version,
            "required": self.required,
            "kwargs": self.kwargs,
        }


def module_resolvable(module: str) -> bool:
    """Static check: the module name can be found, without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def load_executor(manifest: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
    """Resolve and instantiate the executor named in the manifest.

    Raises :class:`PluginUnavailable` with a classified reason when the plugin
    is absent or mis-wired. ``context`` (config, accessors, live reads) is
    forwarded to the factory alongside the manifest kwargs.
    """
    spec = PluginSpec.from_manifest(manifest)
    if not module_resolvable(spec.module):
        raise PluginUnavailable("module_missing", spec.module)
    try:
        module = importlib.import_module(spec.module)
    except ImportError as exc:
        raise PluginUnavailable("module_import_failed", f"{spec.module}: {exc}") from exc
    factory: Optional[Callable[..., Any]] = getattr(module, spec.factory, None)
    if not callable(factory):
        raise PluginUnavailable("factory_missing", f"{spec.module}.{spec.factory}")
    call_kwargs: Dict[str, Any] = dict(spec.kwargs)
    call_kwargs.update(context or {})
    try:
        executor = factory(**call_kwargs)
    except Exception as exc:  # instantiation failure surfaces as a blocker
        raise PluginUnavailable("instantiation_failed", f"{spec.module}.{spec.factory}: {exc}") from exc
    if not callable(getattr(executor, "plan", None)):
        raise PluginUnavailable("executor_contract_missing", "bound executor has no plan()")
    return executor


def classify_plugin(spec: PluginSpec) -> Dict[str, Any]:
    """Static doctor view: resolvable / missing, without instantiating."""
    resolvable = module_resolvable(spec.module)
    if resolvable:
        try:
            module = importlib.import_module(spec.module)
            present = callable(getattr(module, spec.factory, None))
        except ImportError:
            present = False
        return {
            "status": "OK" if present else "MISSING",
            "detail": "factory present" if present else f"factory {spec.factory} absent",
            "spec": spec.describe(),
        }
    return {"status": "MISSING", "detail": f"module {spec.module} not resolvable", "spec": spec.describe()}


def plugin_manifest_path(manifest: Dict[str, Any]) -> Optional[str]:
    """Optional explicit path recorded for provenance; never used to execute."""
    block = (manifest or {}).get("moma", {}).get("executor", {})
    if isinstance(block, dict):
        return block.get("manifest_path")
    return None


def required_reasons(results: List[Dict[str, Any]]) -> List[str]:
    """Blocking plugin problems for required executors."""
    return [f"{r['spec']['module']}: {r['detail']}" for r in results if r["status"] != "OK" and r["spec"]["required"]]
