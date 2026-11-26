"""State mapping provider interface (2026 state -> legacy state45).

The frozen pipeline consumes a 45-d state history window. 2026 data ships two
different source widths:

    demos (LeRobot v3 parquet): observation.state width 61
    rawdata (HDF5):            state width 623

The 61/623 -> 45 field mapping has **not** been verified against remote data
this round. No placeholder zero-fill, no 256-slice reuse, and no padding bridge
is allowed. Until a verified provider is registered, every mapping call fails
closed with :class:`MappingNotDeclaredError`.
"""
from dataclasses import dataclass
from typing import List, Optional

from . import contracts
from ._arr import is_finite, to_plain
from .errors import MappingNotDeclaredError, Port2026Error


@dataclass(frozen=True)
class FieldMapping:
    source_index: Optional[int]
    role: str
    status: str  # "bound" | "unknown_pending_remote"
    note: str = ""


@dataclass(frozen=True)
class SchemaDescriptor:
    source_name: str
    source_width: int
    target_width: int
    fields: List[FieldMapping]

    def unknown_fields(self) -> List[FieldMapping]:
        return [f for f in self.fields if f.status != "bound"]


class StateMappingProvider:
    """Abstract mapping implementation registered by the integration lane."""

    def descriptor(self) -> SchemaDescriptor:
        raise NotImplementedError

    def map_state(self, row) -> list:
        """Map one source-state row to the frozen 45-d state."""
        raise NotImplementedError


class UnmappedProvider(StateMappingProvider):
    """Fail-closed placeholder used until the remote field mapping is reviewed."""

    def __init__(self, source_name: str, source_width: int):
        self._descriptor = SchemaDescriptor(
            source_name=source_name,
            source_width=source_width,
            target_width=contracts.STATE45,
            fields=[
                FieldMapping(None, f"legacy_state45_slot_{i}", "unknown_pending_remote",
                             "mapping to legacy 45 not verified against remote 2026 data")
                for i in range(contracts.STATE45)
            ],
        )

    def descriptor(self) -> SchemaDescriptor:
        return self._descriptor

    def map_state(self, row) -> list:
        raise MappingNotDeclaredError(
            f"{self._descriptor.source_name}: {self._descriptor.source_width}->45 mapping "
            "is not declared; do not zero-fill or reuse the 256 slice"
        )


_REGISTRY = {}


def register_provider(name: str, provider: StateMappingProvider) -> None:
    if not isinstance(provider, StateMappingProvider):
        raise Port2026Error("provider must implement StateMappingProvider")
    if name in _REGISTRY:
        raise Port2026Error(f"provider already registered: {name}")
    _REGISTRY[name] = provider


def get_provider(name: str) -> StateMappingProvider:
    if name not in _REGISTRY:
        raise Port2026Error(f"unknown state mapping provider: {name}")
    return _REGISTRY[name]


def clear_registry() -> None:
    """Test-only helper; never called from production wiring."""
    _REGISTRY.clear()


def validate_mapped_state(row) -> list:
    plain = to_plain(row)
    if len(plain) != contracts.STATE45:
        raise MappingNotDeclaredError(f"mapped state width must be {contracts.STATE45}, got {len(plain)}")
    if not is_finite(plain):
        raise MappingNotDeclaredError("mapped state contains non-finite values")
    return plain
