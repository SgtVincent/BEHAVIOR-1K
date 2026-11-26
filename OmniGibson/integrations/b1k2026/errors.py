# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: MIT
"""Contract-level exception types for the 2026 integration line."""

from __future__ import annotations

__all__ = [
    "ContractViolation",
    "SchemaMismatch",
    "ChunkLengthExceeded",
    "UnverifiedAdapter",
]


class ContractViolation(ValueError):
    """Raised when a request/response object violates the frozen contract."""


class SchemaMismatch(ContractViolation):
    """Raised when action width, schema tag, or finiteness is inconsistent."""


class ChunkLengthExceeded(ContractViolation):
    """Raised when an opt-in chunk sequence exceeds its declared chunk size K."""


class UnverifiedAdapter(RuntimeError):
    """Raised when an adapter without environment-canary evidence is invoked.

    CPU contract tests and sealed manifests do not count as a passing adapter;
    certification requires a bound resource: build, reset, first frame, and
    ``env.step`` evidence for the specific profile and dependency version.
    """
