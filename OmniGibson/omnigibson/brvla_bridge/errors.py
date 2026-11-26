"""Exceptions for the BRVLA 2026 candidate portable package.

All classes are stdlib-only. The package never imports torch, numpy, omnigibson,
Isaac Sim, Kit, msgpack, or the frozen model at module import time.
"""

class Port2026Error(Exception):
    """Base class for every candidate-package failure."""

class SchemaContractError(Port2026Error):
    """Online observation / action schema violates the frozen contract."""

class MappingNotDeclaredError(Port2026Error):
    """A 2026 state field mapping to legacy width 45 has not been verified."""

class BackendNotDeclaredError(Port2026Error):
    """Controller backend requested without an integrator-issued receipt."""

class ContractViolation(Port2026Error):
    """Server metadata, layout, step count, or frozen identity mismatch."""


class ControlledActionBlocked(ContractViolation):
    """An action without a legal joint-native23 label reached the step boundary.

    Raised by the consumer guard immediately before the official ``env.step``.
    The model query may have succeeded; the action is held and zero simulator
    steps are taken. Same-width physical23 / IK-target23 rows must never be
    stepped as joint-native23 commands.
    """
