"""S2 decision contract: sanitizer, terminal predicate, validation, idempotency."""
from .contract import (  # noqa: F401
    TERMINAL,
    ARGS,
    FORBIDDEN,
    strict_json,
    digest,
    sanitized,
    capped_observation,
    full_task_terminal,
    tool_schema,
    validate,
    RequestTracker,
)
from .rule_service import RuleSelectionError, RuleS2Service, select_rule  # noqa: F401

__all__ = [
    "TERMINAL", "ARGS", "FORBIDDEN", "strict_json", "digest", "sanitized",
    "capped_observation", "full_task_terminal", "tool_schema", "validate",
    "RequestTracker", "RuleSelectionError", "RuleS2Service", "select_rule",
]
