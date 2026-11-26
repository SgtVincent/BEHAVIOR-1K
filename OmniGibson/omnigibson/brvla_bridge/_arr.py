"""Stdlib duck-typed array helpers.

We deliberately do not import numpy. Values may be plain lists/tuples, or duck
arrays exposing ``tolist`` / ``shape`` (e.g. numpy, torch tensors on the remote
environment). Helpers copy them into nested plain Python before validation so
all candidate logic and tests run without a numerical dependency installed.
"""
import math

from .errors import SchemaContractError


def to_plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, bytes, bool)):
        return value
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def shape_of(value):
    """Return nested shape for plain sequences; duck ``.shape`` otherwise.

    Empty sequences stop descending immediately: an empty list reports a
    leading ``0`` dimension and must not feed ``[]`` back into the loop.
    """
    if hasattr(value, "shape") and not isinstance(value, (str, bytes)):
        return tuple(value.shape)
    v = to_plain(value)
    if not isinstance(v, list):
        return ()
    shape, cur = [], v
    while isinstance(cur, list):
        shape.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    return tuple(shape)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_finite(value):
    v = to_plain(value)
    if _is_number(v):
        return math.isfinite(v)
    if isinstance(v, list):
        return all(is_finite(x) for x in v)
    return True  # strings/None/booleans are not numeric state


def require_shape(value, expected, label):
    shape = shape_of(value)
    if tuple(shape) != tuple(expected):
        raise SchemaContractError(f"{label}: expected shape {expected}, got {list(shape)}")
    return to_plain(value)


def require_finite(value, label):
    if not is_finite(value):
        raise SchemaContractError(f"{label}: non-finite value present")
    return to_plain(value)
