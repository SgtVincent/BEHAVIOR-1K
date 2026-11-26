"""Tiny stdlib assertion helpers so candidate tests need no pytest on the remote env."""
import contextlib

from brvla_bridge.errors import Port2026Error


@contextlib.contextmanager
def raises(expected):
    try:
        yield
    except expected:
        return
    except Port2026Error as exc:  # mapped base class where applicable
        if isinstance(exc, expected):
            return
        raise AssertionError(f"wrong exception type: {type(exc).__name__}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - test harness surfaces the real failure
        raise AssertionError(f"expected {expected.__name__}, got {type(exc).__name__}: {exc}") from exc
    raise AssertionError(f"expected {expected.__name__}, no exception raised")
