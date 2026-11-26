# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: MIT
"""Sim-free public integration contracts for the 2026 BEHAVIOR-1K eval line.

This package contains only stdlib / typing contract code: schema tags,
websocket request/response shapes, and structural protocols. It does not
import Isaac Sim, torch, cuRobo, or any third-party dependency. Real adapters
that bind those dependencies live behind the seams in this package and are
certified only after an environment canary (build + first reset + first step).
"""

__version__ = "0.1.0-contract"
