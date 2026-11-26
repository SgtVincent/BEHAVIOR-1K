# Copyright (c) 2026 B1K2026 integration line
# SPDX-License-Identifier: Apache-2.0
"""Plugin factory entry point resolved by ``moma.mp_plugin.load_executor``.

Manifest block (private manifest lives outside git):

.. code-block:: yaml

   moma:
     executor:
       module: omnigibson.curobo_v2.factory
       factory: make_curobo_v2_executor
       version: 0.1.0-pre
       required: true
       kwargs:
         robot_profile: r1pro
         device_uuid: GPU-...
         robot_config_yaml: /private/path/r1pro.curobo.yaml

The loader calls ``factory(**call_kwargs)`` where ``call_kwargs`` merges the
manifest ``kwargs`` with the runner context (``cfg``, ``robot``). This module
imports no torch / curobo / omni dependency; heavy work lands in the backend
lazy bind on the GPU-bound stage.
"""
from __future__ import annotations

from typing import Any, Dict

from .executor import CuroboV2Executor

__all__ = ["make_curobo_v2_executor", "PLUGIN_VERSION"]

PLUGIN_VERSION = "0.1.0-pre"


def make_curobo_v2_executor(**kwargs: Any) -> CuroboV2Executor:
    """Construct the executor plugin.

    Accepted kwargs are either backend binding parameters
    (``robot_profile``, ``device_uuid``, ``robot_config_yaml``, ``urdf_path``,
    ``asset_path``, ``cache_dir``, ``canary_evidence``, ``private_manifest``)
    or runner context keys (``cfg``, ``robot``), which are retained for seam
    hooks without being touched at construction.
    """
    context: Dict[str, Any] = dict(kwargs)
    return CuroboV2Executor(context=context)
