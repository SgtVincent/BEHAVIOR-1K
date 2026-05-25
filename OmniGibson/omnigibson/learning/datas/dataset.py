"""Compatibility shim for OpenPi-Comet.

OpenPi-Comet includes a custom dataset implementation under its own repository.
For BEHAVIOR-1K runtime and evaluation, OmniGibson already provides
`BehaviorLeRobotDataset` in `omnigibson.learning.datas.lerobot_dataset`.

This module exists so any OpenPi-Comet references to
`omnigibson.learning.datas.dataset` continue to import successfully.
"""

from omnigibson.learning.datas.lerobot_dataset import BehaviorLeRobotDataset

__all__ = ["BehaviorLeRobotDataset"]