"""Ground-truth plan loader for BEHAVIOR-1K demo annotations.

This module provides `GTPlanLoader`, a lightweight utility that loads
skill-level annotations from a demo JSON file, sorts them by start frame,
and exposes an iterator-like interface for consuming skills sequentially.

It also supports mapping raw skill descriptions to richer prompts via an
optional `task_mapping.json` (e.g. openpi-comet's task mapping).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _flatten_numeric(x: Any) -> List[int]:
    """Recursively flatten nested lists/tuples of numbers into a flat int list."""
    out: List[int] = []
    if isinstance(x, (list, tuple)):
        for y in x:
            out.extend(_flatten_numeric(y))
    elif isinstance(x, (int, float)):
        out.append(int(x))
    return out


def _normalize_frame_duration(raw: Any) -> List[int]:
    """Ensure frame_duration is a [start, end] int list."""
    vals = _flatten_numeric(raw)
    if len(vals) < 2:
        raise ValueError(f"Invalid frame_duration: {raw}")
    return [int(vals[0]), int(vals[-1])]


class GTPlanLoader:
    """Load and iterate over ground-truth skill annotations for a single demo.

    Args:
        demo_data_path: Root path to the demo dataset (contains ``annotations/``).
        task_name: Task name used to locate the annotation sub-directory.
        demo_id: Demo identifier (e.g. ``00000010``).

    Example:
        >>> loader = GTPlanLoader(
        ...     demo_data_path="/path/to/2025-challenge-demos",
        ...     task_name="turning_on_radio",
        ...     demo_id="00000010",
        ... )
        >>> plan = loader.load_plan()
        >>> while not loader.is_exhausted():
        ...     skill = loader.get_current_skill()
        ...     prompt = loader.get_skill_prompt(skill["skill_description"])
        ...     loader.advance()
    """

    # Default path to the openpi-comet task mapping file.
    # Callers can override via the class attribute or environment variable.
    DEFAULT_TASK_MAPPING_PATH: str = (
        "/mnt/bn/behavior-data-hl/chenjunting/repo/openpi-comet/scripts/task_mapping.json"
    )

    def __init__(
        self,
        demo_data_path: str,
        task_name: str,
        demo_id: str,
    ) -> None:
        self.demo_data_path = os.path.expanduser(str(demo_data_path))
        self.task_name = str(task_name)
        self.demo_id = str(demo_id)

        self._plan: List[Dict[str, Any]] = []
        self._index: int = 0
        self._task_mapping: Optional[Dict[str, Any]] = None
        self._annotations: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _annotation_path(self) -> str:
        """Return the expected annotation JSON path."""
        # The annotations are stored under:
        #   {demo_data_path}/annotations/{task_name}/{demo_id}.json
        # Note: eval_subtask_reset.py uses task-{idx:04d} internally, but the
        # caller already provides the resolved task_name directory name.
        return os.path.join(
            self.demo_data_path,
            "annotations",
            self.task_name,
            f"{self.demo_id}.json",
        )

    def _load_task_mapping(self) -> Optional[Dict[str, Any]]:
        """Load the optional task_mapping.json if it exists."""
        path = os.environ.get("GTPLAN_TASK_MAPPING", self.DEFAULT_TASK_MAPPING_PATH)
        if not os.path.isfile(path):
            logger.debug("Task mapping file not found at %s", path)
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            logger.warning("Task mapping file at %s is not a JSON object", path)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse task mapping JSON at %s: %s", path, exc)
        except OSError as exc:
            logger.warning("Failed to read task mapping at %s: %s", path, exc)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_plan(self) -> List[Dict[str, Any]]:
        """Load and return the full skill plan.

        The plan is derived from the ``skill_annotation`` field in the demo
        annotation JSON.  Each skill dict is guaranteed to contain at least:

        - ``skill_description`` (str)
        - ``frame_duration`` (List[int, int])

        Skills are sorted by ``frame_duration[0]`` (start frame).

        Returns:
            A list of skill dictionaries.  If annotations are missing or empty,
            an empty list is returned.
        """
        annotation_path = self._annotation_path()
        if not os.path.isfile(annotation_path):
            logger.warning("Annotation file not found: %s", annotation_path)
            self._plan = []
            self._index = 0
            return list(self._plan)

        try:
            with open(annotation_path, "r", encoding="utf-8") as f:
                self._annotations = json.load(f)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse annotation JSON at %s: %s", annotation_path, exc)
            self._plan = []
            self._index = 0
            return list(self._plan)
        except OSError as exc:
            logger.warning("Failed to read annotation file at %s: %s", annotation_path, exc)
            self._plan = []
            self._index = 0
            return list(self._plan)

        raw_skills = self._annotations.get("skill_annotation", [])
        if not isinstance(raw_skills, list):
            logger.warning(
                "Expected 'skill_annotation' to be a list, got %s in %s",
                type(raw_skills).__name__,
                annotation_path,
            )
            self._plan = []
            self._index = 0
            return list(self._plan)

        if not raw_skills:
            logger.debug("Empty skill_annotation list in %s", annotation_path)
            self._plan = []
            self._index = 0
            return list(self._plan)

        normalized: List[Dict[str, Any]] = []
        for seg in raw_skills:
            if not isinstance(seg, dict):
                continue
            seg_copy = dict(seg)
            try:
                frame_duration = _normalize_frame_duration(seg_copy.get("frame_duration"))
            except ValueError as exc:
                logger.warning("Skipping skill with invalid frame_duration: %s", exc)
                continue
            seg_copy["frame_duration"] = frame_duration
            # Ensure skill_description is a string.
            desc = seg_copy.get("skill_description", "")
            if isinstance(desc, list) and desc:
                desc = str(desc[0])
            elif not isinstance(desc, str):
                desc = str(desc)
            seg_copy["skill_description"] = desc
            normalized.append(seg_copy)

        # Sort by start frame.
        normalized.sort(key=lambda x: x["frame_duration"][0])
        self._plan = normalized
        self._index = 0
        return list(self._plan)

    def get_current_skill(self) -> Optional[Dict[str, Any]]:
        """Return the current skill dict, or ``None`` if exhausted / not loaded."""
        if not self._plan or self._index < 0 or self._index >= len(self._plan):
            return None
        return dict(self._plan[self._index])

    def advance(self) -> bool:
        """Advance to the next skill.

        Returns:
            ``True`` if the advance succeeded (there was a next skill),
            ``False`` if already at or past the end.
        """
        if not self._plan:
            return False
        if self._index >= len(self._plan):
            return False
        self._index += 1
        return self._index < len(self._plan)

    def get_skill_prompt(self, skill_desc: str) -> str:
        """Map a skill description to a richer prompt when possible.

        The lookup order is:

        1. If a ``task_mapping.json`` is available and the current task has a
           ``skill`` list, iterate through the list and return the first entry
           that is a case-insensitive substring match (or exact match) of
           ``skill_desc``.
        2. Otherwise return ``skill_desc`` unchanged.

        Args:
            skill_desc: Raw skill description (e.g. ``"pick up from"``).

        Returns:
            A prompt string (either from the mapping or the original description).
        """
        if self._task_mapping is None:
            self._task_mapping = self._load_task_mapping()

        mapping = self._task_mapping
        if mapping is None:
            return skill_desc

        task_entry = mapping.get(self.task_name)
        if task_entry is None:
            return skill_desc

        skill_list = task_entry.get("skill")
        if not isinstance(skill_list, list):
            return skill_desc

        skill_desc_lower = skill_desc.lower().strip()

        # Pass 1: exact match (case-insensitive).
        for candidate in skill_list:
            if not isinstance(candidate, str):
                continue
            if candidate.lower().strip() == skill_desc_lower:
                return candidate

        # Pass 2: skill_desc is a substring of candidate (e.g. "pick up" -> "pick up from").
        for candidate in skill_list:
            if not isinstance(candidate, str):
                continue
            if skill_desc_lower in candidate.lower().strip():
                return candidate

        # Pass 3: candidate is a substring of skill_desc (e.g. "place on" inside "place on next to").
        for candidate in skill_list:
            if not isinstance(candidate, str):
                continue
            if candidate.lower().strip() in skill_desc_lower:
                return candidate

        return skill_desc

    def is_exhausted(self) -> bool:
        """Return ``True`` when all skills have been consumed."""
        if not self._plan:
            return True
        return self._index >= len(self._plan)

    def reset(self) -> None:
        """Reset the internal index to the first skill."""
        self._index = 0

    def __len__(self) -> int:
        """Return the number of skills in the loaded plan."""
        return len(self._plan)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"task_name={self.task_name!r}, "
            f"demo_id={self.demo_id!r}, "
            f"n_skills={len(self._plan)}, "
            f"index={self._index})"
        )
