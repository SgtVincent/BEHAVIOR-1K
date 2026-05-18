"""Unit tests for GTPlanLoader.

All filesystem dependencies (annotation JSON and task_mapping JSON) are mocked
so the tests run without requiring the full BEHAVIOR-1K dataset.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from omnigibson.learning.gt_plan_loader import GTPlanLoader, _flatten_numeric, _normalize_frame_duration


class TestNormalizeHelpers(unittest.TestCase):
    """Tests for the small pure helper functions."""

    def test_flatten_numeric_nested(self) -> None:
        self.assertEqual(_flatten_numeric([1, [2, 3], (4, [5, 6])]), [1, 2, 3, 4, 5, 6])

    def test_flatten_numeric_scalar(self) -> None:
        self.assertEqual(_flatten_numeric(42), [42])

    def test_flatten_numeric_float(self) -> None:
        self.assertEqual(_flatten_numeric(3.7), [3])

    def test_normalize_frame_duration_ok(self) -> None:
        self.assertEqual(_normalize_frame_duration([10, 20]), [10, 20])

    def test_normalize_frame_duration_nested(self) -> None:
        self.assertEqual(_normalize_frame_duration([[5], [15]]), [5, 15])

    def test_normalize_frame_duration_invalid(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_frame_duration([5])


class TestGTPlanLoader(unittest.TestCase):
    """Tests for GTPlanLoader with mocked filesystem."""

    def _make_loader(
        self,
        demo_data_path: str,
        task_name: str = "turning_on_radio",
        demo_id: str = "00000010",
    ) -> GTPlanLoader:
        return GTPlanLoader(demo_data_path=demo_data_path, task_name=task_name, demo_id=demo_id)

    def _write_annotation(
        self,
        root: str,
        task_name: str,
        demo_id: str,
        data: Dict[str, Any],
    ) -> str:
        """Write an annotation JSON file to the expected path and return its path."""
        dir_path = os.path.join(root, "annotations", task_name)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, f"{demo_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    # ------------------------------------------------------------------
    # Normal path
    # ------------------------------------------------------------------

    def test_load_plan_basic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {
                        "skill_description": "move to",
                        "frame_duration": [30, 60],
                        "extra_field": 1,
                    },
                    {
                        "skill_description": "pick up from",
                        "frame_duration": [0, 30],
                        "extra_field": 2,
                    },
                ],
                "primitive_annotation": [],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            loader = self._make_loader(tmpdir)
            plan = loader.load_plan()

            self.assertEqual(len(plan), 2)
            # Should be sorted by frame_duration[0]
            self.assertEqual(plan[0]["skill_description"], "pick up from")
            self.assertEqual(plan[0]["frame_duration"], [0, 30])
            self.assertEqual(plan[1]["skill_description"], "move to")
            self.assertEqual(plan[1]["frame_duration"], [30, 60])

    def test_get_current_skill_and_advance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "a", "frame_duration": [0, 10]},
                    {"skill_description": "b", "frame_duration": [10, 20]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            loader = self._make_loader(tmpdir)
            loader.load_plan()

            self.assertFalse(loader.is_exhausted())
            skill = loader.get_current_skill()
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertEqual(skill["skill_description"], "a")

            ok = loader.advance()
            self.assertTrue(ok)
            skill = loader.get_current_skill()
            self.assertIsNotNone(skill)
            assert skill is not None
            self.assertEqual(skill["skill_description"], "b")

            ok = loader.advance()
            self.assertFalse(ok)
            self.assertTrue(loader.is_exhausted())
            self.assertIsNone(loader.get_current_skill())

    def test_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "a", "frame_duration": [0, 10]},
                    {"skill_description": "b", "frame_duration": [10, 20]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            loader = self._make_loader(tmpdir)
            loader.load_plan()
            loader.advance()
            self.assertEqual(loader.get_current_skill()["skill_description"], "b")  # type: ignore[index]

            loader.reset()
            self.assertEqual(loader.get_current_skill()["skill_description"], "a")  # type: ignore[index]
            self.assertFalse(loader.is_exhausted())

    # ------------------------------------------------------------------
    # Edge cases: missing / empty annotations
    # ------------------------------------------------------------------

    def test_missing_annotation_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = self._make_loader(tmpdir)
            plan = loader.load_plan()
            self.assertEqual(plan, [])
            self.assertTrue(loader.is_exhausted())
            self.assertIsNone(loader.get_current_skill())

    def test_empty_skill_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {"skill_annotation": [], "primitive_annotation": []}
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            loader = self._make_loader(tmpdir)
            plan = loader.load_plan()
            self.assertEqual(plan, [])
            self.assertTrue(loader.is_exhausted())

    def test_missing_skill_annotation_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {"primitive_annotation": []}
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            loader = self._make_loader(tmpdir)
            plan = loader.load_plan()
            self.assertEqual(plan, [])

    def test_skill_description_as_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": ["pick up from"], "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            loader = self._make_loader(tmpdir)
            plan = loader.load_plan()
            self.assertEqual(plan[0]["skill_description"], "pick up from")

    def test_invalid_frame_duration_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "bad", "frame_duration": [5]},
                    {"skill_description": "good", "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            loader = self._make_loader(tmpdir)
            plan = loader.load_plan()
            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0]["skill_description"], "good")

    def test_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = os.path.join(tmpdir, "annotations", "turning_on_radio")
            os.makedirs(dir_path, exist_ok=True)
            path = os.path.join(dir_path, "00000010.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("not json")

            loader = self._make_loader(tmpdir)
            plan = loader.load_plan()
            self.assertEqual(plan, [])

    def test_non_dict_skill_entries_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    "not-a-dict",
                    {"skill_description": "valid", "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            loader = self._make_loader(tmpdir)
            plan = loader.load_plan()
            self.assertEqual(len(plan), 1)
            self.assertEqual(plan[0]["skill_description"], "valid")

    # ------------------------------------------------------------------
    # get_skill_prompt
    # ------------------------------------------------------------------

    def test_get_skill_prompt_with_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "pick up from", "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            mapping = {
                "turning_on_radio": {
                    "task": "Turn on the radio...",
                    "skill": ["press", "move to", "place on", "pick up from"],
                },
            }
            loader = self._make_loader(tmpdir)
            loader.load_plan()

            with patch.object(loader, "_load_task_mapping", return_value=mapping):
                prompt = loader.get_skill_prompt("pick up from")
                self.assertEqual(prompt, "pick up from")

    def test_get_skill_prompt_fallback_to_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "unknown skill", "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            mapping = {
                "turning_on_radio": {
                    "skill": ["press", "move to"],
                },
            }
            loader = self._make_loader(tmpdir)
            loader.load_plan()

            with patch.object(loader, "_load_task_mapping", return_value=mapping):
                prompt = loader.get_skill_prompt("unknown skill")
                self.assertEqual(prompt, "unknown skill")

    def test_get_skill_prompt_missing_task_in_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "move to", "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "some_other_task", "00000010", annotation)

            mapping = {"turning_on_radio": {"skill": ["move to"]}}
            loader = GTPlanLoader(tmpdir, "some_other_task", "00000010")
            loader.load_plan()

            with patch.object(loader, "_load_task_mapping", return_value=mapping):
                prompt = loader.get_skill_prompt("move to")
                self.assertEqual(prompt, "move to")

    def test_get_skill_prompt_no_mapping_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "move to", "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            loader = self._make_loader(tmpdir)
            loader.load_plan()

            with patch.object(loader, "_load_task_mapping", return_value=None):
                prompt = loader.get_skill_prompt("move to")
                self.assertEqual(prompt, "move to")

    def test_get_skill_prompt_caches_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "move to", "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            mapping = {"turning_on_radio": {"skill": ["move to"]}}
            loader = self._make_loader(tmpdir)
            loader.load_plan()

            with patch.object(loader, "_load_task_mapping", return_value=mapping) as mock_load:
                _ = loader.get_skill_prompt("move to")
                _ = loader.get_skill_prompt("move to")
                mock_load.assert_called_once()

    def test_get_skill_prompt_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "Pick Up From", "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            mapping = {
                "turning_on_radio": {
                    "skill": ["pick up from"],
                },
            }
            loader = self._make_loader(tmpdir)
            loader.load_plan()

            with patch.object(loader, "_load_task_mapping", return_value=mapping):
                prompt = loader.get_skill_prompt("Pick Up From")
                self.assertEqual(prompt, "pick up from")

    def test_get_skill_prompt_substring_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "place on next to", "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            mapping = {
                "turning_on_radio": {
                    "skill": ["place on", "place on next to"],
                },
            }
            loader = self._make_loader(tmpdir)
            loader.load_plan()

            with patch.object(loader, "_load_task_mapping", return_value=mapping):
                # Exact match for the longer candidate
                prompt = loader.get_skill_prompt("place on next to")
                self.assertEqual(prompt, "place on next to")

                # The description contains the shorter candidate, but exact match wins.
                prompt2 = loader.get_skill_prompt("place on")
                self.assertEqual(prompt2, "place on")

    def test_get_skill_prompt_desc_in_candidate(self) -> None:
        # When skill_desc is a substring of a candidate, prefer the longer candidate.
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "pick up", "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            mapping = {
                "turning_on_radio": {
                    "skill": ["pick up from"],
                },
            }
            loader = self._make_loader(tmpdir)
            loader.load_plan()

            with patch.object(loader, "_load_task_mapping", return_value=mapping):
                prompt = loader.get_skill_prompt("pick up")
                self.assertEqual(prompt, "pick up from")

    # ------------------------------------------------------------------
    # Len / repr
    # ------------------------------------------------------------------

    def test_len(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "a", "frame_duration": [0, 10]},
                    {"skill_description": "b", "frame_duration": [10, 20]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            loader = self._make_loader(tmpdir)
            self.assertEqual(len(loader), 0)
            loader.load_plan()
            self.assertEqual(len(loader), 2)

    def test_repr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation = {
                "skill_annotation": [
                    {"skill_description": "a", "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            loader = self._make_loader(tmpdir)
            loader.load_plan()
            r = repr(loader)
            self.assertIn("GTPlanLoader", r)
            self.assertIn("n_skills=1", r)

    # ------------------------------------------------------------------
    # Environment variable override for task mapping path
    # ------------------------------------------------------------------

    def test_task_mapping_path_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mapping_path = os.path.join(tmpdir, "custom_mapping.json")
            mapping = {
                "turning_on_radio": {
                    "skill": ["custom_prompt"],
                },
            }
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f)

            annotation = {
                "skill_annotation": [
                    {"skill_description": "custom_prompt", "frame_duration": [0, 10]},
                ],
            }
            self._write_annotation(tmpdir, "turning_on_radio", "00000010", annotation)

            loader = self._make_loader(tmpdir)
            loader.load_plan()

            with patch.dict(os.environ, {"GTPLAN_TASK_MAPPING": mapping_path}):
                # Force reload by clearing cached mapping
                loader._task_mapping = None
                prompt = loader.get_skill_prompt("custom_prompt")
                self.assertEqual(prompt, "custom_prompt")


if __name__ == "__main__":
    unittest.main()
