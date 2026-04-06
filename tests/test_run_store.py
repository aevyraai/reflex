# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for run store — checkpointing, resume, and multi-run versioning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aevyra_reflex.run_store import (
    CheckpointState,
    IterationState,
    Run,
    RunStore,
)


class TestRunStore(unittest.TestCase):
    """Test RunStore directory management."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=Path(self.tmpdir) / ".reflex")

    def test_create_run_makes_directories(self):
        run = self.store.create_run(
            config={"strategy": "auto"},
            dataset_path="data.jsonl",
            prompt_path="prompt.md",
            initial_prompt="You are helpful.",
        )
        self.assertTrue(run.run_dir.exists())
        self.assertTrue(run.iterations_dir.exists())
        self.assertTrue(run.config_path.exists())

    def test_create_run_sequential_ids(self):
        run1 = self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="a",
        )
        run2 = self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="b",
        )
        run3 = self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="c",
        )
        self.assertEqual(run1.run_id, "001")
        self.assertEqual(run2.run_id, "002")
        self.assertEqual(run3.run_id, "003")

    def test_get_run_by_id(self):
        created = self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="a",
        )
        fetched = self.store.get_run("001")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.run_id, "001")
        self.assertEqual(fetched.run_dir, created.run_dir)

    def test_get_run_not_found(self):
        self.assertIsNone(self.store.get_run("999"))

    def test_get_latest_run(self):
        self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="a",
        )
        run2 = self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="b",
        )
        latest = self.store.get_latest_run()
        self.assertEqual(latest.run_id, run2.run_id)

    def test_get_latest_run_empty(self):
        self.assertIsNone(self.store.get_latest_run())

    def test_list_runs_empty(self):
        self.assertEqual(self.store.list_runs(), [])

    def test_list_runs_with_data(self):
        self.store.create_run(
            config={"strategy": "iterative"},
            dataset_path="d.jsonl",
            prompt_path="p.md",
            initial_prompt="a",
        )
        self.store.create_run(
            config={"strategy": "pdo"},
            dataset_path="d.jsonl",
            prompt_path="p.md",
            initial_prompt="b",
        )
        summaries = self.store.list_runs()
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0].run_id, "001")
        self.assertEqual(summaries[1].run_id, "002")


class TestRun(unittest.TestCase):
    """Test Run persistence operations."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=Path(self.tmpdir) / ".reflex")
        self.run = self.store.create_run(
            config={"strategy": "iterative", "max_iterations": 10},
            dataset_path="data.jsonl",
            prompt_path="prompt.md",
            initial_prompt="You are helpful.",
        )

    def test_config_saved_and_loadable(self):
        config = self.run.load_config()
        self.assertEqual(config["dataset_path"], "data.jsonl")
        self.assertEqual(config["initial_prompt"], "You are helpful.")
        self.assertEqual(config["optimizer_config"]["strategy"], "iterative")

    def test_save_and_load_baseline(self):
        self.run.save_baseline({"mean_score": 0.58, "scores_by_metric": {"rouge": 0.58}})
        baseline = self.run.load_baseline()
        self.assertAlmostEqual(baseline["mean_score"], 0.58)
        self.assertIn("timestamp", baseline)

    def test_load_baseline_before_save(self):
        self.assertIsNone(self.run.load_baseline())

    def test_save_and_load_iteration(self):
        state = IterationState(
            iteration=1,
            system_prompt="You are very helpful.",
            score=0.65,
            scores_by_metric={"rouge": 0.65},
            reasoning="Added specificity.",
        )
        self.run.save_iteration(state)

        iterations = self.run.load_iterations()
        self.assertEqual(len(iterations), 1)
        self.assertEqual(iterations[0].iteration, 1)
        self.assertAlmostEqual(iterations[0].score, 0.65)
        self.assertEqual(iterations[0].reasoning, "Added specificity.")

    def test_multiple_iterations_ordered(self):
        for i in range(1, 4):
            self.run.save_iteration(IterationState(
                iteration=i,
                system_prompt=f"prompt v{i}",
                score=0.5 + i * 0.1,
            ))

        iterations = self.run.load_iterations()
        self.assertEqual(len(iterations), 3)
        self.assertEqual([it.iteration for it in iterations], [1, 2, 3])
        self.assertAlmostEqual(iterations[2].score, 0.8)

    def test_save_and_load_checkpoint(self):
        cp = CheckpointState(
            run_id="001",
            initial_prompt="You are helpful.",
            current_prompt="You are very helpful and precise.",
            completed_iterations=3,
            best_prompt="You are very helpful and precise.",
            best_score=0.78,
            score_trajectory=[0.62, 0.71, 0.78],
            previous_reasoning="Focus on precision.",
            baseline={"mean_score": 0.58},
        )
        self.run.save_checkpoint(cp)

        loaded = self.run.load_checkpoint()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.run_id, "001")
        self.assertEqual(loaded.completed_iterations, 3)
        self.assertAlmostEqual(loaded.best_score, 0.78)
        self.assertEqual(loaded.score_trajectory, [0.62, 0.71, 0.78])
        self.assertEqual(loaded.previous_reasoning, "Focus on precision.")
        self.assertAlmostEqual(loaded.baseline["mean_score"], 0.58)

    def test_checkpoint_updates_best_prompt_file(self):
        cp = CheckpointState(
            run_id="001",
            initial_prompt="original",
            current_prompt="improved",
            completed_iterations=1,
            best_prompt="improved",
            best_score=0.75,
        )
        self.run.save_checkpoint(cp)

        self.assertTrue(self.run.best_prompt_path.exists())
        self.assertEqual(self.run.best_prompt_path.read_text(), "improved")

    def test_load_checkpoint_before_save(self):
        self.assertIsNone(self.run.load_checkpoint())

    def test_save_and_load_result(self):
        result = {
            "best_prompt": "optimized prompt",
            "best_score": 0.86,
            "iterations": [{"iteration": 1, "score": 0.86}],
            "converged": True,
        }
        self.run.save_result(result)

        loaded = self.run.load_result()
        self.assertEqual(loaded["best_score"], 0.86)
        self.assertEqual(loaded["run_id"], "001")
        self.assertTrue(self.run.is_complete)

    def test_is_complete_before_result(self):
        self.assertFalse(self.run.is_complete)

    def test_has_checkpoint(self):
        self.assertFalse(self.run.has_checkpoint)
        self.run.save_checkpoint(CheckpointState(
            run_id="001", initial_prompt="a", current_prompt="b",
            completed_iterations=1, best_prompt="b", best_score=0.5,
        ))
        self.assertTrue(self.run.has_checkpoint)


class TestFindIncompleteRun(unittest.TestCase):
    """Test finding interrupted runs for resume."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=Path(self.tmpdir) / ".reflex")

    def test_finds_interrupted_run(self):
        run = self.store.create_run(
            config={"strategy": "auto"},
            dataset_path="data.jsonl",
            prompt_path="p.md",
            initial_prompt="a",
        )
        # Simulate interrupt: checkpoint exists, no result
        run.save_checkpoint(CheckpointState(
            run_id=run.run_id, initial_prompt="a", current_prompt="b",
            completed_iterations=3, best_prompt="b", best_score=0.7,
        ))

        found = self.store.find_incomplete_run()
        self.assertIsNotNone(found)
        self.assertEqual(found.run_id, run.run_id)

    def test_skips_completed_run(self):
        run = self.store.create_run(
            config={}, dataset_path="data.jsonl",
            prompt_path="p.md", initial_prompt="a",
        )
        run.save_checkpoint(CheckpointState(
            run_id=run.run_id, initial_prompt="a", current_prompt="b",
            completed_iterations=3, best_prompt="b", best_score=0.7,
        ))
        run.save_result({"best_score": 0.86})

        found = self.store.find_incomplete_run()
        self.assertIsNone(found)

    def test_finds_latest_interrupted(self):
        # Run 1: completed
        run1 = self.store.create_run(
            config={}, dataset_path="data.jsonl",
            prompt_path="p.md", initial_prompt="a",
        )
        run1.save_checkpoint(CheckpointState(
            run_id=run1.run_id, initial_prompt="a", current_prompt="b",
            completed_iterations=5, best_prompt="b", best_score=0.8,
        ))
        run1.save_result({"best_score": 0.8})

        # Run 2: interrupted
        run2 = self.store.create_run(
            config={}, dataset_path="data.jsonl",
            prompt_path="p.md", initial_prompt="c",
        )
        run2.save_checkpoint(CheckpointState(
            run_id=run2.run_id, initial_prompt="c", current_prompt="d",
            completed_iterations=2, best_prompt="d", best_score=0.6,
        ))

        found = self.store.find_incomplete_run()
        self.assertEqual(found.run_id, run2.run_id)

    def test_filter_by_dataset(self):
        run1 = self.store.create_run(
            config={}, dataset_path="other.jsonl",
            prompt_path="p.md", initial_prompt="a",
        )
        run1.save_checkpoint(CheckpointState(
            run_id=run1.run_id, initial_prompt="a", current_prompt="b",
            completed_iterations=2, best_prompt="b", best_score=0.6,
        ))

        # Should not find it when filtering by different dataset
        found = self.store.find_incomplete_run(dataset_path="data.jsonl")
        self.assertIsNone(found)

        # Should find it with matching dataset
        found = self.store.find_incomplete_run(dataset_path="other.jsonl")
        self.assertIsNotNone(found)


class TestAtomicWrite(unittest.TestCase):
    """Test that writes are atomic (temp file + rename)."""

    def test_no_tmp_files_left(self):
        tmpdir = tempfile.mkdtemp()
        store = RunStore(root=Path(tmpdir) / ".reflex")
        run = store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="a",
        )
        run.save_baseline({"mean_score": 0.5})
        run.save_iteration(IterationState(
            iteration=1, system_prompt="p", score=0.6,
        ))
        run.save_checkpoint(CheckpointState(
            run_id="001", initial_prompt="a", current_prompt="p",
            completed_iterations=1, best_prompt="p", best_score=0.6,
        ))

        # No .tmp files should remain
        tmp_files = list(run.run_dir.rglob("*.tmp"))
        self.assertEqual(tmp_files, [])


class TestBranchedRuns(unittest.TestCase):
    """Test branched_from support in create_run() and list_runs()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=Path(self.tmpdir) / ".reflex")

    def test_create_run_without_branched_from(self):
        run = self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="a",
        )
        config = run.load_config()
        self.assertNotIn("branched_from", config)

    def test_create_run_with_branched_from_saved_to_config(self):
        run = self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="a",
            branched_from={"run_id": "003", "iteration": 5},
        )
        config = run.load_config()
        self.assertIn("branched_from", config)
        self.assertEqual(config["branched_from"]["run_id"], "003")
        self.assertEqual(config["branched_from"]["iteration"], 5)

    def test_list_runs_populates_parent_fields(self):
        # Root run
        self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="a",
        )
        # Branch run
        self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="b",
            branched_from={"run_id": "001", "iteration": 3},
        )

        summaries = self.store.list_runs()
        root = summaries[0]
        branch = summaries[1]

        self.assertIsNone(root.parent_run_id)
        self.assertIsNone(root.parent_iteration)

        self.assertEqual(branch.parent_run_id, "001")
        self.assertEqual(branch.parent_iteration, 3)

    def test_list_runs_no_parent_fields_without_branched_from(self):
        self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="a",
        )
        summaries = self.store.list_runs()
        self.assertIsNone(summaries[0].parent_run_id)
        self.assertIsNone(summaries[0].parent_iteration)


class TestIterationTokenFields(unittest.TestCase):
    """Test eval_tokens / reasoning_tokens on IterationState."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=Path(self.tmpdir) / ".reflex")
        self.run = self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="a",
        )

    def test_default_token_fields_are_zero(self):
        state = IterationState(iteration=1, system_prompt="p", score=0.5)
        self.assertEqual(state.eval_tokens, 0)
        self.assertEqual(state.reasoning_tokens, 0)

    def test_token_fields_saved_and_loaded(self):
        state = IterationState(
            iteration=1, system_prompt="p", score=0.7,
            eval_tokens=8500, reasoning_tokens=3200,
        )
        self.run.save_iteration(state)

        loaded = self.run.load_iterations()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].eval_tokens, 8500)
        self.assertEqual(loaded[0].reasoning_tokens, 3200)

    def test_token_fields_default_to_zero_when_missing_from_file(self):
        # Write an iteration file without token fields (simulates old runs)
        iter_file = self.run.iterations_dir / "001.json"
        iter_file.write_text(json.dumps({
            "iteration": 1, "system_prompt": "p", "score": 0.5,
            "scores_by_metric": {}, "reasoning": "", "timestamp": "",
        }))

        loaded = self.run.load_iterations()
        self.assertEqual(loaded[0].eval_tokens, 0)
        self.assertEqual(loaded[0].reasoning_tokens, 0)


class TestRunLog(unittest.TestCase):
    """Test save_log() / load_log()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=Path(self.tmpdir) / ".reflex")
        self.run = self.store.create_run(
            config={}, dataset_path="d.jsonl",
            prompt_path="p.md", initial_prompt="a",
        )

    def test_load_log_before_save_returns_empty(self):
        self.assertEqual(self.run.load_log(), [])

    def test_save_and_load_log(self):
        entries = [
            {"type": "queued", "ts": "2026-04-06T10:00:00+00:00", "message": "Starting…"},
            {"type": "iteration", "ts": "2026-04-06T10:00:10+00:00", "iteration": 1, "score": 0.72},
            {"type": "complete", "ts": "2026-04-06T10:01:00+00:00", "best_score": 0.86},
        ]
        self.run.save_log(entries)

        loaded = self.run.load_log()
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[0]["type"], "queued")
        self.assertEqual(loaded[1]["iteration"], 1)
        self.assertAlmostEqual(loaded[1]["score"], 0.72)
        self.assertEqual(loaded[2]["best_score"], 0.86)

    def test_save_log_overwrites_previous(self):
        self.run.save_log([{"type": "queued", "ts": "t1", "message": "first"}])
        self.run.save_log([{"type": "complete", "ts": "t2", "best_score": 0.9}])

        loaded = self.run.load_log()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["type"], "complete")

    def test_save_log_empty(self):
        self.run.save_log([])
        loaded = self.run.load_log()
        self.assertEqual(loaded, [])

    def test_log_file_written_atomically(self):
        # After save, no .tmp file should remain
        self.run.save_log([{"type": "queued", "ts": "t", "message": "x"}])
        tmp_files = list(self.run.run_dir.glob("*.tmp"))
        self.assertEqual(tmp_files, [])

    def test_log_path_property(self):
        self.assertEqual(self.run.log_path, self.run.run_dir / "run.log.jsonl")


if __name__ == "__main__":
    unittest.main()
