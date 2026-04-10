"""Tests for 3-way train/val/test split and early stopping.

All tests avoid importing aevyra_verdict so they can run in the sandbox.
The _split_dataset_3way logic and result/config dataclasses are tested
directly via inline mirrors or by importing from the source tree.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aevyra_reflex.optimizer import OptimizerConfig, _EarlyStop
from aevyra_reflex.result import (
    EvalSnapshot,
    IterationRecord,
    OptimizationResult,
    SampleSnapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_dataset(n: int):
    """Minimal stand-in for a verdict Dataset with n conversations."""
    obj = MagicMock()
    obj.conversations = [MagicMock() for _ in range(n)]
    return obj


def _fake_dataset_cls(convos):
    """Return an object that looks like Dataset(conversations=convos)."""
    obj = MagicMock()
    obj.conversations = convos
    return obj


# Inline mirror of _split_dataset_3way so we can test the logic without
# importing aevyra_verdict (which calls Dataset() internally).
def _split_3way(convos, train_ratio, val_ratio, seed=42):
    import random

    n = len(convos)
    n_test = max(1, n - round(n * train_ratio))
    n_val = max(1, round(n * val_ratio)) if val_ratio > 0.0 else 0
    n_train = max(1, n - n_test - n_val)

    while n_train + n_val + n_test > n:
        n_train -= 1

    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    train = [convos[indices[i]] for i in range(n_train)]
    val = [convos[indices[i]] for i in range(n_train, n_train + n_val)]
    test = [convos[indices[i]] for i in range(n_train + n_val, n)]
    return train, val, test


# ===========================================================================
# OptimizerConfig defaults
# ===========================================================================

class TestOptimizerConfigValFields(unittest.TestCase):

    def test_val_ratio_default(self):
        c = OptimizerConfig()
        self.assertEqual(c.val_ratio, 0.1)

    def test_early_stopping_patience_default(self):
        c = OptimizerConfig()
        self.assertEqual(c.early_stopping_patience, 3)

    def test_val_ratio_custom(self):
        c = OptimizerConfig(val_ratio=0.1)
        self.assertEqual(c.val_ratio, 0.1)

    def test_early_stopping_patience_custom(self):
        c = OptimizerConfig(early_stopping_patience=3)
        self.assertEqual(c.early_stopping_patience, 3)


# ===========================================================================
# _split_dataset_3way logic
# ===========================================================================

class TestSplitDataset3Way(unittest.TestCase):

    def setUp(self):
        self.convos = list(range(100))

    def test_sizes_sum_to_n(self):
        train, val, test = _split_3way(self.convos, train_ratio=0.8, val_ratio=0.1)
        self.assertEqual(len(train) + len(val) + len(test), 100)

    def test_approximate_proportions(self):
        train, val, test = _split_3way(self.convos, train_ratio=0.8, val_ratio=0.1)
        # 70 train / 10 val / 20 test (approximately)
        self.assertAlmostEqual(len(train), 70, delta=2)
        self.assertAlmostEqual(len(val), 10, delta=2)
        self.assertAlmostEqual(len(test), 20, delta=2)

    def test_no_overlap_between_splits(self):
        train, val, test = _split_3way(self.convos, train_ratio=0.8, val_ratio=0.1)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(train) & set(test)), 0)
        self.assertEqual(len(set(val) & set(test)), 0)

    def test_deterministic_with_same_seed(self):
        a_train, a_val, a_test = _split_3way(self.convos, 0.8, 0.1, seed=42)
        b_train, b_val, b_test = _split_3way(self.convos, 0.8, 0.1, seed=42)
        self.assertEqual(a_train, b_train)
        self.assertEqual(a_val, b_val)
        self.assertEqual(a_test, b_test)

    def test_different_seeds_give_different_splits(self):
        a_train, _, _ = _split_3way(self.convos, 0.8, 0.1, seed=42)
        b_train, _, _ = _split_3way(self.convos, 0.8, 0.1, seed=99)
        self.assertNotEqual(a_train, b_train)

    def test_zero_val_ratio_gives_no_val(self):
        train, val, test = _split_3way(self.convos, train_ratio=0.8, val_ratio=0.0)
        self.assertEqual(len(val), 0)
        self.assertEqual(len(train) + len(test), 100)

    def test_small_dataset_at_least_one_in_each(self):
        convos = list(range(5))
        train, val, test = _split_3way(convos, train_ratio=0.8, val_ratio=0.2)
        self.assertGreaterEqual(len(train), 1)
        self.assertGreaterEqual(len(val), 1)
        self.assertGreaterEqual(len(test), 1)

    def test_test_size_unaffected_by_val(self):
        """test size should be the same whether val is enabled or not."""
        _, _, test_no_val = _split_3way(self.convos, train_ratio=0.8, val_ratio=0.0)
        _, _, test_with_val = _split_3way(self.convos, train_ratio=0.8, val_ratio=0.1)
        # Both should have ~20 test examples
        self.assertAlmostEqual(len(test_no_val), len(test_with_val), delta=1)


# ===========================================================================
# IterationRecord val_score field
# ===========================================================================

class TestIterationRecordValScore(unittest.TestCase):

    def test_val_score_defaults_to_none(self):
        r = IterationRecord(iteration=1, system_prompt="p", score=0.7)
        self.assertIsNone(r.val_score)

    def test_val_score_can_be_set(self):
        r = IterationRecord(iteration=1, system_prompt="p", score=0.7, val_score=0.65)
        self.assertEqual(r.val_score, 0.65)


# ===========================================================================
# OptimizationResult val fields
# ===========================================================================

class TestOptimizationResultValFields(unittest.TestCase):

    def _minimal_result(self, **kwargs):
        return OptimizationResult(
            best_prompt="p",
            best_score=0.8,
            iterations=[],
            converged=False,
            **kwargs,
        )

    def test_val_size_default_zero(self):
        r = self._minimal_result()
        self.assertEqual(r.val_size, 0)

    def test_val_trajectory_default_empty(self):
        r = self._minimal_result()
        self.assertEqual(r.val_trajectory, [])

    def test_early_stopped_default_false(self):
        r = self._minimal_result()
        self.assertFalse(r.early_stopped)

    def test_set_val_fields(self):
        r = self._minimal_result(
            val_size=10,
            val_trajectory=[0.6, 0.7, 0.68],
            early_stopped=True,
        )
        self.assertEqual(r.val_size, 10)
        self.assertEqual(r.val_trajectory, [0.6, 0.7, 0.68])
        self.assertTrue(r.early_stopped)


# ===========================================================================
# summary() output
# ===========================================================================

class TestSummaryValDisplay(unittest.TestCase):

    def _result_with_val(self, val_size=10, val_traj=None, early_stopped=False):
        baseline = EvalSnapshot(mean_score=0.5, samples=[
            SampleSnapshot(input="q", response="a", ideal="a", score=0.5),
        ])
        final = EvalSnapshot(mean_score=0.7, samples=[
            SampleSnapshot(input="q", response="b", ideal="b", score=0.7),
        ])
        return OptimizationResult(
            best_prompt="optimized",
            best_score=0.7,
            iterations=[
                IterationRecord(iteration=1, system_prompt="p", score=0.7, val_score=0.65),
            ],
            converged=False,
            baseline=baseline,
            final=final,
            train_size=70,
            test_size=20,
            val_size=val_size,
            val_trajectory=val_traj or [0.6, 0.65, 0.63],
            early_stopped=early_stopped,
        )

    def test_3way_split_displayed_in_summary(self):
        s = self._result_with_val().summary()
        self.assertIn("Train/val/test", s)
        self.assertIn("70", s)
        self.assertIn("10", s)
        self.assertIn("20", s)

    def test_val_trajectory_line_present(self):
        s = self._result_with_val(val_traj=[0.6, 0.65, 0.63]).summary()
        self.assertIn("Val traj", s)
        self.assertIn("0.600", s)

    def test_early_stopped_line_present(self):
        s = self._result_with_val(early_stopped=True).summary()
        self.assertIn("Early stopped", s)
        self.assertIn("Yes", s)

    def test_no_early_stopped_line_when_false(self):
        s = self._result_with_val(early_stopped=False).summary()
        self.assertNotIn("Early stopped", s)

    def test_train_traj_label(self):
        """Trajectory line should now say 'Train traj' when val is present."""
        s = self._result_with_val().summary()
        self.assertIn("Train traj", s)


# ===========================================================================
# to_dict() includes val fields
# ===========================================================================

class TestToDictValFields(unittest.TestCase):

    def test_to_dict_includes_val_fields_when_set(self):
        baseline = EvalSnapshot(mean_score=0.5)
        final = EvalSnapshot(mean_score=0.7)
        r = OptimizationResult(
            best_prompt="p", best_score=0.7, iterations=[], converged=False,
            baseline=baseline, final=final,
            train_size=70, test_size=20, val_size=10,
            val_trajectory=[0.6, 0.7], early_stopped=True,
        )
        d = r.to_dict()
        self.assertEqual(d["val_size"], 10)
        self.assertEqual(d["val_trajectory"], [0.6, 0.7])
        self.assertTrue(d["early_stopped"])

    def test_to_dict_omits_val_fields_when_zero(self):
        r = OptimizationResult(
            best_prompt="p", best_score=0.7, iterations=[], converged=False,
        )
        d = r.to_dict()
        self.assertNotIn("val_size", d)
        self.assertNotIn("val_trajectory", d)
        self.assertNotIn("early_stopped", d)

    def test_iteration_val_score_in_dict(self):
        r = OptimizationResult(
            best_prompt="p", best_score=0.7,
            iterations=[IterationRecord(1, "p", 0.7, val_score=0.65)],
            converged=False,
        )
        d = r.to_dict()
        self.assertEqual(d["iterations"][0]["val_score"], 0.65)

    def test_iteration_val_score_absent_when_none(self):
        r = OptimizationResult(
            best_prompt="p", best_score=0.7,
            iterations=[IterationRecord(1, "p", 0.7)],
            converged=False,
        )
        d = r.to_dict()
        self.assertNotIn("val_score", d["iterations"][0])


# ===========================================================================
# _EarlyStop exception
# ===========================================================================

class TestEarlyStopException(unittest.TestCase):

    def test_early_stop_is_exception(self):
        self.assertTrue(issubclass(_EarlyStop, Exception))

    def test_early_stop_can_be_raised_and_caught(self):
        with self.assertRaises(_EarlyStop):
            raise _EarlyStop()

    def test_early_stop_caught_as_exception(self):
        caught = False
        try:
            raise _EarlyStop()
        except Exception:
            caught = True
        self.assertTrue(caught)


# ===========================================================================
# Early stopping logic (unit test on the index-of-best logic)
# ===========================================================================

class TestEarlyStoppingLogic(unittest.TestCase):

    def _iters_since_best(self, val_history):
        """Replicate the early-stopping check from _checkpointing_callback."""
        best_val_overall = max(val_history)
        best_val_idx = len(val_history) - 1 - next(
            i for i, v in enumerate(reversed(val_history))
            if v == best_val_overall
        )
        return len(val_history) - 1 - best_val_idx

    def test_no_plateau_when_improving(self):
        self.assertEqual(self._iters_since_best([0.5, 0.6, 0.7, 0.8]), 0)

    def test_plateau_of_two(self):
        self.assertEqual(self._iters_since_best([0.5, 0.8, 0.75, 0.72]), 2)

    def test_plateau_of_three(self):
        self.assertEqual(self._iters_since_best([0.8, 0.75, 0.72, 0.71]), 3)

    def test_last_iter_is_best(self):
        self.assertEqual(self._iters_since_best([0.5, 0.6, 0.9]), 0)

    def test_ties_counted_from_most_recent(self):
        # Best score appears at index 1 and 3; most-recent best is at 3 → 0 since best
        self.assertEqual(self._iters_since_best([0.5, 0.9, 0.8, 0.9]), 0)


if __name__ == "__main__":
    unittest.main()
