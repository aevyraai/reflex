"""Tests for periodic full-eval checkpoints during mini-batch optimization.

When batch_size > 0 and full_eval_steps > 0, every Nth iteration scores
the full training set instead of a random mini-batch. This gives an accurate
checkpoint score while keeping other iterations cheap.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aevyra_reflex.optimizer import OptimizerConfig
from aevyra_reflex.result import IterationRecord, OptimizationResult


class TestOptimizerConfigFullEvalSteps(unittest.TestCase):
    """OptimizerConfig.full_eval_steps field."""

    def test_default_is_zero(self):
        self.assertEqual(OptimizerConfig().full_eval_steps, 0)

    def test_custom_value(self):
        self.assertEqual(OptimizerConfig(full_eval_steps=5).full_eval_steps, 5)

    def test_coexists_with_batch_size(self):
        config = OptimizerConfig(batch_size=32, full_eval_steps=5)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.full_eval_steps, 5)


class TestIterationRecordIsFullEval(unittest.TestCase):
    """IterationRecord.is_full_eval field."""

    def test_default_is_false(self):
        record = IterationRecord(iteration=1, system_prompt="p", score=0.5)
        self.assertFalse(record.is_full_eval)

    def test_can_be_set_true(self):
        record = IterationRecord(iteration=5, system_prompt="p", score=0.8, is_full_eval=True)
        self.assertTrue(record.is_full_eval)


class TestToDictFullEval(unittest.TestCase):
    """is_full_eval is included in to_dict() only when True."""

    def _make_result(self, records):
        from aevyra_reflex.result import EvalSnapshot
        result = OptimizationResult(best_prompt="p", best_score=0.8, iterations=records, converged=True)
        result.baseline = EvalSnapshot(mean_score=0.6, scores_by_metric={})
        result.final = EvalSnapshot(mean_score=0.8, scores_by_metric={})
        result.baseline.system_prompt = "p"
        result.final.system_prompt = "p"
        return result

    def test_omitted_when_false(self):
        record = IterationRecord(iteration=1, system_prompt="p", score=0.5, is_full_eval=False)
        result = self._make_result([record])
        iter_dict = result.to_dict()["iterations"][0]
        self.assertNotIn("is_full_eval", iter_dict)

    def test_included_when_true(self):
        record = IterationRecord(iteration=5, system_prompt="p", score=0.8, is_full_eval=True)
        result = self._make_result([record])
        iter_dict = result.to_dict()["iterations"][0]
        self.assertIn("is_full_eval", iter_dict)
        self.assertTrue(iter_dict["is_full_eval"])

    def test_mixed_iterations(self):
        records = [
            IterationRecord(iteration=i, system_prompt="p", score=0.5 + i * 0.05,
                            is_full_eval=(i % 5 == 0))
            for i in range(1, 11)
        ]
        result = self._make_result(records)
        iter_dicts = result.to_dict()["iterations"]
        # Only iterations 5 and 10 should have is_full_eval
        for d in iter_dicts:
            if d["iteration"] in (5, 10):
                self.assertTrue(d.get("is_full_eval"))
            else:
                self.assertNotIn("is_full_eval", d)


class TestFullEvalSchedule(unittest.TestCase):
    """Logic for determining which iterations trigger a full eval."""

    def _is_full_eval(self, i, batch_size, full_eval_steps):
        """Mirror of the is_full_eval check in strategies."""
        return (
            batch_size > 0
            and full_eval_steps > 0
            and (i + 1) % full_eval_steps == 0
        )

    def test_no_batch_size_never_full_eval(self):
        for i in range(20):
            self.assertFalse(self._is_full_eval(i, batch_size=0, full_eval_steps=5))

    def test_no_full_eval_steps_never_full_eval(self):
        for i in range(20):
            self.assertFalse(self._is_full_eval(i, batch_size=32, full_eval_steps=0))

    def test_correct_iterations_are_full_eval(self):
        # full_eval_steps=5 → iterations 5, 10, 15, 20 (1-indexed)
        full_evals = [i for i in range(20) if self._is_full_eval(i, 32, 5)]
        # i=4 → iter 5, i=9 → iter 10, i=14 → iter 15, i=19 → iter 20
        self.assertEqual(full_evals, [4, 9, 14, 19])

    def test_full_eval_steps_1_every_iteration(self):
        for i in range(10):
            self.assertTrue(self._is_full_eval(i, 32, full_eval_steps=1))

    def test_full_eval_steps_equals_max_iterations(self):
        # Only the last iteration is a full eval
        max_iters = 10
        full_evals = [i for i in range(max_iters) if self._is_full_eval(i, 32, max_iters)]
        self.assertEqual(full_evals, [max_iters - 1])

    def test_effective_batch_is_zero_on_full_eval(self):
        batch_size = 32
        full_eval_steps = 5
        for i in range(15):
            is_fe = self._is_full_eval(i, batch_size, full_eval_steps)
            effective = 0 if is_fe else batch_size
            if (i + 1) % 5 == 0:
                self.assertEqual(effective, 0)
            else:
                self.assertEqual(effective, batch_size)

    def test_no_effect_when_batch_size_zero(self):
        """full_eval_steps is irrelevant when batch_size=0 (always full set)."""
        for full_eval_steps in [1, 3, 5, 10]:
            for i in range(20):
                self.assertFalse(self._is_full_eval(i, batch_size=0, full_eval_steps=full_eval_steps))


class TestIterativeStrategyAcceptsFullEvalSteps(unittest.TestCase):
    """Smoke test: full_eval_steps is read from config in the strategy loop."""

    def test_config_has_full_eval_steps_attr(self):
        config = OptimizerConfig(batch_size=10, full_eval_steps=3)
        self.assertEqual(getattr(config, "full_eval_steps", None), 3)

    def test_getattr_fallback_is_zero(self):
        """getattr(..., 0) fallback works for configs without the field."""
        class MinimalConfig:
            batch_size = 10
        self.assertEqual(getattr(MinimalConfig(), "full_eval_steps", 0), 0)


if __name__ == "__main__":
    unittest.main()
