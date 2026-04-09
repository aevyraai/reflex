"""Tests for mini-batch sampling during optimization iterations.

All tests avoid importing aevyra_verdict so they can run in the sandbox.
The batch-sampling logic is tested via the _run_eval function signature and
the OptimizerConfig / OptimizationResult dataclasses.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aevyra_reflex.optimizer import OptimizerConfig
from aevyra_reflex.result import OptimizationResult


# ---------------------------------------------------------------------------
# Inline mirror of the mini-batch sampling logic from _run_eval
# (avoids importing aevyra_verdict)
# ---------------------------------------------------------------------------

def _sample_batch(conversations, batch_size: int, iteration_seed: int) -> list:
    """Mirror of the sampling logic inside _run_eval."""
    import random as _random
    all_convos = list(conversations)
    if batch_size > 0 and batch_size < len(all_convos):
        rng = _random.Random(iteration_seed)
        indices = sorted(rng.sample(range(len(all_convos)), batch_size))
        return [all_convos[i] for i in indices]
    return all_convos


class TestOptimizerConfigBatchFields(unittest.TestCase):
    """OptimizerConfig gains batch_size and batch_seed fields."""

    def test_batch_size_default(self):
        config = OptimizerConfig()
        self.assertEqual(config.batch_size, 0)

    def test_batch_seed_default(self):
        config = OptimizerConfig()
        self.assertEqual(config.batch_seed, 42)

    def test_batch_size_custom(self):
        config = OptimizerConfig(batch_size=16)
        self.assertEqual(config.batch_size, 16)

    def test_batch_seed_custom(self):
        config = OptimizerConfig(batch_seed=99)
        self.assertEqual(config.batch_seed, 99)

    def test_batch_size_zero_is_full_set(self):
        """batch_size=0 is the sentinel for 'full dataset'."""
        config = OptimizerConfig(batch_size=0)
        self.assertEqual(config.batch_size, 0)


class TestOptimizationResultBatchField(unittest.TestCase):
    """OptimizationResult stores the batch_size used for the run."""

    def test_batch_size_default(self):
        result = OptimizationResult(best_prompt="p", best_score=0.5, iterations=[], converged=False)
        self.assertEqual(result.batch_size, 0)

    def test_batch_size_set(self):
        result = OptimizationResult(best_prompt="p", best_score=0.5, iterations=[], converged=False)
        result.batch_size = 32
        self.assertEqual(result.batch_size, 32)

    def test_to_dict_omits_batch_size_when_zero(self):
        result = OptimizationResult(best_prompt="p", best_score=0.5, iterations=[], converged=False)
        d = result.to_dict()
        self.assertNotIn("batch_size", d)

    def test_to_dict_includes_batch_size_when_set(self):
        result = OptimizationResult(best_prompt="p", best_score=0.5, iterations=[], converged=False)
        result.batch_size = 16
        d = result.to_dict()
        self.assertIn("batch_size", d)
        self.assertEqual(d["batch_size"], 16)


class TestSummaryBatchDisplay(unittest.TestCase):
    """summary() includes a batch size line when batch_size > 0."""

    def _make_result(self, batch_size=0, train_size=0, test_size=0):
        from aevyra_reflex.result import EvalSnapshot
        result = OptimizationResult(best_prompt="p", best_score=0.8, iterations=[], converged=True)
        result.baseline = EvalSnapshot(mean_score=0.6, scores_by_metric={})
        result.final = EvalSnapshot(mean_score=0.8, scores_by_metric={})
        result.baseline.system_prompt = "p"
        result.final.system_prompt = "p"
        result.batch_size = batch_size
        result.train_size = train_size
        result.test_size = test_size
        return result

    def test_no_batch_size_line_when_zero(self):
        result = self._make_result(batch_size=0)
        summary = result.summary()
        self.assertNotIn("Batch size", summary)

    def test_batch_size_line_when_set(self):
        result = self._make_result(batch_size=20)
        summary = result.summary()
        self.assertIn("Batch size", summary)
        self.assertIn("20", summary)
        self.assertIn("mini-batch", summary)

    def test_batch_size_with_train_test_split(self):
        result = self._make_result(batch_size=10, train_size=80, test_size=20)
        summary = result.summary()
        self.assertIn("Train / test", summary)
        self.assertIn("Batch size", summary)
        self.assertIn("10", summary)


class TestMiniBatchSamplingLogic(unittest.TestCase):
    """The inline _sample_batch helper mirrors _run_eval's sampling."""

    def test_full_set_when_batch_size_zero(self):
        convos = list(range(100))
        result = _sample_batch(convos, batch_size=0, iteration_seed=0)
        self.assertEqual(result, convos)

    def test_full_set_when_batch_equals_dataset(self):
        convos = list(range(50))
        result = _sample_batch(convos, batch_size=50, iteration_seed=0)
        self.assertEqual(result, convos)

    def test_full_set_when_batch_exceeds_dataset(self):
        convos = list(range(30))
        result = _sample_batch(convos, batch_size=100, iteration_seed=0)
        self.assertEqual(result, convos)

    def test_batch_size_respected(self):
        convos = list(range(100))
        result = _sample_batch(convos, batch_size=20, iteration_seed=0)
        self.assertEqual(len(result), 20)

    def test_sampled_indices_sorted(self):
        """Indices are sorted so conversations stay in original order."""
        convos = list(range(100))
        result = _sample_batch(convos, batch_size=10, iteration_seed=7)
        self.assertEqual(result, sorted(result))

    def test_sample_is_subset(self):
        convos = list(range(50))
        result = _sample_batch(convos, batch_size=10, iteration_seed=0)
        for item in result:
            self.assertIn(item, convos)

    def test_deterministic_same_seed(self):
        convos = list(range(200))
        r1 = _sample_batch(convos, batch_size=20, iteration_seed=42)
        r2 = _sample_batch(convos, batch_size=20, iteration_seed=42)
        self.assertEqual(r1, r2)

    def test_different_seeds_produce_different_batches(self):
        convos = list(range(200))
        r1 = _sample_batch(convos, batch_size=20, iteration_seed=0)
        r2 = _sample_batch(convos, batch_size=20, iteration_seed=1)
        self.assertNotEqual(r1, r2)

    def test_iteration_seed_shift(self):
        """Using batch_seed + i gives a unique batch per iteration."""
        convos = list(range(200))
        base_seed = 42
        batches = [_sample_batch(convos, batch_size=10, iteration_seed=base_seed + i)
                   for i in range(5)]
        # All batches should be distinct
        unique = [set(b) for b in batches]
        for a in range(len(unique)):
            for b in range(a + 1, len(unique)):
                self.assertNotEqual(unique[a], unique[b])

    def test_no_duplicates_in_batch(self):
        convos = list(range(100))
        result = _sample_batch(convos, batch_size=30, iteration_seed=5)
        self.assertEqual(len(result), len(set(result)))

    def test_small_dataset_batch_larger_than_dataset(self):
        convos = list(range(5))
        result = _sample_batch(convos, batch_size=10, iteration_seed=0)
        self.assertEqual(result, convos)

    def test_batch_size_one(self):
        convos = list(range(50))
        result = _sample_batch(convos, batch_size=1, iteration_seed=0)
        self.assertEqual(len(result), 1)


class TestIterativeStrategyBatchParams(unittest.TestCase):
    """_run_eval in iterative.py accepts batch_size and iteration_seed."""

    def test_run_eval_accepts_batch_params(self):
        """Signature check — ensure _run_eval has the new params."""
        import inspect
        from aevyra_reflex.strategies.iterative import _run_eval
        sig = inspect.signature(_run_eval)
        self.assertIn("batch_size", sig.parameters)
        self.assertIn("iteration_seed", sig.parameters)

    def test_batch_size_default_is_zero(self):
        import inspect
        from aevyra_reflex.strategies.iterative import _run_eval
        sig = inspect.signature(_run_eval)
        self.assertEqual(sig.parameters["batch_size"].default, 0)

    def test_iteration_seed_default_is_zero(self):
        import inspect
        from aevyra_reflex.strategies.iterative import _run_eval
        sig = inspect.signature(_run_eval)
        self.assertEqual(sig.parameters["iteration_seed"].default, 0)


if __name__ == "__main__":
    unittest.main()
