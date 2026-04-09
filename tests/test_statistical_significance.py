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

"""Tests for statistical significance: _compute_significance, multi-run eval,
EvalSnapshot.std_score/n_runs, OptimizationResult.p_value/is_significant."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Minimal stand-ins for aevyra_verdict types (avoids the package import)
# ---------------------------------------------------------------------------

@dataclass
class _SampleSnapshot:
    input: str = ""
    response: str = ""
    ideal: str = ""
    score: float = 0.0


@dataclass
class _EvalSnapshot:
    mean_score: float = 0.0
    scores_by_metric: dict = field(default_factory=dict)
    system_prompt: str = ""
    samples: list = field(default_factory=list)
    total_tokens: int = 0
    std_score: float = 0.0
    n_runs: int = 1


@dataclass
class _OptimizationResult:
    best_prompt: str = ""
    best_score: float = 0.0
    iterations: list = field(default_factory=list)
    converged: bool = False
    baseline: _EvalSnapshot | None = None
    final: _EvalSnapshot | None = None
    p_value: float | None = None
    is_significant: bool | None = None
    total_eval_tokens: int = 0
    total_reasoning_tokens: int = 0
    strategy_name: str = ""
    phase_history: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Inline significance implementation (mirrors optimizer._compute_significance)
# We test the logic directly without importing aevyra_reflex to keep tests
# runnable even when verdict is not installed.
# ---------------------------------------------------------------------------

def _compute_significance(baseline, final):
    """Inline mirror of PromptOptimizer._compute_significance for testing."""
    import math

    if not baseline.samples or not final.samples:
        return None, None

    n = min(len(baseline.samples), len(final.samples))
    if n < 2:
        return None, None

    b_scores = [s.score for s in baseline.samples[:n]]
    f_scores = [s.score for s in final.samples[:n]]
    diffs = [f - b for f, b in zip(f_scores, b_scores)]

    if all(d == 0.0 for d in diffs):
        return None, None

    try:
        from scipy.stats import wilcoxon  # type: ignore[import]
        _, p_value = wilcoxon(b_scores, f_scores)
        return float(p_value), bool(p_value < 0.05)
    except ImportError:
        pass

    # Fallback: paired t-test
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    if var_d == 0.0:
        return None, None
    t_stat = mean_d / math.sqrt(var_d / n)

    # Approximate p-value (same beta_inc_approx as optimizer)
    from aevyra_reflex.optimizer import _beta_inc_approx
    df = n - 1
    x = df / (df + t_stat ** 2)
    p_approx = _beta_inc_approx(x, df / 2, 0.5)
    return float(p_approx), bool(p_approx < 0.05)


def _make_snapshots(baseline_scores, final_scores):
    """Build two EvalSnapshots from plain score lists."""
    b = _EvalSnapshot(
        mean_score=sum(baseline_scores) / len(baseline_scores),
        samples=[_SampleSnapshot(score=s) for s in baseline_scores],
    )
    f = _EvalSnapshot(
        mean_score=sum(final_scores) / len(final_scores),
        samples=[_SampleSnapshot(score=s) for s in final_scores],
    )
    return b, f


# ===========================================================================
# Tests
# ===========================================================================

class TestComputeSignificanceEdgeCases(unittest.TestCase):
    """Edge cases: empty samples, single sample, all-zero diffs."""

    def test_no_samples_returns_none(self):
        b = _EvalSnapshot(mean_score=0.5, samples=[])
        f = _EvalSnapshot(mean_score=0.7, samples=[])
        p, sig = _compute_significance(b, f)
        self.assertIsNone(p)
        self.assertIsNone(sig)

    def test_single_sample_returns_none(self):
        b, f = _make_snapshots([0.5], [0.8])
        p, sig = _compute_significance(b, f)
        self.assertIsNone(p)
        self.assertIsNone(sig)

    def test_all_zero_diffs_returns_none(self):
        # If all pairs have the same score, the test is undefined
        b, f = _make_snapshots([0.6, 0.7, 0.8], [0.6, 0.7, 0.8])
        p, sig = _compute_significance(b, f)
        self.assertIsNone(p)
        self.assertIsNone(sig)

    def test_none_baseline_samples(self):
        b = _EvalSnapshot(mean_score=0.5)  # samples defaults to []
        f, _ = _make_snapshots([0.7, 0.8, 0.9], [0.7, 0.8, 0.9])
        p, sig = _compute_significance(b, f)
        self.assertIsNone(p)
        self.assertIsNone(sig)


class TestComputeSignificanceResult(unittest.TestCase):
    """Results with a meaningful improvement should be significant."""

    def test_large_improvement_is_significant(self):
        # All samples improve by 0.4 — very clearly significant
        baseline_scores = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2]
        final_scores    = [0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6, 0.5, 0.6]
        b, f = _make_snapshots(baseline_scores, final_scores)
        p, sig = _compute_significance(b, f)
        self.assertIsNotNone(p)
        self.assertTrue(sig, f"Expected significant, got p={p}")
        self.assertLess(p, 0.05)

    def test_no_improvement_is_not_significant(self):
        # Scores are the same — not significant
        scores = [0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.6, 0.4, 0.7, 0.3]
        b, f = _make_snapshots(scores, scores)
        p, sig = _compute_significance(b, f)
        # All diffs are zero → returns None
        self.assertIsNone(p)
        self.assertIsNone(sig)

    def test_p_value_is_float(self):
        baseline_scores = [0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3]
        final_scores    = [0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9]
        b, f = _make_snapshots(baseline_scores, final_scores)
        p, sig = _compute_significance(b, f)
        if p is not None:
            self.assertIsInstance(p, float)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_is_significant_is_bool(self):
        baseline_scores = [0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3]
        final_scores    = [0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9]
        b, f = _make_snapshots(baseline_scores, final_scores)
        _, sig = _compute_significance(b, f)
        if sig is not None:
            self.assertIsInstance(sig, bool)

    def test_p_value_bounded(self):
        """p-value must always be in [0, 1]."""
        import random
        rng = random.Random(42)
        baseline_scores = [rng.uniform(0.3, 0.5) for _ in range(20)]
        final_scores = [s + rng.uniform(0.1, 0.3) for s in baseline_scores]
        b, f = _make_snapshots(baseline_scores, final_scores)
        p, _ = _compute_significance(b, f)
        if p is not None:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)


class TestOptimizerConfigEvalRuns(unittest.TestCase):
    """OptimizerConfig.eval_runs field."""

    def test_default_eval_runs_is_1(self):
        from aevyra_reflex.optimizer import OptimizerConfig
        cfg = OptimizerConfig()
        self.assertEqual(cfg.eval_runs, 1)

    def test_custom_eval_runs(self):
        from aevyra_reflex.optimizer import OptimizerConfig
        cfg = OptimizerConfig(eval_runs=5)
        self.assertEqual(cfg.eval_runs, 5)


class TestEvalSnapshotStatFields(unittest.TestCase):
    """EvalSnapshot.std_score and n_runs defaults and assignment."""

    def test_default_std_score_is_zero(self):
        from aevyra_reflex.result import EvalSnapshot
        snap = EvalSnapshot(mean_score=0.5)
        self.assertEqual(snap.std_score, 0.0)

    def test_default_n_runs_is_1(self):
        from aevyra_reflex.result import EvalSnapshot
        snap = EvalSnapshot(mean_score=0.5)
        self.assertEqual(snap.n_runs, 1)

    def test_custom_std_and_n_runs(self):
        from aevyra_reflex.result import EvalSnapshot
        snap = EvalSnapshot(mean_score=0.72, std_score=0.03, n_runs=5)
        self.assertAlmostEqual(snap.std_score, 0.03)
        self.assertEqual(snap.n_runs, 5)


class TestOptimizationResultStatFields(unittest.TestCase):
    """OptimizationResult.p_value and is_significant defaults."""

    def test_default_p_value_is_none(self):
        from aevyra_reflex.result import OptimizationResult
        r = OptimizationResult(best_prompt="x", best_score=0.8, iterations=[], converged=False)
        self.assertIsNone(r.p_value)

    def test_default_is_significant_is_none(self):
        from aevyra_reflex.result import OptimizationResult
        r = OptimizationResult(best_prompt="x", best_score=0.8, iterations=[], converged=False)
        self.assertIsNone(r.is_significant)

    def test_set_p_value_and_significance(self):
        from aevyra_reflex.result import OptimizationResult
        r = OptimizationResult(best_prompt="x", best_score=0.8, iterations=[], converged=False)
        r.p_value = 0.021
        r.is_significant = True
        self.assertAlmostEqual(r.p_value, 0.021)
        self.assertTrue(r.is_significant)


class TestSummarySignificanceLine(unittest.TestCase):
    """summary() includes the significance line."""

    def _make_result(self, p_value, is_significant):
        from aevyra_reflex.result import EvalSnapshot, OptimizationResult, SampleSnapshot
        samples_b = [SampleSnapshot(input="q", response="a", ideal="a", score=s)
                     for s in [0.5, 0.6, 0.4, 0.5, 0.6]]
        samples_f = [SampleSnapshot(input="q", response="a", ideal="a", score=s)
                     for s in [0.8, 0.9, 0.8, 0.7, 0.9]]
        b = EvalSnapshot(mean_score=0.52, samples=samples_b, system_prompt="before")
        f = EvalSnapshot(mean_score=0.84, samples=samples_f, system_prompt="after")
        r = OptimizationResult(
            best_prompt="after",
            best_score=0.84,
            iterations=[],
            converged=False,
            baseline=b,
            final=f,
            p_value=p_value,
            is_significant=is_significant,
        )
        return r

    def test_significant_shows_check_mark(self):
        r = self._make_result(p_value=0.012, is_significant=True)
        s = r.summary()
        self.assertIn("p=0.0120", s)
        self.assertIn("✓ significant", s)

    def test_not_significant_shows_cross(self):
        r = self._make_result(p_value=0.312, is_significant=False)
        s = r.summary()
        self.assertIn("p=0.3120", s)
        self.assertIn("✗ not significant", s)

    def test_no_p_value_shows_install_hint(self):
        r = self._make_result(p_value=None, is_significant=None)
        s = r.summary()
        self.assertIn("scipy", s)

    def test_std_score_shown_when_nonzero(self):
        from aevyra_reflex.result import EvalSnapshot, OptimizationResult
        b = EvalSnapshot(mean_score=0.62, std_score=0.018, n_runs=3, system_prompt="before")
        f = EvalSnapshot(mean_score=0.74, std_score=0.011, n_runs=3, system_prompt="after")
        r = OptimizationResult(
            best_prompt="after",
            best_score=0.74,
            iterations=[],
            converged=False,
            baseline=b,
            final=f,
        )
        s = r.summary()
        self.assertIn("± 0.0180", s)
        self.assertIn("± 0.0110", s)
        self.assertIn("(3 runs)", s)


class TestToDictIncludesStatFields(unittest.TestCase):
    """to_dict() includes p_value, is_significant, std_score, n_runs."""

    def test_to_dict_includes_p_value(self):
        from aevyra_reflex.result import EvalSnapshot, OptimizationResult
        b = EvalSnapshot(mean_score=0.5, std_score=0.02, n_runs=3)
        f = EvalSnapshot(mean_score=0.75, std_score=0.01, n_runs=3)
        r = OptimizationResult(
            best_prompt="x", best_score=0.75, iterations=[], converged=False,
            baseline=b, final=f, p_value=0.031, is_significant=True,
        )
        d = r.to_dict()
        self.assertAlmostEqual(d["p_value"], 0.031)
        self.assertTrue(d["is_significant"])
        self.assertAlmostEqual(d["baseline"]["std_score"], 0.02)
        self.assertEqual(d["baseline"]["n_runs"], 3)
        self.assertAlmostEqual(d["final"]["std_score"], 0.01)
        self.assertEqual(d["final"]["n_runs"], 3)

    def test_to_dict_omits_p_value_when_none(self):
        from aevyra_reflex.result import EvalSnapshot, OptimizationResult
        b = EvalSnapshot(mean_score=0.5)
        f = EvalSnapshot(mean_score=0.75)
        r = OptimizationResult(
            best_prompt="x", best_score=0.75, iterations=[], converged=False,
            baseline=b, final=f,
        )
        d = r.to_dict()
        self.assertNotIn("p_value", d)
        self.assertNotIn("is_significant", d)


if __name__ == "__main__":
    unittest.main()
