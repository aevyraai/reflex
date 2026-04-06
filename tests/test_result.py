"""Tests for OptimizationResult — summary, analysis, serialization."""

import json

import pytest

from aevyra_reflex.result import (
    EvalSnapshot,
    IterationRecord,
    OptimizationResult,
    SampleSnapshot,
)


class TestOptimizationResult:
    """Core OptimizationResult properties."""

    def test_score_trajectory(self, optimization_result):
        traj = optimization_result.score_trajectory
        assert len(traj) == 10
        assert traj[0] == 0.35
        assert traj[-1] == 0.88

    def test_improvement(self, optimization_result):
        assert optimization_result.improvement == pytest.approx(0.58, abs=0.01)

    def test_improvement_pct(self, optimization_result):
        # (0.88 - 0.30) / 0.30 * 100 ≈ 193.3%
        assert optimization_result.improvement_pct == pytest.approx(193.3, abs=1.0)

    def test_improvement_none_without_baseline(self):
        result = OptimizationResult(
            best_prompt="x", best_score=0.5, iterations=[], converged=False,
        )
        assert result.improvement is None
        assert result.improvement_pct is None


class TestSummary:
    """Tests for the summary() output."""

    def test_summary_contains_scores(self, optimization_result):
        s = optimization_result.summary()
        assert "0.3000" in s  # baseline
        assert "0.8800" in s  # final

    def test_summary_contains_trajectory(self, optimization_result):
        s = optimization_result.summary()
        assert "0.350 →" in s

    def test_summary_contains_what_happened(self, optimization_result):
        s = optimization_result.summary()
        assert "WHAT HAPPENED" in s

    def test_summary_contains_before_after(self, optimization_result):
        s = optimization_result.summary()
        assert "BEFORE / AFTER EXAMPLE" in s
        assert "BEFORE (score:" in s
        assert "AFTER (score:" in s

    def test_summary_without_baseline(self):
        result = OptimizationResult(
            best_prompt="x", best_score=0.5,
            iterations=[IterationRecord(1, "x", 0.5)],
            converged=False,
        )
        s = result.summary()
        assert "Best score" in s
        assert "WHAT HAPPENED" not in s

    def test_summary_per_metric_breakdown(self, optimization_result):
        s = optimization_result.summary()
        assert "rouge" in s


class TestTrajectoryAnalysis:
    """Tests for _analyze_trajectory()."""

    def test_steady_climb(self, optimization_result):
        lines = optimization_result._analyze_trajectory()
        text = " ".join(lines)
        assert "improved steadily" in text or "closed" in text

    def test_plateau_detection(self, plateau_result):
        lines = plateau_result._analyze_trajectory()
        text = " ".join(lines)
        assert "plateau" in text.lower()

    def test_overoptimization_detection(self, overoptimized_result):
        lines = overoptimized_result._analyze_trajectory()
        text = " ".join(lines)
        assert "over-optimization" in text.lower() or "regressions" in text.lower()

    def test_gap_closed(self, optimization_result):
        lines = optimization_result._analyze_trajectory()
        text = " ".join(lines)
        assert "gap" in text

    def test_single_iteration_no_crash(self, baseline_snapshot, final_snapshot):
        result = OptimizationResult(
            best_prompt="x", best_score=0.5,
            iterations=[IterationRecord(1, "x", 0.5)],
            converged=False,
            baseline=baseline_snapshot,
            final=final_snapshot,
        )
        lines = result._analyze_trajectory()
        # Single iteration — not enough data for trajectory analysis
        assert isinstance(lines, list)

    def test_suggestions_present_when_not_converged(self, plateau_result):
        lines = plateau_result._analyze_trajectory()
        text = " ".join(lines)
        assert "To improve further" in text or "try" in text.lower()

    def test_converged_result_no_suggestions(self, optimization_result):
        lines = optimization_result._analyze_trajectory()
        text = " ".join(lines)
        assert "To improve further" not in text


class TestStrategyAnalysis:
    """Tests for _analyze_strategy()."""

    def test_auto_strategy_shows_all_phases(self, optimization_result):
        lines = optimization_result._analyze_strategy()
        text = " ".join(lines)
        assert "structural" in text
        assert "iterative" in text
        assert "fewshot" in text

    def test_auto_strategy_educational_lessons(self, optimization_result):
        lines = optimization_result._analyze_strategy()
        text = " ".join(lines)
        assert "Lesson:" in text
        # Should have a lesson for each phase
        assert "Structure matters" in text
        assert "Specificity matters" in text
        assert "Examples matter" in text

    def test_single_strategy_lesson(self):
        result = OptimizationResult(
            best_prompt="x", best_score=0.7,
            iterations=[
                IterationRecord(1, "v1", 0.4),
                IterationRecord(2, "v2", 0.7),
            ],
            converged=False,
            strategy_name="iterative",
        )
        lines = result._analyze_strategy()
        text = " ".join(lines)
        assert "Specificity matters" in text

    def test_hurt_phase_lesson(self):
        result = OptimizationResult(
            best_prompt="x", best_score=0.5,
            iterations=[IterationRecord(1, "v1", 0.5)],
            converged=False,
            strategy_name="auto",
            phase_history=[
                {"phase": 1, "axis": "fewshot", "score_before": 0.6, "score_after": 0.5},
            ],
        )
        lines = result._analyze_strategy()
        text = " ".join(lines)
        assert "worse" in text.lower()

    def test_neutral_phase_lesson(self):
        result = OptimizationResult(
            best_prompt="x", best_score=0.5,
            iterations=[IterationRecord(1, "v1", 0.5)],
            converged=False,
            strategy_name="auto",
            phase_history=[
                {"phase": 1, "axis": "iterative", "score_before": 0.50, "score_after": 0.51},
            ],
        )
        lines = result._analyze_strategy()
        text = " ".join(lines)
        assert "no effect" in text.lower()

    def test_no_strategy_name_returns_empty(self):
        result = OptimizationResult(
            best_prompt="x", best_score=0.5,
            iterations=[], converged=False,
        )
        assert result._analyze_strategy() == []


class TestPromptChangeAnalysis:
    """Tests for _analyze_prompt_changes()."""

    def test_detects_length_increase(self, optimization_result):
        lines = optimization_result._analyze_prompt_changes()
        text = " ".join(lines)
        assert "Longer" in text or "longer" in text

    def test_detects_added_features(self):
        baseline = EvalSnapshot(
            mean_score=0.3,
            system_prompt="Be helpful.",
        )
        result = OptimizationResult(
            best_prompt="## Role\nYou are an expert.\n\n## Rules\n- **Never** fabricate\n- Do not editorialize\n\n<format>structured output</format>",
            best_score=0.8,
            iterations=[IterationRecord(1, "x", 0.8)],
            converged=False,
            baseline=baseline,
            final=EvalSnapshot(mean_score=0.8),
        )
        lines = result._analyze_prompt_changes()
        text = " ".join(lines)
        assert "markdown headers" in text
        assert "bold emphasis" in text
        assert "XML tags" in text

    def test_no_baseline_prompt_returns_empty(self):
        result = OptimizationResult(
            best_prompt="x", best_score=0.5,
            iterations=[], converged=False,
            baseline=EvalSnapshot(mean_score=0.3),
        )
        lines = result._analyze_prompt_changes()
        assert lines == []

    def test_same_prompt_returns_empty(self):
        result = OptimizationResult(
            best_prompt="Same prompt",
            best_score=0.5,
            iterations=[], converged=False,
            baseline=EvalSnapshot(mean_score=0.3, system_prompt="Same prompt"),
        )
        lines = result._analyze_prompt_changes()
        assert lines == []


class TestBeforeAfterExample:
    """Tests for _before_after_example()."""

    def test_picks_most_improved_sample(self, optimization_result):
        lines = optimization_result._before_after_example()
        text = "\n".join(lines)
        # Should contain BEFORE and AFTER sections
        assert "BEFORE (score:" in text
        assert "AFTER (score:" in text
        assert "Score change:" in text

    def test_no_samples_returns_empty(self):
        result = OptimizationResult(
            best_prompt="x", best_score=0.5,
            iterations=[], converged=False,
            baseline=EvalSnapshot(mean_score=0.3),
            final=EvalSnapshot(mean_score=0.5),
        )
        assert result._before_after_example() == []

    def test_truncates_long_text(self):
        long_text = "x" * 1000
        baseline = EvalSnapshot(
            mean_score=0.3,
            samples=[SampleSnapshot(long_text, long_text, long_text, 0.2)],
        )
        final = EvalSnapshot(
            mean_score=0.8,
            samples=[SampleSnapshot(long_text, long_text, long_text, 0.9)],
        )
        result = OptimizationResult(
            best_prompt="x", best_score=0.8,
            iterations=[], converged=False,
            baseline=baseline, final=final,
        )
        lines = result._before_after_example()
        text = "\n".join(lines)
        assert "..." in text  # truncation happened


class TestSerialization:
    """Tests for to_dict() and to_json()."""

    def test_to_dict_contains_all_fields(self, optimization_result):
        d = optimization_result.to_dict()
        assert d["best_prompt"] == optimization_result.best_prompt
        assert d["best_score"] == 0.88
        assert d["converged"] is True
        assert len(d["iterations"]) == 10
        assert "baseline" in d
        assert "final" in d
        assert d["strategy_name"] == "auto"
        assert len(d["phase_history"]) == 3
        assert "improvement" in d
        assert "improvement_pct" in d

    def test_to_dict_without_optional_fields(self):
        result = OptimizationResult(
            best_prompt="x", best_score=0.5,
            iterations=[], converged=False,
        )
        d = result.to_dict()
        assert "strategy_name" not in d
        assert "phase_history" not in d
        assert "improvement" not in d

    def test_to_json_roundtrip(self, optimization_result, tmp_path):
        path = tmp_path / "result.json"
        optimization_result.to_json(path)
        loaded = json.loads(path.read_text())
        assert loaded["best_score"] == 0.88
        assert loaded["strategy_name"] == "auto"

    def test_save_best_prompt(self, optimization_result, tmp_path):
        path = tmp_path / "best.md"
        optimization_result.save_best_prompt(path)
        assert path.read_text() == optimization_result.best_prompt


# ---------------------------------------------------------------------------
# Token fields
# ---------------------------------------------------------------------------

class TestTokenFields:
    """Tests for eval_tokens / reasoning_tokens fields on result objects."""

    def test_iteration_record_default_tokens_zero(self):
        r = IterationRecord(iteration=1, system_prompt="p", score=0.5)
        assert r.eval_tokens == 0
        assert r.reasoning_tokens == 0

    def test_iteration_record_tokens_set(self):
        r = IterationRecord(
            iteration=1, system_prompt="p", score=0.5,
            eval_tokens=8000, reasoning_tokens=3000,
        )
        assert r.eval_tokens == 8000
        assert r.reasoning_tokens == 3000

    def test_eval_snapshot_default_tokens_zero(self):
        snap = EvalSnapshot(mean_score=0.5)
        assert snap.total_tokens == 0

    def test_eval_snapshot_tokens_set(self):
        snap = EvalSnapshot(mean_score=0.5, total_tokens=12000)
        assert snap.total_tokens == 12000

    def test_optimization_result_default_tokens_zero(self):
        result = OptimizationResult(
            best_prompt="p", best_score=0.5, iterations=[], converged=False,
        )
        assert result.total_eval_tokens == 0
        assert result.total_reasoning_tokens == 0

    def test_to_dict_includes_token_fields(self):
        result = OptimizationResult(
            best_prompt="p", best_score=0.8,
            iterations=[
                IterationRecord(
                    iteration=1, system_prompt="p", score=0.8,
                    eval_tokens=9000, reasoning_tokens=3500,
                )
            ],
            converged=False,
            total_eval_tokens=9000,
            total_reasoning_tokens=3500,
        )
        d = result.to_dict()
        assert d["total_eval_tokens"] == 9000
        assert d["total_reasoning_tokens"] == 3500
        assert d["iterations"][0]["eval_tokens"] == 9000
        assert d["iterations"][0]["reasoning_tokens"] == 3500

    def test_to_dict_baseline_includes_total_tokens(self):
        result = OptimizationResult(
            best_prompt="p", best_score=0.8, iterations=[], converged=False,
            baseline=EvalSnapshot(mean_score=0.6, total_tokens=5000),
            final=EvalSnapshot(mean_score=0.8, total_tokens=6000),
        )
        d = result.to_dict()
        assert d["baseline"]["total_tokens"] == 5000
        assert d["final"]["total_tokens"] == 6000

    def test_summary_shows_token_lines_when_nonzero(self):
        result = OptimizationResult(
            best_prompt="p", best_score=0.8,
            iterations=[IterationRecord(iteration=1, system_prompt="p", score=0.8)],
            converged=True,
            baseline=EvalSnapshot(mean_score=0.6),
            final=EvalSnapshot(mean_score=0.8),
            total_eval_tokens=82000,
            total_reasoning_tokens=31000,
        )
        summary = result.summary()
        assert "Eval tokens" in summary
        assert "Reasoning tokens" in summary
        assert "82.0K" in summary
        assert "31.0K" in summary

    def test_summary_omits_token_lines_when_zero(self):
        result = OptimizationResult(
            best_prompt="p", best_score=0.8,
            iterations=[IterationRecord(iteration=1, system_prompt="p", score=0.8)],
            converged=True,
            baseline=EvalSnapshot(mean_score=0.6),
            final=EvalSnapshot(mean_score=0.8),
            total_eval_tokens=0,
            total_reasoning_tokens=0,
        )
        summary = result.summary()
        assert "Eval tokens" not in summary
        assert "Reasoning tokens" not in summary
