"""Shared fixtures for aevyra-reflex tests."""

import pytest

from aevyra_reflex.result import (
    EvalSnapshot,
    IterationRecord,
    OptimizationResult,
    SampleSnapshot,
)


@pytest.fixture
def sample_snapshots_baseline():
    """5 low-scoring baseline sample snapshots."""
    return [
        SampleSnapshot("What is 2+2?", "The answer is probably 4.", "4", 0.30),
        SampleSnapshot("Capital of France?", "I think it might be Paris.", "Paris", 0.35),
        SampleSnapshot("Summarize photosynthesis.", "Plants use light.", "Plants convert sunlight to energy via chlorophyll.", 0.25),
        SampleSnapshot("What is HTTP?", "A protocol.", "HyperText Transfer Protocol for web communication.", 0.20),
        SampleSnapshot("Translate 'hello' to Spanish.", "Hello in Spanish.", "hola", 0.40),
    ]


@pytest.fixture
def sample_snapshots_final():
    """5 higher-scoring final sample snapshots."""
    return [
        SampleSnapshot("What is 2+2?", "4", "4", 0.95),
        SampleSnapshot("Capital of France?", "Paris", "Paris", 0.90),
        SampleSnapshot("Summarize photosynthesis.", "Plants convert sunlight into chemical energy using chlorophyll in their leaves.", "Plants convert sunlight to energy via chlorophyll.", 0.75),
        SampleSnapshot("What is HTTP?", "HTTP (HyperText Transfer Protocol) is the protocol used for web communication between browsers and servers.", "HyperText Transfer Protocol for web communication.", 0.80),
        SampleSnapshot("Translate 'hello' to Spanish.", "hola", "hola", 1.00),
    ]


@pytest.fixture
def baseline_snapshot(sample_snapshots_baseline):
    """Baseline eval snapshot."""
    return EvalSnapshot(
        mean_score=0.30,
        scores_by_metric={"rouge": 0.30},
        system_prompt="You are a helpful assistant.",
        samples=sample_snapshots_baseline,
    )


@pytest.fixture
def final_snapshot(sample_snapshots_final):
    """Final eval snapshot."""
    return EvalSnapshot(
        mean_score=0.88,
        scores_by_metric={"rouge": 0.88},
        system_prompt="You are an expert assistant. Be precise and concise. Always answer directly.",
        samples=sample_snapshots_final,
    )


@pytest.fixture
def iteration_records():
    """10-iteration trajectory with a dip at iteration 5."""
    scores = [0.35, 0.45, 0.52, 0.60, 0.55, 0.65, 0.72, 0.78, 0.85, 0.88]
    return [
        IterationRecord(i + 1, f"prompt_v{i + 1}", s, {"rouge": s})
        for i, s in enumerate(scores)
    ]


@pytest.fixture
def optimization_result(baseline_snapshot, final_snapshot, iteration_records):
    """A complete optimization result with all fields populated."""
    return OptimizationResult(
        best_prompt=final_snapshot.system_prompt,
        best_score=0.88,
        iterations=iteration_records,
        converged=True,
        baseline=baseline_snapshot,
        final=final_snapshot,
        strategy_name="auto",
        phase_history=[
            {"phase": 1, "axis": "structural", "iterations_used": 3, "score_before": 0.35, "score_after": 0.52},
            {"phase": 2, "axis": "iterative", "iterations_used": 4, "score_before": 0.52, "score_after": 0.78},
            {"phase": 3, "axis": "fewshot", "iterations_used": 3, "score_before": 0.78, "score_after": 0.88},
        ],
    )


@pytest.fixture
def plateau_result(baseline_snapshot, final_snapshot):
    """Result where score plateaus after initial jump."""
    scores = [0.50, 0.65, 0.66, 0.65, 0.66, 0.65]
    iterations = [
        IterationRecord(i + 1, f"prompt_v{i + 1}", s, {"rouge": s})
        for i, s in enumerate(scores)
    ]
    final_snapshot.mean_score = 0.65
    return OptimizationResult(
        best_prompt="plateau prompt",
        best_score=0.65,
        iterations=iterations,
        converged=False,
        baseline=baseline_snapshot,
        final=final_snapshot,
        strategy_name="iterative",
    )


@pytest.fixture
def overoptimized_result(baseline_snapshot, final_snapshot):
    """Result where score peaks early then declines."""
    scores = [0.50, 0.75, 0.80, 0.65, 0.55, 0.50]
    iterations = [
        IterationRecord(i + 1, f"prompt_v{i + 1}", s, {"rouge": s})
        for i, s in enumerate(scores)
    ]
    # best_prompt should be from the peak iteration
    return OptimizationResult(
        best_prompt="peak prompt",
        best_score=0.80,
        iterations=iterations,
        converged=False,
        baseline=baseline_snapshot,
        final=final_snapshot,
        strategy_name="iterative",
    )
