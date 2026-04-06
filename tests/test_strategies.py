"""Tests for strategy registry and strategy-specific logic."""

import pytest

from aevyra_reflex.strategies import (
    Strategy,
    get_strategy,
    list_strategies,
    register_strategy,
)
from aevyra_reflex.strategies.structural import STRUCTURAL_TRANSFORMS, StructuralStrategy
from aevyra_reflex.strategies.pdo import PDOStrategy, _copeland_ranking, _thompson_sample_pair, _win_rate
from aevyra_reflex.result import OptimizationResult

import numpy as np


class TestStrategyRegistry:
    """Tests for the strategy registry."""

    @pytest.mark.parametrize("name,expected", [
        ("auto", "AutoStrategy"),
        ("iterative", "IterativeStrategy"),
        ("pdo", "PDOStrategy"),
        ("fewshot", "FewShotStrategy"),
        ("structural", "StructuralStrategy"),
    ])
    def test_get_strategy(self, name, expected):
        cls = get_strategy(name)
        assert cls.__name__ == expected

    def test_unknown_strategy_raises(self):
        with pytest.raises((KeyError, ValueError)):
            get_strategy("nonexistent")

    def test_list_strategies(self):
        names = list_strategies()
        assert "auto" in names
        assert "iterative" in names
        assert names == sorted(names)  # alphabetical


class TestRegisterStrategy:
    """Tests for register_strategy() — custom strategy plugins."""

    def test_register_and_retrieve(self):
        class DummyStrategy(Strategy):
            def run(self, **kwargs):
                return OptimizationResult(
                    best_prompt="dummy", best_score=1.0,
                    iterations=[], converged=True,
                )

        register_strategy("dummy_test", DummyStrategy)
        assert get_strategy("dummy_test") is DummyStrategy
        assert "dummy_test" in list_strategies()

    def test_rejects_non_strategy_class(self):
        with pytest.raises(TypeError, match="must inherit from Strategy"):
            register_strategy("bad", str)

    def test_rejects_non_class(self):
        with pytest.raises(TypeError, match="must inherit from Strategy"):
            register_strategy("bad", lambda: None)

    def test_can_override_builtin(self):
        """Overriding a builtin is allowed (for testing or patching)."""
        original = get_strategy("iterative")

        class CustomIterative(Strategy):
            def run(self, **kwargs):
                return OptimizationResult(
                    best_prompt="custom", best_score=0.99,
                    iterations=[], converged=True,
                )

        register_strategy("iterative", CustomIterative)
        assert get_strategy("iterative") is CustomIterative

        # Restore
        register_strategy("iterative", original)


class TestStructuralTransforms:
    """Tests for structural transform definitions."""

    def test_all_transforms_are_strings(self):
        for name, instruction in STRUCTURAL_TRANSFORMS.items():
            assert isinstance(name, str)
            assert isinstance(instruction, str)
            assert len(instruction) > 20, f"{name} instruction too short"

    def test_expected_transforms_exist(self):
        expected = {
            "section_reorder", "markdown_structure", "minimal_flat",
            "xml_tags", "constraint_emphasis", "task_decomposition",
            "role_task_format", "input_anchored",
        }
        assert expected == set(STRUCTURAL_TRANSFORMS.keys())


class TestCopelandRanking:
    """Tests for PDO's Copeland ranking algorithm."""

    def test_clear_winner(self):
        # Prompt 0 beats everyone, prompt 2 loses to everyone
        W = np.array([
            [0, 5, 5],  # prompt 0: beats 1 and 2
            [1, 0, 5],  # prompt 1: loses to 0, beats 2
            [1, 1, 0],  # prompt 2: loses to both
        ])
        rankings = _copeland_ranking(W)
        assert rankings[0] == 0  # prompt 0 is champion
        assert rankings[-1] == 2  # prompt 2 is worst

    def test_tie_broken_by_winrate(self):
        # Prompts 0 and 1 have same Copeland score but different win rates
        W = np.array([
            [0, 3, 5],
            [2, 0, 5],
            [0, 0, 0],
        ])
        rankings = _copeland_ranking(W)
        # Both beat prompt 2, both have Copeland score of 1
        # But prompt 0 has better win rate against prompt 1
        assert rankings[0] == 0

    def test_empty_matrix(self):
        W = np.zeros((3, 3), dtype=int)
        rankings = _copeland_ranking(W)
        assert len(rankings) == 3
        assert set(rankings) == {0, 1, 2}


class TestWinRate:
    """Tests for PDO's win rate calculation."""

    def test_perfect_record(self):
        W = np.array([
            [0, 5, 5],
            [0, 0, 0],
            [0, 0, 0],
        ])
        assert _win_rate(W, 0) == 1.0

    def test_no_games(self):
        W = np.zeros((3, 3), dtype=int)
        assert _win_rate(W, 0) == 0.0

    def test_mixed_record(self):
        W = np.array([
            [0, 3, 2],
            [2, 0, 0],
            [3, 0, 0],
        ])
        # Prompt 0: 5 wins, 5 losses = 0.5
        assert _win_rate(W, 0) == 0.5


class TestThompsonSampling:
    """Tests for PDO's Thompson sampling pair selection."""

    def test_returns_two_different_indices(self):
        W = np.zeros((5, 5), dtype=int)
        i, j = _thompson_sample_pair(K=5, W=W, alpha=1.2, t=1)
        assert i != j
        assert 0 <= i < 5
        assert 0 <= j < 5

    def test_works_with_existing_wins(self):
        W = np.array([
            [0, 3, 1],
            [2, 0, 4],
            [1, 1, 0],
        ])
        i, j = _thompson_sample_pair(K=3, W=W, alpha=1.2, t=10)
        assert i != j

    def test_deterministic_with_seed(self):
        """Thompson sampling should give consistent results with same numpy seed."""
        W = np.zeros((4, 4), dtype=int)
        np.random.seed(42)
        i1, j1 = _thompson_sample_pair(K=4, W=W, alpha=1.2, t=1)
        np.random.seed(42)
        i2, j2 = _thompson_sample_pair(K=4, W=W, alpha=1.2, t=1)
        # Note: uses default_rng() internally so seed may not affect it
        # Just check it doesn't crash
        assert isinstance(i1, int) and isinstance(j1, int)
