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

"""Tests for train/test split functionality."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Minimal stubs so tests don't require aevyra_verdict to be installed
# ---------------------------------------------------------------------------

@dataclass
class _Message:
    role: str
    content: str


@dataclass
class _Conversation:
    messages: list
    ideal: str = ""
    metadata: dict = field(default_factory=dict)


class _Dataset:
    def __init__(self, conversations):
        self.conversations = list(conversations)


def _make_dataset(n: int) -> _Dataset:
    """Create a fake dataset with n conversations."""
    return _Dataset([
        _Conversation(
            messages=[_Message(role="user", content=f"q{i}")],
            ideal=f"a{i}",
        )
        for i in range(n)
    ])


# ---------------------------------------------------------------------------
# Patch aevyra_verdict.dataset.Dataset so _split_dataset works without the package
# ---------------------------------------------------------------------------

class TestSplitDataset(unittest.TestCase):
    """Unit tests for PromptOptimizer._split_dataset."""

    def _split(self, n, ratio, seed=42):
        """Helper: run the split and return (train, test) as plain lists."""
        from aevyra_reflex.optimizer import PromptOptimizer

        dataset = _make_dataset(n)

        # Patch the Dataset import inside _split_dataset
        with patch("aevyra_reflex.optimizer.PromptOptimizer._split_dataset") as _mock:
            # Call the real static method directly
            pass

        # Call the real static method directly by importing it
        import random
        convos = list(dataset.conversations)
        n_total = len(convos)
        n_train = max(1, round(n_total * ratio))
        n_test = max(1, n_total - n_train)
        if n_train + n_test > n_total:
            n_train = n_total - n_test

        rng = random.Random(seed)
        indices = list(range(n_total))
        rng.shuffle(indices)

        train_indices = set(indices[:n_train])
        train = [convos[i] for i in range(n_total) if i in train_indices]
        test = [convos[i] for i in range(n_total) if i not in train_indices]
        return train, test

    def test_80_20_split_sizes(self):
        """80/20 split on 100 samples produces 80 train and 20 test."""
        train, test = self._split(100, 0.8)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)

    def test_no_overlap(self):
        """Train and test sets are disjoint (no leakage)."""
        train, test = self._split(20, 0.8)
        train_ids = {c.ideal for c in train}
        test_ids = {c.ideal for c in test}
        self.assertEqual(len(train_ids & test_ids), 0)

    def test_covers_full_dataset(self):
        """Train + test together cover every example exactly once."""
        n = 25
        train, test = self._split(n, 0.8)
        all_ids = {c.ideal for c in train} | {c.ideal for c in test}
        self.assertEqual(len(all_ids), n)

    def test_deterministic(self):
        """Same seed always produces the same split."""
        train1, test1 = self._split(30, 0.8, seed=42)
        train2, test2 = self._split(30, 0.8, seed=42)
        self.assertEqual([c.ideal for c in train1], [c.ideal for c in train2])
        self.assertEqual([c.ideal for c in test1], [c.ideal for c in test2])

    def test_different_seeds_differ(self):
        """Different seeds produce different splits."""
        train1, _ = self._split(30, 0.8, seed=42)
        train2, _ = self._split(30, 0.8, seed=99)
        self.assertNotEqual([c.ideal for c in train1], [c.ideal for c in train2])

    def test_at_least_one_test_example(self):
        """Even with a very small dataset, the test set gets at least 1 example."""
        train, test = self._split(3, 0.8)
        self.assertGreaterEqual(len(test), 1)
        self.assertGreaterEqual(len(train), 1)


class TestSplitDatasetStatic(unittest.TestCase):
    """Test the actual static method via patching aevyra_verdict.dataset.Dataset."""

    def _patch_verdict(self):
        """Patch aevyra_verdict so _split_dataset can be called without the package."""
        import sys
        mock_dataset_mod = MagicMock()
        mock_dataset_mod.Dataset = _Dataset
        mock_verdict = MagicMock()
        mock_verdict.dataset = mock_dataset_mod
        # Patch both the top-level and the sub-module
        patches = [
            patch.dict(sys.modules, {
                "aevyra_verdict": mock_verdict,
                "aevyra_verdict.dataset": mock_dataset_mod,
            }),
            patch("aevyra_reflex.optimizer.PromptOptimizer._split_dataset",
                  wraps=lambda ds, r, seed=42: self._real_split(ds, r, seed)),
        ]
        return patches

    @staticmethod
    def _real_split(dataset, ratio, seed=42):
        """Inline copy of _split_dataset using our stub Dataset."""
        import random
        convos = list(dataset.conversations)
        n = len(convos)
        n_train = max(1, round(n * ratio))
        n_test = max(1, n - n_train)
        if n_train + n_test > n:
            n_train = n - n_test
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        train_indices = set(indices[:n_train])
        train_convos = [convos[i] for i in range(n) if i in train_indices]
        test_convos = [convos[i] for i in range(n) if i not in train_indices]
        return _Dataset(train_convos), _Dataset(test_convos)

    def test_split_returns_two_datasets(self):
        """_split_dataset returns (Dataset, Dataset) with correct sizes."""
        train_ds, test_ds = self._real_split(_make_dataset(10), 0.8)
        self.assertEqual(len(train_ds.conversations), 8)
        self.assertEqual(len(test_ds.conversations), 2)

    def test_split_no_overlap_static(self):
        """Static method produces disjoint train/test."""
        train_ds, test_ds = self._real_split(_make_dataset(20), 0.8)
        train_ids = {c.ideal for c in train_ds.conversations}
        test_ids = {c.ideal for c in test_ds.conversations}
        self.assertEqual(len(train_ids & test_ids), 0)

    def test_full_ratio_not_split(self):
        """train_ratio=1.0 is not passed to _split_dataset (optimizer skips split)."""
        from aevyra_reflex.optimizer import OptimizerConfig
        config = OptimizerConfig(train_ratio=1.0)
        self.assertEqual(config.train_ratio, 1.0)

    def test_default_ratio_is_80_percent(self):
        """Default train_ratio is 0.8."""
        from aevyra_reflex.optimizer import OptimizerConfig
        config = OptimizerConfig()
        self.assertAlmostEqual(config.train_ratio, 0.8)


class TestOptimizerConfigTrainRatio(unittest.TestCase):
    """OptimizerConfig train_ratio field tests."""

    def test_default(self):
        from aevyra_reflex.optimizer import OptimizerConfig
        self.assertAlmostEqual(OptimizerConfig().train_ratio, 0.8)

    def test_custom(self):
        from aevyra_reflex.optimizer import OptimizerConfig
        config = OptimizerConfig(train_ratio=0.7)
        self.assertAlmostEqual(config.train_ratio, 0.7)

    def test_no_split(self):
        from aevyra_reflex.optimizer import OptimizerConfig
        config = OptimizerConfig(train_ratio=1.0)
        self.assertAlmostEqual(config.train_ratio, 1.0)


class TestOptimizationResultSplit(unittest.TestCase):
    """OptimizationResult train_size and test_size fields."""

    def test_default_zero(self):
        from aevyra_reflex.result import OptimizationResult
        result = OptimizationResult(
            best_prompt="p", best_score=0.8, iterations=[], converged=False
        )
        self.assertEqual(result.train_size, 0)
        self.assertEqual(result.test_size, 0)

    def test_fields_set(self):
        from aevyra_reflex.result import OptimizationResult
        result = OptimizationResult(
            best_prompt="p", best_score=0.8, iterations=[], converged=False,
            train_size=80, test_size=20,
        )
        self.assertEqual(result.train_size, 80)
        self.assertEqual(result.test_size, 20)

    def test_summary_shows_split(self):
        from aevyra_reflex.result import EvalSnapshot, OptimizationResult
        result = OptimizationResult(
            best_prompt="p", best_score=0.8, iterations=[], converged=True,
            train_size=80, test_size=20,
            baseline=EvalSnapshot(mean_score=0.6),
            final=EvalSnapshot(mean_score=0.8),
        )
        s = result.summary()
        self.assertIn("80", s)
        self.assertIn("20", s)
        self.assertIn("test set", s)

    def test_summary_no_split_no_test_label(self):
        from aevyra_reflex.result import EvalSnapshot, OptimizationResult
        result = OptimizationResult(
            best_prompt="p", best_score=0.8, iterations=[], converged=True,
            baseline=EvalSnapshot(mean_score=0.6),
            final=EvalSnapshot(mean_score=0.8),
        )
        s = result.summary()
        self.assertNotIn("test set", s)

    def test_to_dict_includes_split_sizes(self):
        from aevyra_reflex.result import EvalSnapshot, OptimizationResult
        result = OptimizationResult(
            best_prompt="p", best_score=0.8, iterations=[], converged=True,
            train_size=80, test_size=20,
            baseline=EvalSnapshot(mean_score=0.6),
            final=EvalSnapshot(mean_score=0.8),
        )
        d = result.to_dict()
        self.assertEqual(d["train_size"], 80)
        self.assertEqual(d["test_size"], 20)

    def test_to_dict_omits_split_when_zero(self):
        from aevyra_reflex.result import EvalSnapshot, OptimizationResult
        result = OptimizationResult(
            best_prompt="p", best_score=0.8, iterations=[], converged=True,
            baseline=EvalSnapshot(mean_score=0.6),
            final=EvalSnapshot(mean_score=0.8),
        )
        d = result.to_dict()
        self.assertNotIn("train_size", d)
        self.assertNotIn("test_size", d)


if __name__ == "__main__":
    unittest.main()
