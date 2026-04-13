"""Tests for PromptOptimizer — provider resolution, config, Ollama detection, verdict integration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aevyra_reflex.optimizer import (
    PROVIDER_ALIASES,
    OptimizerConfig,
    PromptOptimizer,
    _resolve_provider,
    parse_verdict_results,
)
from aevyra_reflex.result import EvalSnapshot, IterationRecord, OptimizationResult
from aevyra_reflex.run_store import RunStore


class TestProviderAliases:
    """Tests for the provider alias registry."""

    def test_all_aliases_have_required_keys(self):
        for name, alias in PROVIDER_ALIASES.items():
            assert "base_url" in alias, f"{name} missing base_url"
            assert "env_key" in alias, f"{name} missing env_key"
            assert alias["base_url"].startswith("https://"), f"{name} base_url not https"

    def test_known_aliases_exist(self):
        # openrouter is intentionally absent — verdict has a native OpenRouterProvider
        expected = {"together", "fireworks", "deepinfra", "groq"}
        assert expected.issubset(set(PROVIDER_ALIASES.keys()))
        assert "openrouter" not in PROVIDER_ALIASES, (
            "openrouter must not be in PROVIDER_ALIASES — it is handled natively by "
            "verdict's OpenRouterProvider which reads OPENROUTER_API_KEY directly. "
            "Adding it here caused confusing OPENAI_API_KEY errors."
        )


class TestResolveProvider:
    """Tests for _resolve_provider()."""

    def test_openrouter_passes_through(self):
        # openrouter is NOT in PROVIDER_ALIASES — it passes through to verdict's
        # native OpenRouterProvider which reads OPENROUTER_API_KEY itself.
        r = _resolve_provider("openrouter", "meta-llama/llama-3.1-8b-instruct")
        assert r["provider_name"] == "openrouter"
        assert r["model"] == "meta-llama/llama-3.1-8b-instruct"
        assert r["api_key"] is None   # key is left to OpenRouterProvider to resolve
        assert r["base_url"] is None

    def test_openrouter_missing_key_raises_correct_error(self, monkeypatch):
        """Regression: missing OPENROUTER_API_KEY must surface a clear ValueError,
        not the confusing 'OPENAI_API_KEY must be set' error that occurred when
        openrouter was aliased to the openai provider."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from aevyra_verdict.providers.openrouter_provider import OpenRouterProvider
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterProvider(model="meta-llama/llama-3.1-8b-instruct")

    def test_together_resolves(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "tog-test")
        r = _resolve_provider("together", "meta-llama/Llama-3.1-8B-Instruct")
        assert r["provider_name"] == "openai"
        assert "together" in r["base_url"]
        assert r["api_key"] == "tog-test"

    def test_groq_resolves(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        r = _resolve_provider("groq", "llama-3.1-8b-instant")
        assert r["provider_name"] == "openai"
        assert "groq" in r["base_url"]

    def test_native_openai_passthrough(self):
        r = _resolve_provider("openai", "gpt-4o-mini")
        assert r["provider_name"] == "openai"
        assert r["base_url"] is None
        assert r["api_key"] is None

    def test_local_passthrough(self):
        r = _resolve_provider("local", "llama3.2:1b")
        assert r["provider_name"] == "local"
        assert r["base_url"] is None

    def test_custom_api_key_passthrough(self):
        # Explicitly passed api_key propagates for any non-alias provider
        r = _resolve_provider("openrouter", "test-model", api_key="custom-key")
        assert r["api_key"] == "custom-key"

    def test_unknown_alias_passes_through(self):
        r = _resolve_provider("some_new_provider", "some-model")
        assert r["provider_name"] == "some_new_provider"
        assert r["base_url"] is None


class TestOptimizerConfig:
    """Tests for OptimizerConfig defaults."""

    def test_defaults(self):
        c = OptimizerConfig()
        assert c.max_iterations == 10
        assert c.score_threshold == 0.85
        assert c.strategy == "iterative"
        assert c.max_workers == 4

    def test_custom_values(self):
        c = OptimizerConfig(
            max_iterations=20,
            score_threshold=0.95,
            strategy="pdo",
            max_workers=8,
        )
        assert c.max_iterations == 20
        assert c.max_workers == 8


class TestOllamaDetection:
    """Tests for _check_parallel_config()."""

    def _make_optimizer(self, providers, strategy="structural", max_workers=4):
        config = OptimizerConfig(strategy=strategy, max_workers=max_workers)
        opt = PromptOptimizer(config)
        opt._providers = providers
        opt._metrics = ["dummy"]
        return opt, config

    def test_ollama_no_env_falls_back_to_1(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)
        opt, config = self._make_optimizer(
            [{"provider_name": "local", "model": "llama3.2:1b"}],
        )
        opt._check_parallel_config()
        assert config.max_workers == 1

    def test_ollama_with_env_caps_workers(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "2")
        opt, config = self._make_optimizer(
            [{"provider_name": "local", "model": "llama3.2:8b"}],
            max_workers=4,
        )
        opt._check_parallel_config()
        assert config.max_workers == 2

    def test_ollama_with_env_matching_workers(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_NUM_PARALLEL", "4")
        opt, config = self._make_optimizer(
            [{"provider_name": "local", "model": "llama3.2:8b"}],
            max_workers=4,
        )
        opt._check_parallel_config()
        assert config.max_workers == 4

    def test_openai_no_change(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)
        opt, config = self._make_optimizer(
            [{"provider_name": "openai", "model": "gpt-4o-mini"}],
        )
        opt._check_parallel_config()
        assert config.max_workers == 4

    def test_iterative_strategy_no_change(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)
        opt, config = self._make_optimizer(
            [{"provider_name": "local", "model": "llama3.2:1b"}],
            strategy="iterative",
        )
        opt._check_parallel_config()
        assert config.max_workers == 4  # iterative doesn't need parallel

    def test_auto_strategy_triggers_detection(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)
        opt, config = self._make_optimizer(
            [{"provider_name": "local", "model": "llama3.2:1b"}],
            strategy="auto",
        )
        opt._check_parallel_config()
        assert config.max_workers == 1

    def test_ollama_via_base_url(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_NUM_PARALLEL", raising=False)
        opt, config = self._make_optimizer(
            [{"provider_name": "openai", "model": "llama3.2:1b", "base_url": "http://localhost:11434/v1"}],
        )
        opt._check_parallel_config()
        assert config.max_workers == 1


class TestAddProvider:
    """Tests for PromptOptimizer.add_provider() with alias resolution."""

    def test_add_openrouter_resolves(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        opt = PromptOptimizer()
        opt.add_provider("openrouter", "meta-llama/llama-3.1-8b-instruct")
        p = opt._providers[0]
        assert p["provider_name"] == "openai"
        assert "openrouter" in p["base_url"]
        assert p["api_key"] == "sk-or-test"

    def test_add_local_passthrough(self):
        opt = PromptOptimizer()
        opt.add_provider("local", "llama3.2:1b")
        p = opt._providers[0]
        assert p["provider_name"] == "local"

    def test_add_provider_with_label(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "tog-test")
        opt = PromptOptimizer()
        opt.add_provider("together", "meta-llama/Llama-3.1-8B-Instruct", label="llama-together")
        p = opt._providers[0]
        assert p["label"] == "llama-together"


# ---------------------------------------------------------------------------
# Verdict results integration
# ---------------------------------------------------------------------------

# Sample verdict results JSON matching the real format
VERDICT_RESULTS_JSON = {
    "dataset": "test_data",
    "metrics": ["rouge_rougeL", "bleu"],
    "models": {
        "openai/gpt-4o-mini": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "success_rate": 1.0,
            "mean_latency_ms": 312.4,
            "total_tokens": 15230,
            "rouge_rougeL_mean": 0.8765,
            "rouge_rougeL_stdev": 0.0456,
            "bleu_mean": 0.7234,
            "bleu_stdev": 0.0891,
        },
        "local/llama3.1": {
            "provider": "local",
            "model": "llama3.1",
            "success_rate": 0.98,
            "mean_latency_ms": 89.1,
            "total_tokens": 12100,
            "rouge_rougeL_mean": 0.6541,
            "rouge_rougeL_stdev": 0.0723,
            "bleu_mean": 0.5123,
            "bleu_stdev": 0.1045,
        },
        "openrouter/qwen/qwen3.5-9b": {
            "provider": "openai",
            "model": "qwen/qwen3.5-9b",
            "success_rate": 1.0,
            "mean_latency_ms": 245.7,
            "total_tokens": 14500,
            "rouge_rougeL_mean": 0.7541,
            "rouge_rougeL_stdev": 0.0512,
            "bleu_mean": 0.6890,
            "bleu_stdev": 0.0678,
        },
    },
    "per_sample": [],
}


@pytest.fixture
def verdict_json_path(tmp_path):
    """Write the sample verdict results to a temp file."""
    path = tmp_path / "results.json"
    path.write_text(json.dumps(VERDICT_RESULTS_JSON))
    return path


class TestParseVerdictResults:
    """Tests for parse_verdict_results()."""

    def test_finds_best_model(self, verdict_json_path):
        parsed = parse_verdict_results(verdict_json_path)
        assert parsed["best_model"] == "openai/gpt-4o-mini"
        assert parsed["best_score"] == pytest.approx(0.8765)

    def test_uses_first_metric_by_default(self, verdict_json_path):
        parsed = parse_verdict_results(verdict_json_path)
        # First metric is rouge_rougeL, gpt-4o-mini has the highest
        assert parsed["best_score"] == pytest.approx(0.8765)

    def test_specific_metric_changes_ranking(self, verdict_json_path):
        # By BLEU, gpt-4o-mini is still best but with different score
        parsed = parse_verdict_results(verdict_json_path, metric="bleu")
        assert parsed["best_model"] == "openai/gpt-4o-mini"
        assert parsed["best_score"] == pytest.approx(0.7234)

    def test_returns_all_models(self, verdict_json_path):
        parsed = parse_verdict_results(verdict_json_path)
        assert len(parsed["models"]) == 3
        assert "openai/gpt-4o-mini" in parsed["models"]
        assert "local/llama3.1" in parsed["models"]

    def test_returns_metrics_list(self, verdict_json_path):
        parsed = parse_verdict_results(verdict_json_path)
        assert parsed["metrics"] == ["rouge_rougeL", "bleu"]

    def test_target_matches_best(self, verdict_json_path):
        parsed = parse_verdict_results(verdict_json_path)
        assert parsed["target_model"] == parsed["best_model"]
        assert parsed["target_score"] == parsed["best_score"]

    def test_model_info_structure(self, verdict_json_path):
        parsed = parse_verdict_results(verdict_json_path)
        gpt = parsed["models"]["openai/gpt-4o-mini"]
        assert gpt["provider"] == "openai"
        assert gpt["model"] == "gpt-4o-mini"
        assert "rouge_rougeL" in gpt["mean_scores"]
        assert "bleu" in gpt["mean_scores"]

    def test_empty_models_raises(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"dataset": "x", "metrics": ["rouge"], "models": {}}))
        with pytest.raises(ValueError, match="No model results"):
            parse_verdict_results(path)

    def test_no_metrics_raises(self, tmp_path):
        path = tmp_path / "nometric.json"
        path.write_text(json.dumps({"dataset": "x", "metrics": [], "models": {"a": {}}}))
        with pytest.raises(ValueError, match="No metrics"):
            parse_verdict_results(path)

    def test_single_model_is_its_own_target(self, tmp_path):
        data = {
            "dataset": "x",
            "metrics": ["rouge"],
            "models": {
                "only_model": {
                    "provider": "openai",
                    "model": "gpt-4o",
                    "rouge_mean": 0.72,
                },
            },
        }
        path = tmp_path / "single.json"
        path.write_text(json.dumps(data))
        parsed = parse_verdict_results(path)
        assert parsed["best_model"] == "only_model"
        assert parsed["best_score"] == pytest.approx(0.72)


class TestSetTargetFromVerdict:
    """Tests for PromptOptimizer.set_target_from_verdict()."""

    def test_sets_threshold_from_best(self, verdict_json_path):
        opt = PromptOptimizer()
        opt.set_target_from_verdict(verdict_json_path)
        assert opt.config.score_threshold == pytest.approx(0.8765)
        assert opt.config.target_model == "openai/gpt-4o-mini"
        assert opt.config.target_source == "verdict_json"

    def test_specific_metric(self, verdict_json_path):
        opt = PromptOptimizer()
        opt.set_target_from_verdict(verdict_json_path, metric="bleu")
        assert opt.config.score_threshold == pytest.approx(0.7234)

    def test_returns_self_for_chaining(self, verdict_json_path):
        opt = PromptOptimizer()
        result = opt.set_target_from_verdict(verdict_json_path)
        assert result is opt


class TestOptimizerConfigTargetFields:
    """Tests for the new target-related config fields."""

    def test_defaults_to_none(self):
        c = OptimizerConfig()
        assert c.target_model is None
        assert c.target_source is None

    def test_can_set_target_fields(self):
        c = OptimizerConfig(
            score_threshold=0.92,
            target_model="openai/gpt-4o",
            target_source="verdict_json",
        )
        assert c.target_model == "openai/gpt-4o"
        assert c.target_source == "verdict_json"



# ---------------------------------------------------------------------------
# baseline_override + branched_from
# ---------------------------------------------------------------------------

def _make_fake_optimizer():
    """Return a PromptOptimizer wired with mock providers/metrics/dataset."""
    optimizer = PromptOptimizer(config=OptimizerConfig(max_iterations=1, score_threshold=0.9, val_ratio=0.0))

    # Fake dataset
    ds = MagicMock()
    ds.conversations = []
    optimizer.set_dataset(ds)

    # Fake provider
    optimizer._providers = [{"provider_name": "openai", "model": "gpt-4o-mini", "label": "test"}]

    # Fake metric
    optimizer._metrics = [MagicMock()]

    return optimizer


class TestBaselineOverride(unittest.TestCase):
    """optimizer.run() with baseline_override skips the baseline eval."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=Path(self.tmpdir) / ".reflex")

    def _run_with_patches(self, override_baseline):
        """Return (optimizer, call_count, strategy_mock) with LLM and get_strategy patched."""
        optimizer = _make_fake_optimizer()
        call_count = {"n": 0}
        fake_snap = EvalSnapshot(mean_score=0.75, scores_by_metric={"rouge": 0.75})
        fake_result = OptimizationResult(
            best_prompt="improved",
            best_score=0.75,
            iterations=[IterationRecord(iteration=1, system_prompt="improved", score=0.75)],
            converged=False,
            baseline=override_baseline or fake_snap,
            final=fake_snap,
        )

        def fake_run_eval(prompt, dataset=None):
            call_count["n"] += 1
            return fake_snap

        optimizer._run_eval = fake_run_eval

        strategy_mock = MagicMock()
        strategy_mock.run.return_value = fake_result
        return optimizer, call_count, fake_snap, strategy_mock

    def test_baseline_override_skips_eval(self):
        override = EvalSnapshot(mean_score=0.58, scores_by_metric={"rouge": 0.58})
        optimizer, call_count, _, strategy_mock = self._run_with_patches(override)

        with patch("aevyra_reflex.optimizer.LLM"), \
             patch("aevyra_reflex.strategies.get_strategy", return_value=lambda: strategy_mock):
            optimizer.run(
                "You are helpful.",
                run_store=self.store,
                baseline_override=override,
            )

        # baseline skipped — only the final verification eval runs
        self.assertEqual(call_count["n"], 1)

    def test_without_baseline_override_eval_is_called(self):
        optimizer, call_count, _, strategy_mock = self._run_with_patches(None)

        with patch("aevyra_reflex.optimizer.LLM"), \
             patch("aevyra_reflex.strategies.get_strategy", return_value=lambda: strategy_mock):
            optimizer.run("You are helpful.", run_store=self.store)

        # baseline + final verification = at least 2 calls
        self.assertGreaterEqual(call_count["n"], 2)

    def test_baseline_override_saved_to_run(self):
        override = EvalSnapshot(mean_score=0.61, scores_by_metric={"rouge": 0.61})
        optimizer, _, _, strategy_mock = self._run_with_patches(override)

        with patch("aevyra_reflex.optimizer.LLM"), \
             patch("aevyra_reflex.strategies.get_strategy", return_value=lambda: strategy_mock):
            optimizer.run(
                "You are helpful.",
                run_store=self.store,
                baseline_override=override,
            )

        run = self.store.get_latest_run()
        self.assertIsNotNone(run)
        baseline = run.load_baseline()
        self.assertIsNotNone(baseline)
        self.assertAlmostEqual(baseline["mean_score"], 0.61)


class TestBranchedFromPassthrough(unittest.TestCase):
    """branched_from is stored in the run config when passed to optimizer.run()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=Path(self.tmpdir) / ".reflex")

    def _make_patched_optimizer(self):
        optimizer = _make_fake_optimizer()
        fake_snap = EvalSnapshot(mean_score=0.75, scores_by_metric={})
        fake_result = OptimizationResult(
            best_prompt="improved", best_score=0.75,
            iterations=[IterationRecord(iteration=1, system_prompt="improved", score=0.75)],
            converged=False, baseline=fake_snap, final=fake_snap,
        )
        optimizer._run_eval = MagicMock(return_value=fake_snap)
        strategy_mock = MagicMock()
        strategy_mock.run.return_value = fake_result
        return optimizer, strategy_mock

    def test_branched_from_saved_in_run_config(self):
        override = EvalSnapshot(mean_score=0.58, scores_by_metric={})
        optimizer, strategy_mock = self._make_patched_optimizer()

        with patch("aevyra_reflex.optimizer.LLM"), \
             patch("aevyra_reflex.strategies.get_strategy", return_value=lambda: strategy_mock):
            optimizer.run(
                "You are helpful.",
                run_store=self.store,
                baseline_override=override,
                branched_from={"run_id": "003", "iteration": 5},
            )

        run = self.store.get_latest_run()
        config = run.load_config()
        self.assertIn("branched_from", config)
        self.assertEqual(config["branched_from"]["run_id"], "003")
        self.assertEqual(config["branched_from"]["iteration"], 5)

    def test_no_branched_from_not_in_config(self):
        optimizer, strategy_mock = self._make_patched_optimizer()

        with patch("aevyra_reflex.optimizer.LLM"), \
             patch("aevyra_reflex.strategies.get_strategy", return_value=lambda: strategy_mock):
            optimizer.run("You are helpful.", run_store=self.store)

        run = self.store.get_latest_run()
        config = run.load_config()
        self.assertNotIn("branched_from", config)


# ---------------------------------------------------------------------------
# TestMaxWorkersRunConfig
# ---------------------------------------------------------------------------

class TestMaxWorkersRunConfig:
    """max_workers from OptimizerConfig is propagated to verdict RunConfig."""

    def _make_optimizer(self, max_workers=4):
        from aevyra_verdict.dataset import Conversation, Message
        from aevyra_verdict import Dataset, ExactMatch

        conversations = [
            Conversation(
                messages=[Message(role="user", content="hi")],
                ideal="hello",
            )
        ]
        ds = Dataset(conversations=conversations)
        config = OptimizerConfig(
            max_iterations=1,
            score_threshold=0.99,
            val_ratio=0.0,
            train_ratio=1.0,
            max_workers=max_workers,
        )
        opt = PromptOptimizer(config=config)
        opt.set_dataset(ds)
        opt._providers = [{"provider_name": "openai", "model": "gpt-4o-mini", "label": "test"}]
        opt._metrics = [ExactMatch()]
        return opt

    def test_max_workers_passed_to_run_config(self):
        """RunConfig is constructed with max_workers matching OptimizerConfig."""
        opt = self._make_optimizer(max_workers=8)

        captured = {}

        from aevyra_verdict.runner import RunConfig as _RealRunConfig

        def fake_run_config(**kwargs):
            captured.update(kwargs)
            return _RealRunConfig(**kwargs)

        with patch("aevyra_verdict.runner.RunConfig", side_effect=fake_run_config), \
             patch("aevyra_verdict.EvalRunner") as mock_runner:
            mock_runner.return_value.run.return_value = MagicMock(model_results={})
            opt._run_eval("You are helpful.")

        assert "max_workers" in captured, "max_workers not passed to RunConfig"
        assert captured["max_workers"] == 8

    def test_max_workers_default_is_four(self):
        """Default max_workers=4 is passed through."""
        opt = self._make_optimizer(max_workers=4)

        captured = {}

        from aevyra_verdict.runner import RunConfig as _RealRunConfig

        def fake_run_config(**kwargs):
            captured.update(kwargs)
            return _RealRunConfig(**kwargs)

        with patch("aevyra_verdict.runner.RunConfig", side_effect=fake_run_config), \
             patch("aevyra_verdict.EvalRunner") as mock_runner:
            mock_runner.return_value.run.return_value = MagicMock(model_results={})
            opt._run_eval("You are helpful.")

        assert captured.get("max_workers") == 4


# ---------------------------------------------------------------------------
# TestValSplitInRun
# ---------------------------------------------------------------------------

class TestValSplitInRun(unittest.TestCase):
    """optimizer.run() correctly splits dataset into train/val/test."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=Path(self.tmpdir) / ".reflex")

    def _make_optimizer_with_dataset(self, n_conversations=20, val_ratio=0.1):
        from aevyra_verdict.dataset import Conversation, Message
        from aevyra_verdict import Dataset, ExactMatch

        conversations = [
            Conversation(
                messages=[Message(role="user", content=f"item {i}")],
                ideal=f"answer {i}",
            )
            for i in range(n_conversations)
        ]
        ds = Dataset(conversations=conversations)
        config = OptimizerConfig(
            max_iterations=1,
            score_threshold=0.99,
            val_ratio=val_ratio,
            train_ratio=0.7,
        )
        opt = PromptOptimizer(config=config)
        opt.set_dataset(ds)
        opt._providers = [{"provider_name": "openai", "model": "gpt-4o-mini", "label": "test"}]
        opt._metrics = [ExactMatch()]
        return opt

    def test_val_split_produces_three_datasets(self):
        """With val_ratio=0.1, the run uses train/val/test (not just train/test)."""
        opt = self._make_optimizer_with_dataset(n_conversations=20, val_ratio=0.1)

        split_calls = []
        real_split = opt._split_dataset_3way

        def capturing_split(dataset, **kwargs):
            result = real_split(dataset, **kwargs)
            split_calls.append(result)
            return result

        fake_snap = EvalSnapshot(mean_score=0.5, scores_by_metric={})
        fake_result = OptimizationResult(
            best_prompt="improved", best_score=0.5,
            iterations=[IterationRecord(iteration=1, system_prompt="improved", score=0.5)],
            converged=False, baseline=fake_snap, final=fake_snap,
        )
        strategy_mock = MagicMock()
        strategy_mock.run.return_value = fake_result

        with patch("aevyra_reflex.optimizer.LLM"), \
             patch("aevyra_reflex.strategies.get_strategy", return_value=lambda: strategy_mock), \
             patch.object(opt, "_split_dataset_3way", side_effect=capturing_split), \
             patch.object(opt, "_run_eval", return_value=fake_snap):
            opt.run("You are helpful.", run_store=self.store)

        self.assertEqual(len(split_calls), 1, "3-way split should be called once")
        train_ds, val_ds, test_ds = split_calls[0]
        self.assertIsNotNone(val_ds, "val dataset should not be None")
        self.assertGreater(len(train_ds.conversations), 0)
        self.assertGreater(len(val_ds.conversations), 0)
        self.assertGreater(len(test_ds.conversations), 0)
        # Approximate split ratios
        total = len(train_ds.conversations) + len(val_ds.conversations) + len(test_ds.conversations)
        self.assertEqual(total, 20)

    def test_val_split_disabled_produces_two_datasets(self):
        """With val_ratio=0.0, only train/test split is used."""
        opt = self._make_optimizer_with_dataset(n_conversations=20, val_ratio=0.0)

        split_calls = []
        real_split = opt._split_dataset

        def capturing_split(dataset, *args, **kwargs):
            result = real_split(dataset, *args, **kwargs)
            split_calls.append(result)
            return result

        fake_snap = EvalSnapshot(mean_score=0.5, scores_by_metric={})
        fake_result = OptimizationResult(
            best_prompt="improved", best_score=0.5,
            iterations=[IterationRecord(iteration=1, system_prompt="improved", score=0.5)],
            converged=False, baseline=fake_snap, final=fake_snap,
        )
        strategy_mock = MagicMock()
        strategy_mock.run.return_value = fake_result

        with patch("aevyra_reflex.optimizer.LLM"), \
             patch("aevyra_reflex.strategies.get_strategy", return_value=lambda: strategy_mock), \
             patch.object(opt, "_split_dataset", side_effect=capturing_split), \
             patch.object(opt, "_run_eval", return_value=fake_snap):
            opt.run("You are helpful.", run_store=self.store)

        self.assertEqual(len(split_calls), 1, "2-way split should be called once")
        train_ds, test_ds = split_calls[0]
        total = len(train_ds.conversations) + len(test_ds.conversations)
        self.assertEqual(total, 20)


# ---------------------------------------------------------------------------
# TestConvergenceBasedOnTestScore
# ---------------------------------------------------------------------------

class TestConvergenceBasedOnTestScore(unittest.TestCase):
    """result.converged reflects the held-out test score, not the training score."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=Path(self.tmpdir) / ".reflex")

    def _run_with_scores(self, baseline_score, final_score, threshold):
        from aevyra_verdict.dataset import Conversation, Message
        from aevyra_verdict import Dataset, ExactMatch

        conversations = [
            Conversation(
                messages=[Message(role="user", content="q")],
                ideal="a",
            )
        ]
        ds = Dataset(conversations=conversations)
        config = OptimizerConfig(
            max_iterations=1,
            score_threshold=threshold,
            val_ratio=0.0,
            train_ratio=1.0,
        )
        opt = PromptOptimizer(config=config)
        opt.set_dataset(ds)
        opt._providers = [{"provider_name": "openai", "model": "gpt-4o-mini", "label": "test"}]
        opt._metrics = [ExactMatch()]

        baseline_snap = EvalSnapshot(mean_score=baseline_score, scores_by_metric={})
        final_snap = EvalSnapshot(mean_score=final_score, scores_by_metric={})
        fake_result = OptimizationResult(
            best_prompt="improved", best_score=final_score,
            iterations=[IterationRecord(iteration=1, system_prompt="improved", score=final_score)],
            converged=False, baseline=baseline_snap, final=final_snap,
        )
        strategy_mock = MagicMock()
        strategy_mock.run.return_value = fake_result

        eval_calls = []

        def fake_run_eval(prompt, dataset=None):
            snap = final_snap if eval_calls else baseline_snap
            eval_calls.append(prompt)
            return snap

        with patch("aevyra_reflex.optimizer.LLM"), \
             patch("aevyra_reflex.strategies.get_strategy", return_value=lambda: strategy_mock), \
             patch.object(opt, "_run_eval", side_effect=fake_run_eval):
            return opt.run("You are helpful.", run_store=self.store)

    def test_converged_true_when_final_score_meets_threshold(self):
        """converged=True when final test score >= score_threshold."""
        result = self._run_with_scores(
            baseline_score=0.5,
            final_score=0.9,
            threshold=0.85,
        )
        self.assertTrue(result.converged)

    def test_converged_false_when_final_score_below_threshold(self):
        """converged=False when final test score < score_threshold."""
        result = self._run_with_scores(
            baseline_score=0.5,
            final_score=0.7,
            threshold=0.85,
        )
        self.assertFalse(result.converged)

    def test_converged_false_when_baseline_meets_threshold_but_final_does_not(self):
        """converged is based on final test score, not training score."""
        result = self._run_with_scores(
            baseline_score=0.9,  # baseline exceeds threshold
            final_score=0.7,     # but final (test) does not
            threshold=0.85,
        )
        self.assertFalse(result.converged)


# ---------------------------------------------------------------------------
# TestResumeSkipsBaseline
# ---------------------------------------------------------------------------

class TestResumeSkipsBaseline(unittest.TestCase):
    """optimizer.run() with resume_run reuses saved baseline and skips re-evaluation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = RunStore(root=Path(self.tmpdir) / ".reflex")

    def test_resume_skips_baseline_eval(self):
        """When resuming, baseline eval is not called again."""
        from aevyra_verdict.dataset import Conversation, Message
        from aevyra_verdict import Dataset, ExactMatch
        from aevyra_reflex.run_store import CheckpointState

        conversations = [
            Conversation(messages=[Message(role="user", content="q")], ideal="a")
        ]
        ds = Dataset(conversations=conversations)
        config = OptimizerConfig(
            max_iterations=1,
            score_threshold=0.99,
            val_ratio=0.0,
            train_ratio=1.0,
        )
        opt = PromptOptimizer(config=config)
        opt.set_dataset(ds)
        opt._providers = [{"provider_name": "openai", "model": "gpt-4o-mini", "label": "test"}]
        opt._metrics = [ExactMatch()]

        # Create a run with a saved checkpoint containing a baseline
        run = self.store.create_run(
            config={}, dataset_path="", prompt_path="", initial_prompt="initial"
        )
        checkpoint = CheckpointState(
            run_id=run.run_id,
            initial_prompt="initial",
            current_prompt="improved v1",
            completed_iterations=1,
            best_prompt="improved v1",
            best_score=0.6,
            baseline={"mean_score": 0.4, "scores_by_metric": {}},
        )
        run.save_checkpoint(checkpoint)

        fake_snap = EvalSnapshot(mean_score=0.65, scores_by_metric={})
        fake_result = OptimizationResult(
            best_prompt="improved v2", best_score=0.65,
            iterations=[IterationRecord(iteration=2, system_prompt="improved v2", score=0.65)],
            converged=False, baseline=fake_snap, final=fake_snap,
        )
        strategy_mock = MagicMock()
        strategy_mock.run.return_value = fake_result

        eval_calls = []

        def fake_run_eval(prompt, dataset=None):
            eval_calls.append(prompt)
            return fake_snap

        with patch("aevyra_reflex.optimizer.LLM"), \
             patch("aevyra_reflex.strategies.get_strategy", return_value=lambda: strategy_mock), \
             patch.object(opt, "_run_eval", side_effect=fake_run_eval):
            opt.run("initial", run_store=self.store, resume_run=run)

        # Only the final verification eval should fire, not a fresh baseline
        self.assertEqual(len(eval_calls), 1, "Baseline eval should be skipped on resume")

    def test_resume_uses_saved_baseline_score(self):
        """Resumed run uses the checkpoint baseline score, not a fresh eval."""
        from aevyra_verdict.dataset import Conversation, Message
        from aevyra_verdict import Dataset, ExactMatch
        from aevyra_reflex.run_store import CheckpointState

        conversations = [
            Conversation(messages=[Message(role="user", content="q")], ideal="a")
        ]
        ds = Dataset(conversations=conversations)
        opt = PromptOptimizer(config=OptimizerConfig(
            max_iterations=1, score_threshold=0.99, val_ratio=0.0, train_ratio=1.0
        ))
        opt.set_dataset(ds)
        opt._providers = [{"provider_name": "openai", "model": "gpt-4o-mini", "label": "test"}]
        opt._metrics = [ExactMatch()]

        saved_baseline_score = 0.42
        run = self.store.create_run(
            config={}, dataset_path="", prompt_path="", initial_prompt="initial"
        )
        checkpoint = CheckpointState(
            run_id=run.run_id,
            initial_prompt="initial",
            current_prompt="improved",
            completed_iterations=1,
            best_prompt="improved",
            best_score=0.7,
            baseline={"mean_score": saved_baseline_score, "scores_by_metric": {}},
        )
        run.save_checkpoint(checkpoint)

        fake_snap = EvalSnapshot(mean_score=0.7, scores_by_metric={})
        fake_result = OptimizationResult(
            best_prompt="improved", best_score=0.7,
            iterations=[IterationRecord(iteration=2, system_prompt="improved", score=0.7)],
            converged=False, baseline=fake_snap, final=fake_snap,
        )
        strategy_mock = MagicMock()
        strategy_mock.run.return_value = fake_result

        with patch("aevyra_reflex.optimizer.LLM"), \
             patch("aevyra_reflex.strategies.get_strategy", return_value=lambda: strategy_mock), \
             patch.object(opt, "_run_eval", return_value=fake_snap):
            result = opt.run("initial", run_store=self.store, resume_run=run)

        self.assertAlmostEqual(result.baseline.mean_score, saved_baseline_score)
