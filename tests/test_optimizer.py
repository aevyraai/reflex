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
        expected = {"openrouter", "together", "fireworks", "deepinfra", "groq"}
        assert expected.issubset(set(PROVIDER_ALIASES.keys()))


class TestResolveProvider:
    """Tests for _resolve_provider()."""

    def test_openrouter_resolves(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        r = _resolve_provider("openrouter", "meta-llama/llama-3.1-8b-instruct")
        assert r["provider_name"] == "openai"
        assert r["base_url"] == "https://openrouter.ai/api/v1"
        assert r["api_key"] == "sk-or-test"
        assert r["model"] == "meta-llama/llama-3.1-8b-instruct"

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

    def test_custom_overrides_alias_defaults(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-default")
        r = _resolve_provider(
            "openrouter", "test-model",
            api_key="custom-key",
            base_url="https://custom.url/v1",
        )
        assert r["api_key"] == "custom-key"
        assert r["base_url"] == "https://custom.url/v1"

    def test_falls_back_to_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
        r = _resolve_provider("openrouter", "some-model")
        assert r["api_key"] == "sk-openai-fallback"

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
    optimizer = PromptOptimizer(config=OptimizerConfig(max_iterations=1, score_threshold=0.9))

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

    def _make_optimizer_with_mocked_run_eval(self, override_baseline):
        """Return (optimizer, eval_call_count_tracker)."""
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

        def fake_run_eval(prompt):
            call_count["n"] += 1
            return fake_snap

        def fake_strategy_run(**kwargs):
            return fake_result

        optimizer._run_eval = fake_run_eval

        # Patch strategy so we don't need real LLM
        strategy_mock = MagicMock()
        strategy_mock.run.return_value = fake_result
        optimizer._get_strategy = MagicMock(return_value=strategy_mock)

        return optimizer, call_count

    def test_baseline_override_skips_eval(self):
        override = EvalSnapshot(mean_score=0.58, scores_by_metric={"rouge": 0.58})
        optimizer, call_count = self._make_optimizer_with_mocked_run_eval(override)

        optimizer.run(
            "You are helpful.",
            run_store=self.store,
            baseline_override=override,
        )

        # _run_eval should NOT have been called for the baseline
        self.assertEqual(call_count["n"], 0)

    def test_without_baseline_override_eval_is_called(self):
        optimizer, call_count = self._make_optimizer_with_mocked_run_eval(None)

        optimizer.run("You are helpful.", run_store=self.store)

        # _run_eval called once for baseline, once for final verify = 2
        self.assertGreaterEqual(call_count["n"], 1)

    def test_baseline_override_saved_to_run(self):
        override = EvalSnapshot(mean_score=0.61, scores_by_metric={"rouge": 0.61})
        optimizer, _ = self._make_optimizer_with_mocked_run_eval(override)

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

    def test_branched_from_saved_in_run_config(self):
        override = EvalSnapshot(mean_score=0.58, scores_by_metric={})
        optimizer = _make_fake_optimizer()

        fake_snap = EvalSnapshot(mean_score=0.75, scores_by_metric={})
        fake_result = OptimizationResult(
            best_prompt="improved", best_score=0.75,
            iterations=[IterationRecord(iteration=1, system_prompt="improved", score=0.75)],
            converged=False, baseline=override, final=fake_snap,
        )
        optimizer._run_eval = MagicMock(return_value=fake_snap)
        strategy_mock = MagicMock()
        strategy_mock.run.return_value = fake_result
        optimizer._get_strategy = MagicMock(return_value=strategy_mock)

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
        optimizer._get_strategy = MagicMock(return_value=strategy_mock)

        optimizer.run("You are helpful.", run_store=self.store)

        run = self.store.get_latest_run()
        config = run.load_config()
        self.assertNotIn("branched_from", config)
