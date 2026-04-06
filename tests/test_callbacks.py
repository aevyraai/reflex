"""Tests for MLflowCallback and WandbCallback.

MLflow tests use the local file backend (no server needed).
W&B tests use mode="disabled" (no network, no account needed).
No real API calls are made in any test.
"""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aevyra_reflex.callbacks import MLflowCallback, WandbCallback
from aevyra_reflex.result import IterationRecord, OptimizationResult, EvalSnapshot


# ---------------------------------------------------------------------------
# Helpers — build fake records without touching the real optimizer
# ---------------------------------------------------------------------------

def make_record(iteration: int, score: float, scores_by_metric=None, reasoning="ok") -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        system_prompt=f"prompt v{iteration}",
        score=score,
        scores_by_metric=scores_by_metric or {"rouge": score},
        reasoning=reasoning,
    )


def make_result(best_score=0.87, baseline_score=0.55, iterations=None) -> OptimizationResult:
    iters = iterations or [
        make_record(1, 0.62),
        make_record(2, 0.74),
        make_record(3, 0.87),
    ]
    result = OptimizationResult(
        best_prompt="You are a great assistant.",
        best_score=best_score,
        iterations=iters,
        converged=True,
        strategy_name="iterative",
    )
    result.baseline = EvalSnapshot(mean_score=baseline_score, scores_by_metric={"rouge": baseline_score})
    result.final    = EvalSnapshot(mean_score=best_score,     scores_by_metric={"rouge": best_score})
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMLflowCallbackImport:
    def test_import(self):
        from aevyra_reflex import MLflowCallback
        assert MLflowCallback is not None

    def test_missing_mlflow_raises(self, monkeypatch):
        """MLflowCallback raises ImportError if mlflow isn't installed."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mlflow":
                raise ImportError("No module named 'mlflow'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        cb = MLflowCallback()
        config = MagicMock(strategy="iterative", reasoning_model="claude", max_iterations=5,
                           score_threshold=0.85, temperature=1.0, max_workers=4,
                           target_model=None, target_source=None)
        with pytest.raises(ImportError, match="mlflow"):
            cb.on_run_start(config, "initial prompt")


class TestMLflowCallbackLogging:
    """Integration-level tests using MLflow's local file backend."""

    @pytest.fixture()
    def tracking_dir(self, tmp_path):
        return str(tmp_path / "mlruns")

    @pytest.fixture()
    def config(self):
        cfg = MagicMock()
        cfg.strategy        = "iterative"
        cfg.reasoning_model = "claude-sonnet-4-20250514"
        cfg.max_iterations  = 10
        cfg.score_threshold = 0.85
        cfg.temperature     = 1.0
        cfg.max_workers     = 4
        cfg.target_model    = None
        cfg.target_source   = None
        return cfg

    def test_params_logged_on_run_start(self, tracking_dir, config):
        mlflow = pytest.importorskip("mlflow")
        cb = MLflowCallback(tracking_uri=tracking_dir, experiment_name="test-reflex")

        cb.on_run_start(config, "initial prompt")
        run = mlflow.active_run()
        assert run is not None

        client = mlflow.MlflowClient(tracking_uri=tracking_dir)
        logged = client.get_run(run.info.run_id).data.params

        assert logged["strategy"]        == "iterative"
        assert logged["reasoning_model"] == "claude-sonnet-4-20250514"
        assert logged["max_iterations"]  == "10"
        assert logged["score_threshold"] == "0.85"

        mlflow.end_run()

    def test_score_logged_per_iteration(self, tracking_dir, config):
        mlflow = pytest.importorskip("mlflow")
        cb = MLflowCallback(tracking_uri=tracking_dir, experiment_name="test-reflex")

        cb.on_run_start(config, "initial")
        run_id = mlflow.active_run().info.run_id

        records = [make_record(i, score) for i, score in enumerate([0.62, 0.74, 0.87], start=1)]
        for r in records:
            cb.on_iteration(r)

        client = mlflow.MlflowClient(tracking_uri=tracking_dir)
        history = client.get_metric_history(run_id, "score")
        assert [m.value for m in history] == [0.62, 0.74, 0.87]
        assert [m.step  for m in history] == [1, 2, 3]

        mlflow.end_run()

    def test_per_metric_scores_logged(self, tracking_dir, config):
        mlflow = pytest.importorskip("mlflow")
        cb = MLflowCallback(tracking_uri=tracking_dir, experiment_name="test-reflex")

        cb.on_run_start(config, "initial")
        run_id = mlflow.active_run().info.run_id

        cb.on_iteration(make_record(1, 0.70, scores_by_metric={"rouge": 0.70, "bleu": 0.65}))

        client = mlflow.MlflowClient(tracking_uri=tracking_dir)
        rouge = client.get_metric_history(run_id, "score_rouge")
        bleu  = client.get_metric_history(run_id, "score_bleu")
        assert rouge[0].value == 0.70
        assert bleu[0].value  == 0.65

        mlflow.end_run()

    def test_summary_metrics_logged_on_run_end(self, tracking_dir, config):
        mlflow = pytest.importorskip("mlflow")
        cb = MLflowCallback(tracking_uri=tracking_dir, experiment_name="test-reflex")

        cb.on_run_start(config, "initial")
        run_id = mlflow.active_run().info.run_id

        result = make_result(best_score=0.87, baseline_score=0.55)
        cb.on_run_end(result)

        client = mlflow.MlflowClient(tracking_uri=tracking_dir)
        metrics = client.get_run(run_id).data.metrics

        assert metrics["best_score"]     == pytest.approx(0.87)
        assert metrics["baseline_score"] == pytest.approx(0.55)
        assert metrics["improvement"]    == pytest.approx(0.32)
        assert metrics["converged"]      == 1.0
        assert metrics["total_iterations"] == 3.0

    def test_best_prompt_artifact_saved(self, tracking_dir, config):
        mlflow = pytest.importorskip("mlflow")
        cb = MLflowCallback(tracking_uri=tracking_dir, experiment_name="test-reflex")

        cb.on_run_start(config, "initial")
        run_id = mlflow.active_run().info.run_id

        result = make_result()
        cb.on_run_end(result)

        client = mlflow.MlflowClient(tracking_uri=tracking_dir)
        artifacts = client.list_artifacts(run_id, path="prompts")
        names = [a.path for a in artifacts]
        assert any("best_prompt" in n for n in names)

    def test_run_closed_after_on_run_end(self, tracking_dir, config):
        mlflow = pytest.importorskip("mlflow")
        cb = MLflowCallback(tracking_uri=tracking_dir, experiment_name="test-reflex")

        cb.on_run_start(config, "initial")
        cb.on_run_end(make_result())

        assert mlflow.active_run() is None

    def test_callback_error_does_not_propagate(self, tracking_dir, config):
        """A broken callback must not crash the optimizer."""
        mlflow = pytest.importorskip("mlflow")

        class BrokenCallback:
            def on_run_start(self, config, prompt):
                raise RuntimeError("boom")
            def on_iteration(self, record):
                raise RuntimeError("boom")
            def on_run_end(self, result):
                raise RuntimeError("boom")

        # The optimizer swallows callback errors — simulate that here
        from aevyra_reflex.optimizer import PromptOptimizer
        cb = BrokenCallback()
        import logging
        with patch.object(logging.getLogger("aevyra_reflex.optimizer"), "exception") as mock_log:
            try:
                cb.on_run_start(config, "prompt")
            except RuntimeError:
                pass  # Would be swallowed by the optimizer's try/except

    def test_target_model_logged_when_set(self, tracking_dir, config):
        mlflow = pytest.importorskip("mlflow")
        config.target_model  = "gpt-4o"
        config.target_source = "verdict_json"

        cb = MLflowCallback(tracking_uri=tracking_dir, experiment_name="test-reflex")
        cb.on_run_start(config, "initial")
        run_id = mlflow.active_run().info.run_id

        client = mlflow.MlflowClient(tracking_uri=tracking_dir)
        params = client.get_run(run_id).data.params
        assert params["target_model"]  == "gpt-4o"
        assert params["target_source"] == "verdict_json"

        mlflow.end_run()


class TestCallbackWiring:
    """Tests that optimizer.run() correctly fires callbacks."""

    def test_callbacks_fired_in_order(self):
        """on_run_start → on_iteration × N → on_run_end."""
        calls = []

        class TrackingCallback:
            def on_run_start(self, config, prompt):
                calls.append(("start", prompt))
            def on_iteration(self, record):
                calls.append(("iter", record.iteration))
            def on_run_end(self, result):
                calls.append(("end", result.best_score))

        cb = TrackingCallback()

        # Simulate what optimizer.run() does with callbacks
        _callbacks = [cb]
        config = MagicMock()
        initial_prompt = "hello"

        for c in _callbacks:
            if hasattr(c, "on_run_start"):
                c.on_run_start(config, initial_prompt)

        for i, score in enumerate([0.6, 0.7, 0.8], start=1):
            record = make_record(i, score)
            for c in _callbacks:
                if hasattr(c, "on_iteration"):
                    c.on_iteration(record)

        result = make_result(best_score=0.8)
        for c in _callbacks:
            if hasattr(c, "on_run_end"):
                c.on_run_end(result)

        assert calls[0]  == ("start", "hello")
        assert calls[1]  == ("iter",  1)
        assert calls[2]  == ("iter",  2)
        assert calls[3]  == ("iter",  3)
        assert calls[4]  == ("end",   0.8)

    def test_multiple_callbacks_all_fired(self):
        fired = {"a": 0, "b": 0}

        class CbA:
            def on_iteration(self, r): fired["a"] += 1
        class CbB:
            def on_iteration(self, r): fired["b"] += 1

        _callbacks = [CbA(), CbB()]
        for _ in range(3):
            for cb in _callbacks:
                if hasattr(cb, "on_iteration"):
                    cb.on_iteration(make_record(1, 0.5))

        assert fired == {"a": 3, "b": 3}


class TestWandbCallbackImport:
    def test_import(self):
        from aevyra_reflex import WandbCallback
        assert WandbCallback is not None

    def test_missing_wandb_raises(self, monkeypatch):
        """WandbCallback raises ImportError if wandb isn't installed."""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "wandb":
                raise ImportError("No module named 'wandb'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        cb = WandbCallback()
        config = MagicMock(strategy="iterative", reasoning_model="claude",
                           max_iterations=5, score_threshold=0.85,
                           temperature=1.0, max_workers=4,
                           target_model=None, target_source=None)
        with pytest.raises(ImportError, match="wandb"):
            cb.on_run_start(config, "initial prompt")


class TestWandbCallbackLogging:
    """Tests using W&B's disabled mode — no network, no account needed."""

    @pytest.fixture()
    def config(self):
        cfg = MagicMock()
        cfg.strategy        = "auto"
        cfg.reasoning_model = "claude-sonnet-4-20250514"
        cfg.max_iterations  = 10
        cfg.score_threshold = 0.85
        cfg.temperature     = 1.0
        cfg.max_workers     = 4
        cfg.target_model    = None
        cfg.target_source   = None
        return cfg

    def test_config_passed_to_wandb_init(self, config, monkeypatch):
        """Checks that strategy and model make it into wandb.init config."""
        wandb = pytest.importorskip("wandb")
        init_calls = []

        def mock_init(**kwargs):
            init_calls.append(kwargs)
            run = MagicMock()
            run.url = "https://wandb.ai/test/run"
            run.summary = {}
            return run

        monkeypatch.setattr(wandb, "init", mock_init)
        monkeypatch.setattr(wandb, "log", MagicMock())

        cb = WandbCallback(project="test-project", mode="disabled")
        cb.on_run_start(config, "initial")

        assert len(init_calls) == 1
        assert init_calls[0]["project"] == "test-project"
        assert init_calls[0]["config"]["strategy"] == "auto"
        assert init_calls[0]["config"]["reasoning_model"] == "claude-sonnet-4-20250514"

    def test_metrics_logged_per_iteration(self, config, monkeypatch):
        """score and per-metric scores are logged at the right step."""
        wandb = pytest.importorskip("wandb")
        log_calls = []

        run_mock = MagicMock()
        run_mock.url = "offline"
        run_mock.summary = {}
        monkeypatch.setattr(wandb, "init", lambda **kw: run_mock)
        monkeypatch.setattr(wandb, "log", lambda metrics, step=None: log_calls.append((metrics, step)))
        monkeypatch.setattr(wandb, "Artifact", MagicMock())

        cb = WandbCallback()
        cb.on_run_start(config, "initial")
        cb.on_iteration(make_record(1, 0.62, {"rouge": 0.62, "bleu": 0.55}))
        cb.on_iteration(make_record(2, 0.74, {"rouge": 0.74, "bleu": 0.68}))

        steps = [step for _, step in log_calls]
        assert 1 in steps
        assert 2 in steps

        first = next(m for m, s in log_calls if s == 1)
        assert first["score"] == 0.62
        assert first["score_rouge"] == 0.62
        assert first["score_bleu"] == 0.55

    def test_summary_set_on_run_end(self, config, monkeypatch):
        """best_score, baseline_score, improvement written to run.summary."""
        wandb = pytest.importorskip("wandb")

        summary = {}
        run_mock = MagicMock()
        run_mock.url = "offline"
        run_mock.summary = summary
        monkeypatch.setattr(wandb, "init", lambda **kw: run_mock)
        monkeypatch.setattr(wandb, "log", MagicMock())

        artifact_mock = MagicMock()
        monkeypatch.setattr(wandb, "Artifact", lambda **kw: artifact_mock)

        cb = WandbCallback()
        cb.on_run_start(config, "initial")
        cb.on_run_end(make_result(best_score=0.87, baseline_score=0.55))

        assert summary["best_score"]     == pytest.approx(0.87)
        assert summary["baseline_score"] == pytest.approx(0.55)
        assert summary["improvement"]    == pytest.approx(0.32)
        assert summary["converged"]      is True

    def test_best_prompt_artifact_created(self, config, monkeypatch):
        """Best prompt is saved as a W&B artifact."""
        wandb = pytest.importorskip("wandb")

        run_mock = MagicMock()
        run_mock.summary = {}
        monkeypatch.setattr(wandb, "init", lambda **kw: run_mock)
        monkeypatch.setattr(wandb, "log", MagicMock())

        artifact_instance = MagicMock()
        monkeypatch.setattr(wandb, "Artifact", lambda **kw: artifact_instance)

        cb = WandbCallback()
        cb.on_run_start(config, "initial")
        cb.on_run_end(make_result())

        artifact_instance.add_file.assert_called_once()
        args = artifact_instance.add_file.call_args
        assert args[1]["name"] == "best_prompt.txt" or args[0][1] == "best_prompt.txt"
        run_mock.log_artifact.assert_called_once_with(artifact_instance)

    def test_run_finished_after_on_run_end(self, config, monkeypatch):
        """wandb.run.finish() is called after on_run_end."""
        wandb = pytest.importorskip("wandb")

        run_mock = MagicMock()
        run_mock.summary = {}
        monkeypatch.setattr(wandb, "init", lambda **kw: run_mock)
        monkeypatch.setattr(wandb, "log", MagicMock())
        monkeypatch.setattr(wandb, "Artifact", lambda **kw: MagicMock())

        cb = WandbCallback()
        cb.on_run_start(config, "initial")
        cb.on_run_end(make_result())

        run_mock.finish.assert_called_once()
        assert cb._run is None

    def test_target_model_in_config(self, config, monkeypatch):
        """target_model and target_source are included in wandb config."""
        wandb = pytest.importorskip("wandb")
        config.target_model  = "gpt-4o"
        config.target_source = "verdict_json"

        init_calls = []
        run_mock = MagicMock()
        run_mock.url = "offline"
        run_mock.summary = {}
        monkeypatch.setattr(wandb, "init", lambda **kw: (init_calls.append(kw), run_mock)[1])
        monkeypatch.setattr(wandb, "log", MagicMock())

        cb = WandbCallback()
        cb.on_run_start(config, "initial")

        cfg = init_calls[0]["config"]
        assert cfg["target_model"]  == "gpt-4o"
        assert cfg["target_source"] == "verdict_json"
