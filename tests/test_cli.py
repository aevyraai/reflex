"""Tests for CLI argument parsing and metric resolution."""

import importlib
import sys

import pytest
from unittest.mock import MagicMock

from aevyra_reflex.optimizer import PromptOptimizer


# Pre-install a mock typer if the real one isn't available
try:
    import typer as _real_typer  # noqa: F401
except ImportError:
    class _MockExit(SystemExit):
        """Stand-in for typer.Exit that accepts code= kwarg."""
        def __init__(self, code=0):
            super().__init__(code)

    _mock_typer = MagicMock()
    _mock_typer.Typer.return_value = MagicMock()
    _mock_typer.Option = lambda *a, **kw: None
    _mock_typer.Argument = lambda *a, **kw: None
    _mock_typer.Exit = _MockExit
    _mock_typer.echo = lambda *a, **kw: None
    sys.modules["typer"] = _mock_typer

# Resolve the actual Exit exception class that typer.Exit will raise.
# When real typer/click is installed, typer.Exit is click.exceptions.Exit
# (which does NOT inherit from SystemExit).  Our mock inherits from
# SystemExit.  Use a tuple so pytest.raises catches either.
import typer as _typer_mod  # already real or mocked at this point

_TyperExit = _typer_mod.Exit
_exit_exceptions = (_TyperExit, SystemExit)  # catch both


# Shared mocks for aevyra_verdict
_mock_verdict = MagicMock()
_mock_verdict.RougeScore = lambda: MagicMock()
_mock_verdict.BleuScore = lambda: MagicMock()
_mock_verdict.ExactMatch = lambda: MagicMock()
_mock_verdict.LLMJudge = lambda judge_provider=None, criteria=None: MagicMock()

_mock_verdict_providers = MagicMock()
_mock_verdict_providers.get_provider = lambda *a, **kw: MagicMock()


class TestMetricResolution:
    """Tests for _add_metrics() logic."""

    def _call_add_metrics(self, metric_names, judge=None):
        """Helper — imports and calls _add_metrics, returns the optimizer's metrics list."""
        # Patch verdict modules into sys.modules *persistently* for the call
        old_verdict = sys.modules.get("aevyra_verdict")
        old_providers = sys.modules.get("aevyra_verdict.providers")
        try:
            sys.modules["aevyra_verdict"] = _mock_verdict
            sys.modules["aevyra_verdict.providers"] = _mock_verdict_providers

            from aevyra_reflex import cli
            importlib.reload(cli)

            optimizer = PromptOptimizer()
            cli._add_metrics(optimizer, metric_names, judge)
            return optimizer._metrics
        finally:
            # Restore original state
            if old_verdict is not None:
                sys.modules["aevyra_verdict"] = old_verdict
            else:
                sys.modules.pop("aevyra_verdict", None)
            if old_providers is not None:
                sys.modules["aevyra_verdict.providers"] = old_providers
            else:
                sys.modules.pop("aevyra_verdict.providers", None)

    def test_default_rouge_when_no_args(self):
        metrics = self._call_add_metrics([], None)
        assert len(metrics) == 1  # rouge added as default

    def test_judge_only(self):
        metrics = self._call_add_metrics([], "openai/gpt-4o-mini")
        assert len(metrics) == 1  # just the judge

    def test_metric_and_judge_errors(self):
        """Using both --metric and --judge should raise typer.Exit."""
        with pytest.raises(_exit_exceptions):
            self._call_add_metrics(["rouge"], "openai/gpt-4o-mini")

    def test_explicit_rouge(self):
        metrics = self._call_add_metrics(["rouge"], None)
        assert len(metrics) == 1

    def test_multiple_metrics(self):
        metrics = self._call_add_metrics(["rouge", "bleu"], None)
        assert len(metrics) == 2

    def test_unknown_metric_errors(self):
        with pytest.raises(_exit_exceptions):
            self._call_add_metrics(["nonexistent"], None)


class TestCLICallbackFlags:
    """Tests that --mlflow and --wandb flags build the right callbacks."""

    def _run_optimize_dry(self, extra_args: dict):
        """
        Simulate the callback-building block of optimize() without running
        a real optimization. Patches out all I/O and returns the callbacks list
        that would have been passed to optimizer.run().
        """
        captured = {}

        # Minimal stubs
        mock_ds = MagicMock()
        mock_ds.conversations = [MagicMock()] * 10

        mock_optimizer = MagicMock()
        mock_optimizer._providers = []
        mock_optimizer._metrics = []

        mock_result = MagicMock()
        mock_result.summary.return_value = "done"
        mock_result.best_prompt = "best"
        mock_result.converged = True

        def fake_run(*a, callbacks=None, **kw):
            captured["callbacks"] = callbacks or []
            return mock_result

        mock_optimizer.run = fake_run

        old_modules = {}
        patches = {
            "aevyra_verdict": _mock_verdict,
            "aevyra_verdict.providers": _mock_verdict_providers,
        }

        # Stub MLflowCallback and WandbCallback in aevyra_reflex.callbacks
        mock_callbacks_mod = MagicMock()
        class _FakeMLflow:
            def __init__(self, experiment_name=None, tracking_uri=None, **kw):
                self.experiment_name = experiment_name
                self.tracking_uri = tracking_uri
        class _FakeWandb:
            def __init__(self, project=None, **kw):
                self.project = project

        mock_callbacks_mod.MLflowCallback = _FakeMLflow
        mock_callbacks_mod.WandbCallback = _FakeWandb
        patches["aevyra_reflex.callbacks"] = mock_callbacks_mod

        for k, v in patches.items():
            old_modules[k] = sys.modules.get(k)
            sys.modules[k] = v

        try:
            from aevyra_reflex import cli
            importlib.reload(cli)

            # Defaults
            defaults = dict(
                dataset=MagicMock(exists=lambda: True, suffix=".jsonl", name="d.jsonl"),
                prompt=MagicMock(exists=lambda: True, read_text=lambda: "hi"),
                model=["openrouter/llama"],
                target=[],
                verdict_results=None,
                metric=[],
                judge=None,
                judge_criteria=None,
                strategy="auto",
                reasoning_model=None,
                reasoning_api_key=None,
                reasoning_base_url=None,
                source_model=None,
                max_iterations=3,
                threshold=0.85,
                output=None,
                results_json=None,
                max_workers=1,
                input_field=None,
                output_field=None,
                run_dir=None,
                resume=False,
                resume_from=None,
                train_split=0.65,
                batch_size=0,
                full_eval_steps=0,
                eval_runs=1,
                val_split=0.0,
                early_stopping_patience=0,
                mlflow=False,
                mlflow_experiment=None,
                mlflow_tracking_uri=None,
                wandb=False,
                wandb_project=None,
                verbose=False,
            )
            defaults.update(extra_args)

            # Patch internals so optimize() doesn't actually do anything
            cli.PromptOptimizer = lambda config=None: mock_optimizer
            cli._resolve_provider = lambda *a, **kw: {}
            cli.RunStore = MagicMock(return_value=MagicMock(
                find_incomplete_run=lambda **kw: None,
                get_run=lambda *a: None,
            ))
            cli.Dataset = MagicMock(
                from_jsonl=lambda *a, **kw: mock_ds,
                from_csv=lambda *a, **kw: mock_ds,
            )

            try:
                cli.optimize(**defaults)
            except (SystemExit, Exception):
                pass  # typer.Exit or other exits are fine

            return captured.get("callbacks", [])
        finally:
            for k, old in old_modules.items():
                if old is not None:
                    sys.modules[k] = old
                else:
                    sys.modules.pop(k, None)

    def test_no_flags_no_callbacks(self):
        cbs = self._run_optimize_dry({})
        assert cbs == []

    def test_mlflow_flag_adds_mlflow_callback(self):
        cbs = self._run_optimize_dry({"mlflow": True})
        assert len(cbs) == 1
        assert cbs[0].__class__.__name__ == "_FakeMLflow"

    def test_mlflow_experiment_name_passed(self):
        cbs = self._run_optimize_dry({"mlflow": True, "mlflow_experiment": "my-exp"})
        assert cbs[0].experiment_name == "my-exp"

    def test_mlflow_experiment_defaults_to_aevyra_reflex(self):
        cbs = self._run_optimize_dry({"mlflow": True})
        assert cbs[0].experiment_name == "aevyra-reflex"

    def test_mlflow_tracking_uri_passed(self):
        cbs = self._run_optimize_dry({
            "mlflow": True,
            "mlflow_tracking_uri": "http://localhost:5000",
        })
        assert cbs[0].tracking_uri == "http://localhost:5000"

    def test_wandb_flag_adds_wandb_callback(self):
        cbs = self._run_optimize_dry({"wandb": True})
        assert len(cbs) == 1
        assert cbs[0].__class__.__name__ == "_FakeWandb"

    def test_wandb_project_passed(self):
        cbs = self._run_optimize_dry({"wandb": True, "wandb_project": "my-proj"})
        assert cbs[0].project == "my-proj"

    def test_wandb_project_defaults_to_aevyra_reflex(self):
        cbs = self._run_optimize_dry({"wandb": True})
        assert cbs[0].project == "aevyra-reflex"

    def test_both_flags_adds_both_callbacks(self):
        cbs = self._run_optimize_dry({"mlflow": True, "wandb": True})
        assert len(cbs) == 2
        names = {cb.__class__.__name__ for cb in cbs}
        assert "_FakeMLflow" in names
        assert "_FakeWandb" in names
