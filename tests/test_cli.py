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
