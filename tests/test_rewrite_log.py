"""Tests for the causal rewrite log feature.

Covers:
- _extract_change_summary parses the ## Summary of change section
- _extract_change_summary falls back gracefully when section is absent
- _format_rewrite_log renders the history correctly
- _format_rewrite_log handles empty log
- agent.diagnose_and_revise returns 3-tuple including change_summary
- agent.refine returns 3-tuple and accepts rewrite_log
- agent.refine passes rewrite_log into the prompt
- IterationRecord.change_summary field exists
- iterative strategy populates change_summary on each record
- iterative strategy accumulates rewrite_log and passes it to refine()
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# _extract_change_summary
# ---------------------------------------------------------------------------

class TestExtractChangeSummary:

    def _extract(self, text):
        from aevyra_reflex.agent import _extract_change_summary
        return _extract_change_summary(text)

    def test_parses_summary_section(self):
        response = (
            "## Failure patterns\n1. wrong format\n\n"
            "## Summary of change\nAdded numbered steps to fix structure failures\n\n"
            "## Revised prompt\nYou are a helpful assistant."
        )
        assert self._extract(response) == "Added numbered steps to fix structure failures"

    def test_strips_brackets(self):
        response = (
            "## Summary of change\n"
            "[Added numbered steps to fix structure failures]\n"
            "## Revised prompt\nfoo"
        )
        assert self._extract(response) == "Added numbered steps to fix structure failures"

    def test_returns_empty_when_absent(self):
        response = "## Failure patterns\n1. foo\n\n## Revised prompt\nbar"
        assert self._extract(response) == ""

    def test_case_insensitive_header(self):
        response = "## summary of change\nReplaced XML tags with Markdown headers\n## Revised prompt\nfoo"
        assert self._extract(response) == "Replaced XML tags with Markdown headers"

    def test_skips_blank_lines_after_header(self):
        response = "## Summary of change\n\n\nAdded a persona\n## Revised prompt\nfoo"
        assert self._extract(response) == "Added a persona"

    def test_does_not_bleed_into_next_section(self):
        response = (
            "## Summary of change\nAdded examples\n"
            "## Revised prompt\nYou are a helpful assistant."
        )
        result = self._extract(response)
        assert "Revised prompt" not in result
        assert result == "Added examples"


# ---------------------------------------------------------------------------
# _format_rewrite_log
# ---------------------------------------------------------------------------

class TestFormatRewriteLog:

    def _format(self, log):
        from aevyra_reflex.agent import _format_rewrite_log
        return _format_rewrite_log(log)

    def test_empty_log(self):
        result = self._format([])
        assert "no previous rewrites" in result

    def test_single_entry_positive_delta(self):
        log = [{"iteration": 1, "score": 0.71, "delta": 0.09, "change_summary": "Added steps"}]
        result = self._format(log)
        assert "Iter 1" in result
        assert "0.7100" in result
        assert "+0.0900" in result
        assert "helped" in result
        assert "Added steps" in result

    def test_single_entry_negligible_delta(self):
        log = [{"iteration": 2, "score": 0.71, "delta": 0.001, "change_summary": "Added persona"}]
        result = self._format(log)
        assert "no effect" in result

    def test_single_entry_negative_delta(self):
        log = [{"iteration": 3, "score": 0.65, "delta": -0.06, "change_summary": "Removed examples"}]
        result = self._format(log)
        assert "hurt" in result

    def test_multiple_entries_ordered(self):
        log = [
            {"iteration": 1, "score": 0.62, "delta": 0.0, "change_summary": "Initial"},
            {"iteration": 2, "score": 0.71, "delta": 0.09, "change_summary": "Added steps"},
            {"iteration": 3, "score": 0.72, "delta": 0.01, "change_summary": "Tweaked wording"},
        ]
        result = self._format(log)
        lines = [l for l in result.splitlines() if l.strip()]
        assert len(lines) == 3
        assert "Iter 1" in lines[0]
        assert "Iter 2" in lines[1]
        assert "Iter 3" in lines[2]

    def test_missing_summary_shows_placeholder(self):
        log = [{"iteration": 1, "score": 0.5, "delta": 0.0, "change_summary": ""}]
        result = self._format(log)
        assert "no summary" in result


# ---------------------------------------------------------------------------
# LLM.diagnose_and_revise return signature
# ---------------------------------------------------------------------------

class TestDiagnoseAndReviseSignature:

    def _make_llm(self, mock_response):
        with patch("aevyra_reflex.agent._resolve_agent_backend") as mock_resolve:
            backend = MagicMock()
            backend.tokens_used = 0
            backend.generate.return_value = mock_response
            mock_resolve.return_value = backend
            from aevyra_reflex.agent import LLM
            return LLM(model="mock", provider="openai")

    def test_returns_three_tuple(self):
        response = (
            "## Failure patterns\n1. foo\n\n"
            "## Summary of change\nAdded explicit output format\n\n"
            "## Revised prompt\nBetter prompt here."
        )
        llm = self._make_llm(response)
        result = llm.diagnose_and_revise("old prompt", [])
        assert len(result) == 3

    def test_revised_prompt_extracted(self):
        response = (
            "## Failure patterns\n1. foo\n\n"
            "## Summary of change\nAdded format\n\n"
            "## Revised prompt\nNew prompt."
        )
        llm = self._make_llm(response)
        revised, reasoning, summary = llm.diagnose_and_revise("old", [])
        assert revised == "New prompt."

    def test_change_summary_extracted(self):
        response = (
            "## Failure patterns\n1. foo\n\n"
            "## Summary of change\nAdded numbered steps\n\n"
            "## Revised prompt\nNew prompt."
        )
        llm = self._make_llm(response)
        _, _, summary = llm.diagnose_and_revise("old", [])
        assert summary == "Added numbered steps"

    def test_empty_summary_when_not_present(self):
        response = "## Failure patterns\n1. foo\n\n## Revised prompt\nNew."
        llm = self._make_llm(response)
        _, _, summary = llm.diagnose_and_revise("old", [])
        assert summary == ""


# ---------------------------------------------------------------------------
# LLM.refine return signature and rewrite_log injection
# ---------------------------------------------------------------------------

class TestRefineSignature:

    def _make_llm_capturing_prompt(self):
        """Returns (llm, captured) where captured['prompt'] is set on generate()."""
        captured = {}
        with patch("aevyra_reflex.agent._resolve_agent_backend") as mock_resolve:
            backend = MagicMock()
            backend.tokens_used = 0

            def fake_generate(prompt, *, temperature=1.0):
                captured["prompt"] = prompt
                return (
                    "## Summary of change\nReplaced XML tags with Markdown\n\n"
                    "## Revised prompt\nImproved prompt."
                )

            backend.generate.side_effect = fake_generate
            mock_resolve.return_value = backend
            from aevyra_reflex.agent import LLM
            llm = LLM(model="mock", provider="openai")
        return llm, captured

    def test_returns_three_tuple(self):
        llm, _ = self._make_llm_capturing_prompt()
        result = llm.refine(
            system_prompt="old",
            iteration=2,
            score_trajectory=[0.5, 0.6],
            mean_score=0.6,
            target_score=0.85,
            failing_samples=[],
            previous_reasoning="prev",
        )
        assert len(result) == 3

    def test_rewrite_log_appears_in_prompt(self):
        llm, captured = self._make_llm_capturing_prompt()
        log = [{"iteration": 1, "score": 0.5, "delta": 0.0, "change_summary": "Added XML tags"}]
        llm.refine(
            system_prompt="old",
            iteration=2,
            score_trajectory=[0.5, 0.6],
            mean_score=0.6,
            target_score=0.85,
            failing_samples=[],
            previous_reasoning="prev",
            rewrite_log=log,
        )
        assert "Added XML tags" in captured["prompt"]
        assert "Rewrite history" in captured["prompt"]

    def test_empty_rewrite_log_shows_placeholder(self):
        llm, captured = self._make_llm_capturing_prompt()
        llm.refine(
            system_prompt="old",
            iteration=2,
            score_trajectory=[0.5, 0.6],
            mean_score=0.6,
            target_score=0.85,
            failing_samples=[],
            previous_reasoning="prev",
            rewrite_log=[],
        )
        assert "no previous rewrites" in captured["prompt"]

    def test_none_rewrite_log_treated_as_empty(self):
        llm, captured = self._make_llm_capturing_prompt()
        llm.refine(
            system_prompt="old",
            iteration=2,
            score_trajectory=[0.5, 0.6],
            mean_score=0.6,
            target_score=0.85,
            failing_samples=[],
            previous_reasoning="prev",
            rewrite_log=None,
        )
        assert "no previous rewrites" in captured["prompt"]


# ---------------------------------------------------------------------------
# IterationRecord.change_summary field
# ---------------------------------------------------------------------------

class TestIterationRecordChangeSummary:

    def test_field_exists_with_default(self):
        from aevyra_reflex.result import IterationRecord
        record = IterationRecord(iteration=1, system_prompt="p", score=0.5)
        assert hasattr(record, "change_summary")
        assert record.change_summary == ""

    def test_field_can_be_set(self):
        from aevyra_reflex.result import IterationRecord
        record = IterationRecord(
            iteration=1, system_prompt="p", score=0.5,
            change_summary="Added numbered steps",
        )
        assert record.change_summary == "Added numbered steps"
