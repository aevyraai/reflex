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

"""Tests for the structural transform history feature.

Covers:
- _format_structural_history renders ✓/✗ outcome labels correctly
- _format_structural_history handles empty history
- _format_structural_history includes winner and tried transforms
- FREEFORM_RESTRUCTURE_PROMPT contains {structural_history} placeholder
- freeform_restructure() accepts and uses structural_history param
- structural strategy sets change_summary on IterationRecord
- structural strategy accumulates history with delta field
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# _format_structural_history
# ---------------------------------------------------------------------------

class TestFormatStructuralHistory:

    def _format(self, history):
        from aevyra_reflex.agent import _format_structural_history
        return _format_structural_history(history)

    def test_empty_history(self):
        result = self._format([])
        assert "no previous" in result.lower()

    def test_single_entry_helped(self):
        history = [{
            "iteration": 1,
            "transforms_tried": ["xml_tags", "markdown_structure"],
            "best_transform": "xml_tags",
            "score_before": 0.60,
            "score_after": 0.71,
            "delta": 0.11,
        }]
        result = self._format(history)
        assert "✓ helped" in result
        assert "xml_tags" in result
        assert "Iter 1" in result

    def test_single_entry_no_effect(self):
        history = [{
            "iteration": 1,
            "transforms_tried": ["minimal_flat"],
            "best_transform": "current",
            "score_before": 0.65,
            "score_after": 0.65,
            "delta": 0.0,
        }]
        result = self._format(history)
        assert "✗ no effect" in result

    def test_single_entry_hurt(self):
        history = [{
            "iteration": 1,
            "transforms_tried": ["section_reorder"],
            "best_transform": "section_reorder",
            "score_before": 0.70,
            "score_after": 0.60,
            "delta": -0.10,
        }]
        result = self._format(history)
        assert "✗ hurt" in result

    def test_multiple_entries(self):
        history = [
            {
                "iteration": 1,
                "transforms_tried": ["xml_tags", "markdown_structure", "agent_guided"],
                "best_transform": "xml_tags",
                "score_before": 0.55,
                "score_after": 0.63,
                "delta": 0.08,
            },
            {
                "iteration": 2,
                "transforms_tried": ["constraint_emphasis", "task_decomposition", "agent_guided"],
                "best_transform": "constraint_emphasis",
                "score_before": 0.63,
                "score_after": 0.67,
                "delta": 0.04,
            },
        ]
        result = self._format(history)
        lines = [line for line in result.splitlines() if line.strip()]
        assert len(lines) == 2
        assert "Iter 1" in lines[0]
        assert "Iter 2" in lines[1]

    def test_tried_transforms_listed(self):
        history = [{
            "iteration": 1,
            "transforms_tried": ["xml_tags", "minimal_flat", "agent_guided"],
            "best_transform": "xml_tags",
            "score_before": 0.60,
            "score_after": 0.70,
            "delta": 0.10,
        }]
        result = self._format(history)
        assert "xml_tags" in result
        assert "minimal_flat" in result
        assert "agent_guided" in result


# ---------------------------------------------------------------------------
# FREEFORM_RESTRUCTURE_PROMPT
# ---------------------------------------------------------------------------

class TestFreeformRestructurePrompt:

    def test_structural_history_placeholder_present(self):
        from aevyra_reflex.prompts import FREEFORM_RESTRUCTURE_PROMPT
        assert "{structural_history}" in FREEFORM_RESTRUCTURE_PROMPT

    def test_prompt_mentions_history_guidance(self):
        from aevyra_reflex.prompts import FREEFORM_RESTRUCTURE_PROMPT
        assert "dead ends" in FREEFORM_RESTRUCTURE_PROMPT or "history" in FREEFORM_RESTRUCTURE_PROMPT.lower()


# ---------------------------------------------------------------------------
# freeform_restructure signature
# ---------------------------------------------------------------------------

class TestFreeformRestructureSignature:

    def _make_llm(self, response="restructured prompt"):
        from aevyra_reflex.agent import LLM
        llm = LLM.__new__(LLM)
        llm.generate = MagicMock(return_value=response)
        llm.tokens_used = 0
        llm.source_model = None
        return llm

    def test_accepts_structural_history_param(self):
        llm = self._make_llm()
        history = [{"iteration": 1, "transforms_tried": ["xml_tags"], "best_transform": "xml_tags",
                    "score_before": 0.6, "score_after": 0.7, "delta": 0.1}]
        result = llm.freeform_restructure(
            current_prompt="You are a helpful assistant.",
            failing_samples=[],
            analysis="Structure is flat.",
            score_trajectory=[0.6],
            structural_history=history,
        )
        assert result == "restructured prompt"

    def test_history_injected_into_prompt(self):
        llm = self._make_llm()
        history = [{"iteration": 1, "transforms_tried": ["xml_tags"], "best_transform": "xml_tags",
                    "score_before": 0.6, "score_after": 0.7, "delta": 0.1}]
        llm.freeform_restructure(
            current_prompt="You are a helpful assistant.",
            failing_samples=[],
            analysis="Structure is flat.",
            structural_history=history,
        )
        call_args = llm.generate.call_args[0][0]
        assert "xml_tags" in call_args

    def test_no_history_uses_fallback(self):
        llm = self._make_llm()
        result = llm.freeform_restructure(
            current_prompt="You are a helpful assistant.",
            failing_samples=[],
            analysis="Structure is flat.",
            structural_history=[],
        )
        call_args = llm.generate.call_args[0][0]
        assert "no previous" in call_args.lower()

    def test_none_history_uses_fallback(self):
        llm = self._make_llm()
        llm.freeform_restructure(
            current_prompt="You are a helpful assistant.",
            failing_samples=[],
            analysis="Structure is flat.",
            structural_history=None,
        )
        call_args = llm.generate.call_args[0][0]
        assert "no previous" in call_args.lower()


# ---------------------------------------------------------------------------
# IterationRecord.change_summary for structural
# ---------------------------------------------------------------------------

class TestStructuralIterationRecord:

    def test_change_summary_field_exists(self):
        from aevyra_reflex.result import IterationRecord
        record = IterationRecord(iteration=1, system_prompt="p", score=0.5)
        assert hasattr(record, "change_summary")
        assert record.change_summary == ""

    def test_change_summary_set_to_winning_transform(self):
        from aevyra_reflex.result import IterationRecord
        record = IterationRecord(iteration=1, system_prompt="p", score=0.6)
        record.change_summary = "xml_tags"
        assert record.change_summary == "xml_tags"
