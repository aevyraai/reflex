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

"""Tests for AgentTrace, TraceNode, and pipeline-mode PromptOptimizer integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aevyra_reflex import AgentTrace, TraceNode
from aevyra_reflex.optimizer import PromptOptimizer
from aevyra_reflex.result import OptimizationResult
from aevyra_reflex.trace import _fmt


# ---------------------------------------------------------------------------
# TraceNode
# ---------------------------------------------------------------------------

class TestTraceNode:
    def test_defaults(self):
        node = TraceNode(name="step", input="hello", output="world")
        assert node.name == "step"
        assert node.input == "hello"
        assert node.output == "world"
        assert node.optimize is False

    def test_optimize_flag(self):
        node = TraceNode(name="gen", input="q", output="a", optimize=True)
        assert node.optimize is True

    def test_accepts_non_string_values(self):
        node = TraceNode(name="classify", input={"text": "hello"}, output=["label_a"])
        assert node.input == {"text": "hello"}
        assert node.output == ["label_a"]


# ---------------------------------------------------------------------------
# AgentTrace.optimize_node
# ---------------------------------------------------------------------------

class TestAgentTraceOptimizeNode:
    def test_returns_marked_node(self):
        nodes = [
            TraceNode("a", "in_a", "out_a"),
            TraceNode("b", "in_b", "out_b", optimize=True),
            TraceNode("c", "in_c", "out_c"),
        ]
        trace = AgentTrace(nodes=nodes)
        assert trace.optimize_node is nodes[1]

    def test_falls_back_to_last_node_when_none_marked(self):
        nodes = [
            TraceNode("a", "in_a", "out_a"),
            TraceNode("b", "in_b", "out_b"),
        ]
        trace = AgentTrace(nodes=nodes)
        assert trace.optimize_node is nodes[-1]

    def test_returns_none_for_empty_nodes(self):
        trace = AgentTrace(nodes=[])
        assert trace.optimize_node is None


# ---------------------------------------------------------------------------
# AgentTrace.to_trace_text
# ---------------------------------------------------------------------------

class TestAgentTraceToTraceText:
    def test_header_present(self):
        trace = AgentTrace(nodes=[TraceNode("step", "input", "output")])
        text = trace.to_trace_text()
        assert "=== AGENT TRACE ===" in text

    def test_node_names_appear(self):
        nodes = [
            TraceNode("classify_ticket", "ticket text", "billing"),
            TraceNode("retrieve_policy", "billing", "30-day refund policy", optimize=True),
        ]
        trace = AgentTrace(nodes=nodes)
        text = trace.to_trace_text()
        assert "classify_ticket" in text
        assert "retrieve_policy" in text

    def test_optimize_marker_present(self):
        nodes = [
            TraceNode("classify", "in", "out"),
            TraceNode("generate", "in", "out", optimize=True),
        ]
        trace = AgentTrace(nodes=nodes)
        text = trace.to_trace_text()
        assert "[optimize]" in text

    def test_no_optimize_marker_for_unmarked_nodes(self):
        nodes = [TraceNode("classify", "in", "out")]
        trace = AgentTrace(nodes=nodes)
        text = trace.to_trace_text()
        assert "[optimize]" not in text

    def test_inputs_outputs_appear(self):
        nodes = [TraceNode("step", "my input", "my output")]
        text = AgentTrace(nodes=nodes).to_trace_text()
        assert "my input" in text
        assert "my output" in text

    def test_node_ordering(self):
        nodes = [
            TraceNode("first", "a", "b"),
            TraceNode("second", "c", "d"),
            TraceNode("third", "e", "f"),
        ]
        text = AgentTrace(nodes=nodes).to_trace_text()
        pos_first = text.index("first")
        pos_second = text.index("second")
        pos_third = text.index("third")
        assert pos_first < pos_second < pos_third

    def test_dict_values_serialized(self):
        node = TraceNode("step", input={"key": "val"}, output={"result": 42})
        text = AgentTrace(nodes=[node]).to_trace_text()
        assert '"key"' in text or "key" in text

    def test_empty_nodes_produces_header_only(self):
        text = AgentTrace(nodes=[]).to_trace_text()
        assert "=== AGENT TRACE ===" in text


# ---------------------------------------------------------------------------
# AgentTrace.to_dataset_record
# ---------------------------------------------------------------------------

class TestAgentTraceToDatasetRecord:
    def test_record_structure(self):
        trace = AgentTrace(
            nodes=[TraceNode("step", "in", "out")],
            ideal="expected output",
        )
        record = trace.to_dataset_record()
        assert "messages" in record
        assert record["ideal"] == "expected output"
        assert len(record["messages"]) == 1
        assert record["messages"][0]["role"] == "user"
        assert "=== AGENT TRACE ===" in record["messages"][0]["content"]

    def test_none_ideal_preserved(self):
        trace = AgentTrace(nodes=[TraceNode("s", "i", "o")])
        record = trace.to_dataset_record()
        assert record["ideal"] is None


# ---------------------------------------------------------------------------
# _fmt helper
# ---------------------------------------------------------------------------

class TestFmt:
    def test_string_passthrough(self):
        assert _fmt("hello") == "hello"

    def test_dict_json(self):
        result = _fmt({"a": 1})
        assert '"a"' in result
        assert "1" in result

    def test_list_json(self):
        result = _fmt([1, 2, 3])
        assert "1" in result and "3" in result

    def test_non_serializable_falls_back_to_str(self):
        class Unserializable:
            def __repr__(self):
                return "MyObj"
        result = _fmt(Unserializable())
        assert "MyObj" in result


# ---------------------------------------------------------------------------
# Public imports
# ---------------------------------------------------------------------------

class TestPublicExports:
    def test_agent_trace_importable_from_package(self):
        from aevyra_reflex import AgentTrace as AT
        assert AT is AgentTrace

    def test_trace_node_importable_from_package(self):
        from aevyra_reflex import TraceNode as TN
        assert TN is TraceNode


# ---------------------------------------------------------------------------
# PromptOptimizer pipeline mode validation
# ---------------------------------------------------------------------------

class TestPipelineModeValidation:
    def _make_optimizer(self):
        from aevyra_reflex.optimizer import OptimizerConfig
        return PromptOptimizer(OptimizerConfig(strategy="iterative", max_iterations=1))

    def test_pipeline_requires_inputs(self):
        opt = self._make_optimizer()
        opt.set_pipeline(lambda prompt, inp: None)
        opt.add_metric(MagicMock(name="judge"))
        with pytest.raises(ValueError, match="set_inputs"):
            opt.run("initial prompt")

    def test_pipeline_and_dataset_mutually_exclusive(self):
        from aevyra_verdict.dataset import Conversation, Dataset, Message
        opt = self._make_optimizer()
        opt.set_pipeline(lambda prompt, inp: None)
        opt.set_inputs(["a"])
        opt.add_metric(MagicMock(name="judge"))
        opt._dataset = Dataset(conversations=[
            Conversation(messages=[Message(role="user", content="hi")], ideal="yes")
        ])
        with pytest.raises(ValueError, match="mutually exclusive"):
            opt.run("initial prompt")

    def test_pipeline_rejects_empty_inputs(self):
        opt = self._make_optimizer()
        opt.set_pipeline(lambda prompt, inp: None)
        opt.set_inputs([])
        opt.add_metric(MagicMock(name="judge"))
        with pytest.raises(ValueError, match="empty"):
            opt.run("initial prompt")

    def test_pipeline_rejects_pdo_strategy(self):
        from aevyra_reflex.optimizer import OptimizerConfig
        opt = PromptOptimizer(OptimizerConfig(strategy="pdo", max_iterations=1))
        opt.set_pipeline(lambda prompt, inp: None)
        opt.set_inputs(["a"])
        opt.add_metric(MagicMock(name="judge"))
        with pytest.raises(ValueError, match="pdo"):
            opt.run("initial prompt")


# ---------------------------------------------------------------------------
# PromptOptimizer._build_synthetic_dataset
# ---------------------------------------------------------------------------

class TestBuildSyntheticDataset:
    def test_one_conversation_per_input(self):
        opt = PromptOptimizer()
        inputs = ["input1", "input2", "input3"]
        dataset = opt._build_synthetic_dataset(inputs)
        assert len(dataset.conversations) == 3

    def test_pipeline_input_stored_in_metadata(self):
        opt = PromptOptimizer()
        inputs = ["hello", {"key": "value"}]
        dataset = opt._build_synthetic_dataset(inputs)
        assert dataset.conversations[0].metadata["_pipeline_input"] == "hello"
        assert dataset.conversations[1].metadata["_pipeline_input"] == {"key": "value"}

    def test_no_ideals(self):
        opt = PromptOptimizer()
        dataset = opt._build_synthetic_dataset(["x"])
        assert dataset.conversations[0].ideal is None


# ---------------------------------------------------------------------------
# PromptOptimizer._run_pipeline_eval
# ---------------------------------------------------------------------------

class TestRunPipelineEval:
    def _make_trace(self, output: str, ideal: str = "expected") -> AgentTrace:
        return AgentTrace(
            nodes=[TraceNode("generate", "input", output, optimize=True)],
            ideal=ideal,
        )

    def _make_mock_metric(self, score_value: float) -> MagicMock:
        metric = MagicMock()
        result = MagicMock()
        result.score = score_value
        metric.score.return_value = result
        metric.judge_tokens_used = 0
        return metric

    def test_mean_score_computed_correctly(self):
        from aevyra_reflex.optimizer import OptimizerConfig

        inputs = ["input_a", "input_b"]
        traces = {
            "input_a": self._make_trace("response_a", ideal="ideal_a"),
            "input_b": self._make_trace("response_b", ideal="ideal_b"),
        }

        def pipeline_fn(prompt, inp):
            return traces[inp]

        opt = PromptOptimizer(OptimizerConfig(max_workers=1))
        opt.set_pipeline(pipeline_fn)
        opt.set_inputs(inputs)
        opt._metrics = [self._make_mock_metric(0.8)]

        dataset = opt._build_synthetic_dataset(inputs)
        score, failing, total_tokens = opt._run_pipeline_eval("my prompt", dataset, bottom_k=10)

        assert score == pytest.approx(0.8, abs=1e-6)
        assert len(failing) == 2

    def test_bottom_k_limits_failing_samples(self):
        from aevyra_reflex.optimizer import OptimizerConfig

        inputs = [f"input_{i}" for i in range(5)]

        def pipeline_fn(prompt, inp):
            return self._make_trace(f"out_{inp}", ideal="ideal")

        opt = PromptOptimizer(OptimizerConfig(max_workers=1))
        opt.set_pipeline(pipeline_fn)
        opt.set_inputs(inputs)
        opt._metrics = [self._make_mock_metric(0.5)]

        dataset = opt._build_synthetic_dataset(inputs)
        _, failing, _ = opt._run_pipeline_eval("prompt", dataset, bottom_k=2)
        assert len(failing) == 2

    def test_failing_sorted_ascending_by_score(self):
        from aevyra_reflex.optimizer import OptimizerConfig

        inputs = ["low", "high"]
        scores = {"low": 0.1, "high": 0.9}

        def pipeline_fn(prompt, inp):
            return self._make_trace(inp, ideal="ideal")

        def make_metric(inp_scores):
            metric = MagicMock()
            def score_fn(response, ideal, messages):
                r = MagicMock()
                # Return scores based on what's in response
                for key, val in inp_scores.items():
                    if key in response:
                        r.score = val
                        return r
                r.score = 0.5
                return r

            metric.score.side_effect = score_fn
            metric.judge_tokens_used = 0
            return metric

        opt = PromptOptimizer(OptimizerConfig(max_workers=1))
        opt.set_pipeline(pipeline_fn)
        opt.set_inputs(inputs)
        opt._metrics = [make_metric(scores)]

        dataset = opt._build_synthetic_dataset(inputs)
        _, failing, _ = opt._run_pipeline_eval("prompt", dataset, bottom_k=2)
        assert failing[0]["score"] <= failing[1]["score"]

    def test_metric_score_called_with_trace_text(self):
        from aevyra_reflex.optimizer import OptimizerConfig

        inputs = ["ticket"]
        trace = self._make_trace("my response", ideal="expected")

        opt = PromptOptimizer(OptimizerConfig(max_workers=1))
        opt.set_pipeline(lambda prompt, inp: trace)
        opt.set_inputs(inputs)
        metric = self._make_mock_metric(0.7)
        opt._metrics = [metric]

        dataset = opt._build_synthetic_dataset(inputs)
        opt._run_pipeline_eval("prompt", dataset)

        call_kwargs = metric.score.call_args
        assert "=== AGENT TRACE ===" in call_kwargs.kwargs.get("response", call_kwargs.args[0] if call_kwargs.args else "")


# ---------------------------------------------------------------------------
# PDO strategy guard
# ---------------------------------------------------------------------------

class TestPDOPipelineGuard:
    def test_pdo_raises_not_implemented_when_eval_fn_set(self):
        from aevyra_reflex.strategies.pdo import PDOStrategy

        strategy = PDOStrategy()
        with pytest.raises(NotImplementedError, match="pdo"):
            strategy.run(
                initial_prompt="prompt",
                dataset=MagicMock(),
                providers=[],
                metrics=[],
                agent=MagicMock(),
                config=MagicMock(extra_kwargs={}),
                eval_fn=lambda p, d, **kw: (0.5, [], 0),
            )


# ---------------------------------------------------------------------------
# Integration: strategy receives eval_fn in pipeline mode
# ---------------------------------------------------------------------------

class TestPipelineModeStrategyIntegration:
    """Verify that strategy.run() is called with eval_fn when pipeline mode is active."""

    def test_strategy_receives_eval_fn(self):
        from aevyra_reflex.optimizer import OptimizerConfig
        from aevyra_reflex.result import EvalSnapshot, SampleSnapshot

        inputs = ["ticket_a", "ticket_b"]

        def pipeline_fn(prompt: str, inp: str) -> AgentTrace:
            return AgentTrace(
                nodes=[TraceNode("generate", inp, f"response to {inp}", optimize=True)],
                ideal="ideal response",
            )

        received_eval_fn = []

        # Fake strategy that records whether eval_fn was passed, then returns quickly
        class _RecordingStrategy:
            def run(self, *, eval_fn=None, **kwargs):
                received_eval_fn.append(eval_fn)
                # Call eval_fn once to verify it works end-to-end
                if eval_fn is not None:
                    score, failing, tokens = eval_fn(kwargs["initial_prompt"], kwargs["dataset"])
                return OptimizationResult(
                    best_prompt=kwargs["initial_prompt"],
                    best_score=0.5,
                    iterations=[],
                    converged=False,
                )

        mock_metric = MagicMock()
        score_result = MagicMock()
        score_result.score = 0.75
        mock_metric.score.return_value = score_result
        mock_metric.judge_tokens_used = 0
        mock_metric.requires_ideal = False

        opt = PromptOptimizer(OptimizerConfig(
            strategy="iterative",
            max_iterations=1,
            train_ratio=1.0,
            val_ratio=0.0,
            eval_runs=1,
        ))
        opt.set_pipeline(pipeline_fn)
        opt.set_inputs(inputs)
        opt._metrics = [mock_metric]

        with (
            patch("aevyra_reflex.optimizer.get_strategy", return_value=lambda: _RecordingStrategy()),
            patch.object(opt, "_run_eval", return_value=EvalSnapshot(
                mean_score=0.5,
                scores_by_metric={},
                samples=[SampleSnapshot("i", "r", "d", 0.5), SampleSnapshot("i2", "r2", "d2", 0.5)],
                total_tokens=0,
            )),
            patch.object(opt, "_run_eval_single", return_value=EvalSnapshot(
                mean_score=0.5,
                scores_by_metric={},
                samples=[SampleSnapshot("i", "r", "d", 0.5), SampleSnapshot("i2", "r2", "d2", 0.5)],
                total_tokens=0,
            )),
            patch("aevyra_reflex.optimizer.LLM", return_value=MagicMock(tokens_used=0)),
        ):
            opt.run("initial prompt")

        assert len(received_eval_fn) == 1
        assert received_eval_fn[0] is not None, "eval_fn should have been passed to strategy"
