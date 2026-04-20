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

"""Agent trace types for pipeline-level prompt optimization.

Replace the manual build_trace() pattern with structured types that carry
node-level inputs and outputs, and integrate directly with PromptOptimizer.

Usage::

    from aevyra_reflex import AgentTrace, TraceNode

    def run_pipeline(prompt: str, ticket: str) -> AgentTrace:
        ticket_type = classify_ticket(ticket, llm_fn)
        policy      = retrieve_policy(ticket_type)
        response    = generate_response(ticket, ticket_type, policy, prompt, llm_fn)

        return AgentTrace(
            nodes=[
                TraceNode("classify_ticket",    input=ticket,      output=ticket_type),
                TraceNode("retrieve_policy",    input=ticket_type, output=policy),
                TraceNode("generate_response",  input=ticket,      output=response, optimize=True),
            ],
            ideal=expected_response,
        )

    result = (
        PromptOptimizer()
        .set_pipeline(run_pipeline)
        .set_inputs(tickets)
        .add_metric(LLMJudge(judge_provider=..., criteria=judge_criteria))
        .run(starting_prompt)
    )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceNode:
    """One node in an agent pipeline execution trace.

    Args:
        name:     Human-readable node name (e.g. "classify_ticket").
        input:    The node's input value. Any serializable type.
        output:   The node's output value. Any serializable type.
        optimize: Mark this node as the optimization target. Exactly one node
                  per trace should be True. Defaults to False.
    """

    name: str
    input: Any
    output: Any
    optimize: bool = False


@dataclass
class AgentTrace:
    """Full execution trace of one pipeline run.

    Carries the inputs and outputs of every node, the expected ideal output,
    and optional metadata. Passed to PromptOptimizer via set_pipeline().

    The judge receives the full trace as structured text — classification
    decisions, retrieved context, and the final response — and evaluates the
    entire system against the provided criteria and ideal.

    Args:
        nodes:    Ordered list of TraceNode objects, one per pipeline stage.
        ideal:    The expected/reference output for this input. Optional but
                  recommended: used by the judge and displayed in failure reports.
        metadata: Arbitrary key/value metadata attached to the trace.
    """

    nodes: list[TraceNode]
    ideal: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def optimize_node(self) -> TraceNode | None:
        """The node marked optimize=True — whose prompt is being optimized.

        Falls back to the last node if none is explicitly marked.
        """
        for node in self.nodes:
            if node.optimize:
                return node
        return self.nodes[-1] if self.nodes else None

    def to_trace_text(self) -> str:
        """Format the full trace as structured text.

        This text is passed to the judge as the ``response`` to evaluate.
        The judge criteria should be written to score the full trace, not
        just the final response.

        Example output::

            === AGENT TRACE ===

            Node 1 — classify_ticket
              Input:  I was charged twice this month
              Output: billing

            Node 2 — retrieve_policy
              Input:  billing
              Output: - Refunds only within 30 days ...

            Node 3 — generate_response  [optimize]
              Input:  I was charged twice this month
              Output: Thank you for reaching out. For billing issues ...
        """
        lines = ["=== AGENT TRACE ==="]
        for i, node in enumerate(self.nodes, 1):
            marker = "  [optimize]" if node.optimize else ""
            lines.append(f"\nNode {i} — {node.name}{marker}")
            lines.append(f"  Input:  {_fmt(node.input)}")
            lines.append(f"  Output: {_fmt(node.output)}")
        return "\n".join(lines)

    def to_dataset_record(self) -> dict[str, Any]:
        """Convert to a Verdict Dataset-compatible record.

        Useful when you want to pre-build traces and use the traditional
        set_dataset() flow rather than set_pipeline().

        Example::

            records = [run_pipeline(starting_prompt, inp).to_dataset_record()
                       for inp in inputs]
            dataset = Dataset.from_list(records)
        """
        return {
            "messages": [{"role": "user", "content": self.to_trace_text()}],
            "ideal": self.ideal,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    """Format a node value as a concise string for trace display."""
    if isinstance(v, str):
        return v
    try:
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(v)
