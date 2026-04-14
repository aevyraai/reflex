"""
Security incident report → executive brief
==========================================
Demonstrates reflex optimising a vague starting prompt into one that reliably
produces 3-sentence executive briefs from security incident reports.

The baseline model gets a vague instruction ("Summarise this security incident
report.") and typically produces free-form prose.  The judge checks for the
strict 3-sentence format (what/impact/remediation) and scores accordingly.
Reflex iteratively diagnoses the format mismatch and rewrites the prompt until
the score meets the target threshold.

Usage:
    export ANTHROPIC_API_KEY=...
    python security_incidents_run.py
"""

from pathlib import Path

from aevyra_reflex import PromptOptimizer
from aevyra_verdict import Dataset, LLMJudge
from aevyra_verdict.providers import AnthropicProvider

HERE = Path(__file__).parent

# ── dataset ───────────────────────────────────────────────────────────────────

dataset = Dataset.from_jsonl(HERE / "security_incidents.jsonl")

# ── judge metric ──────────────────────────────────────────────────────────────

judge_criteria = (HERE / "security_incidents_judge.md").read_text()

# A stronger judge model evaluates format compliance (1–5):
# does the output contain exactly 3 sentences covering what/impact/remediation?
metric = LLMJudge(
    judge_provider=AnthropicProvider(model="claude-sonnet-4-6"),
    criteria=judge_criteria,
)

# ── run ───────────────────────────────────────────────────────────────────────

result = (
    PromptOptimizer()
    .set_dataset(dataset)
    .add_provider("anthropic", "claude-haiku-4-5-20251001")
    .add_metric(metric)
    .run(
        prompt=(HERE / "security_incidents_prompt.md").read_text(),
        max_iterations=6,
    )
)

print(result.summary())
print("\nBest prompt:\n")
print(result.best_prompt)
