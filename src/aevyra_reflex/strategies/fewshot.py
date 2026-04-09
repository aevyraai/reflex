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

"""Few-shot example optimization strategy.

Inspired by DSPy's approach of optimizing which examples to include in the
prompt. Each iteration:
  1. Run the current prompt (with current examples) through verdict
  2. Identify the best-scoring samples as candidate exemplars
  3. Ask Claude to select and arrange the most informative examples
  4. Build a new prompt with the instruction + curated few-shot examples
  5. If the score meets the threshold, stop; otherwise repeat with new examples
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from aevyra_reflex.agent import Agent
from aevyra_reflex.result import IterationRecord, OptimizationResult
from aevyra_reflex.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class _FewShotConfig:
    """Extracted config for the few-shot strategy."""

    max_examples: int = 5
    candidate_pool_size: int = 20
    bootstrap_rounds: int = 3
    selection_strategy: str = "diverse"  # "diverse", "top_k", "agent_curated"
    bottom_k: int = 10


class FewShotStrategy(Strategy):
    """Optimize by selecting the best few-shot examples to include in the prompt.

    The core idea: the *instruction* stays relatively stable, but which
    examples you show the model — and how you format them — makes a huge
    difference. This strategy:
      1. Bootstraps a pool of candidate exemplars from high-scoring samples
      2. Uses Claude to select the most diverse and informative subset
      3. Iteratively refines both the example selection and how they're formatted
    """

    def run(
        self,
        *,
        initial_prompt: str,
        dataset: Any,
        providers: list[dict[str, Any]],
        metrics: list[Any],
        agent: Agent,
        config: Any,
        on_iteration: Any | None = None,
    ) -> OptimizationResult:
        if not dataset.has_ideals():
            raise ValueError(
                "The fewshot strategy requires labeled examples (ideal answers) to build "
                "demonstrations. Use the iterative or pdo strategy for label-free datasets."
            )

        from aevyra_reflex.strategies.iterative import _run_eval

        from aevyra_verdict.runner import RunConfig

        extra = config.extra_kwargs or {}
        fs_config = _FewShotConfig(
            max_examples=extra.get("max_examples", 5),
            candidate_pool_size=extra.get("candidate_pool_size", 20),
            bootstrap_rounds=extra.get("bootstrap_rounds", 3),
            selection_strategy=extra.get("selection_strategy", "diverse"),
            bottom_k=extra.get("bottom_k", 10),
        )

        run_config = RunConfig(
            temperature=config.eval_temperature,
            max_tokens=config.max_tokens,
        )

        # Phase 1: Bootstrap — run the bare instruction to find good exemplars
        logger.info("Phase 1: Bootstrapping exemplar candidates...")
        candidate_exemplars = _bootstrap_exemplars(
            prompt=initial_prompt,
            dataset=dataset,
            providers=providers,
            metrics=metrics,
            run_config=run_config,
            pool_size=fs_config.candidate_pool_size,
        )
        logger.info(f"  Collected {len(candidate_exemplars)} candidate exemplars")

        # Phase 2: Iterative example selection and formatting
        current_instruction = initial_prompt
        current_examples: list[dict[str, str]] = []
        failing_samples: list = []
        iterations: list[IterationRecord] = []

        _batch_size = getattr(config, "batch_size", 0)
        _batch_seed = getattr(config, "batch_seed", 42)
        _full_eval_steps = getattr(config, "full_eval_steps", 0)

        for i in range(config.max_iterations):
            logger.info(f"Iteration {i + 1}/{config.max_iterations}")

            # Ask Claude to select/refine examples
            if i == 0:
                selected, instruction = agent.select_fewshot_examples(
                    base_instruction=initial_prompt,
                    candidate_exemplars=candidate_exemplars,
                    max_examples=fs_config.max_examples,
                    selection_strategy=fs_config.selection_strategy,
                )
            else:
                selected, instruction = agent.refine_fewshot(
                    current_instruction=current_instruction,
                    current_examples=current_examples,
                    candidate_exemplars=candidate_exemplars,
                    failing_samples=failing_samples,
                    max_examples=fs_config.max_examples,
                    score_trajectory=[r.score for r in iterations],
                )

            current_examples = selected
            current_instruction = instruction

            # Build the composite prompt
            composite_prompt = _build_fewshot_prompt(instruction, selected)

            # Determine full-eval checkpoint or mini-batch for this iteration
            is_full_eval = (
                _batch_size > 0
                and _full_eval_steps > 0
                and (i + 1) % _full_eval_steps == 0
            )
            effective_batch = 0 if is_full_eval else _batch_size

            # Evaluate
            score, failing_samples, eval_tokens = _run_eval(
                prompt=composite_prompt,
                dataset=dataset,
                providers=providers,
                metrics=metrics,
                run_config=run_config,
                bottom_k=fs_config.bottom_k,
                batch_size=effective_batch,
                iteration_seed=_batch_seed + i,
            )

            record = IterationRecord(
                iteration=i + 1,
                system_prompt=composite_prompt,
                score=score,
                reasoning=f"Selected {len(selected)} examples",
                eval_tokens=eval_tokens,
                is_full_eval=is_full_eval,
            )
            iterations.append(record)
            if on_iteration:
                on_iteration(record)

            logger.info(f"  Score: {score:.4f} (target: {config.score_threshold:.4f})")
            logger.info(f"  Examples in prompt: {len(selected)}")

            if score >= config.score_threshold:
                logger.info("Score threshold met — stopping.")
                return OptimizationResult(
                    best_prompt=composite_prompt,
                    best_score=score,
                    iterations=iterations,
                    converged=True,
                )

            # Re-bootstrap periodically to discover new exemplars
            if (i + 1) % fs_config.bootstrap_rounds == 0 and i + 1 < config.max_iterations:
                logger.info("  Re-bootstrapping exemplar pool...")
                new_candidates = _bootstrap_exemplars(
                    prompt=composite_prompt,
                    dataset=dataset,
                    providers=providers,
                    metrics=metrics,
                    run_config=run_config,
                    pool_size=fs_config.candidate_pool_size,
                )
                # Merge pools, dedup by input
                existing_inputs = {e["input"] for e in candidate_exemplars}
                for c in new_candidates:
                    if c["input"] not in existing_inputs:
                        candidate_exemplars.append(c)
                        existing_inputs.add(c["input"])

        best = max(iterations, key=lambda r: r.score)
        return OptimizationResult(
            best_prompt=best.system_prompt,
            best_score=best.score,
            iterations=iterations,
            converged=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bootstrap_exemplars(
    *,
    prompt: str,
    dataset: Any,
    providers: list[dict[str, Any]],
    metrics: list[Any],
    run_config: Any,
    pool_size: int,
) -> list[dict[str, str]]:
    """Run eval and extract top-scoring samples as candidate exemplars.

    Returns a list of dicts: {"input": ..., "output": ..., "score": ...}
    """
    from aevyra_verdict import Dataset, EvalRunner
    from aevyra_verdict.dataset import Conversation, Message

    # Inject system prompt
    injected_convos = []
    for convo in dataset.conversations:
        messages = list(convo.messages)
        if messages and messages[0].role == "system":
            messages[0] = Message(role="system", content=prompt)
        else:
            messages.insert(0, Message(role="system", content=prompt))
        injected_convos.append(Conversation(
            messages=messages,
            ideal=convo.ideal,
            metadata=convo.metadata,
        ))

    injected_dataset = Dataset(conversations=injected_convos)

    runner = EvalRunner(config=run_config)
    for p in providers:
        runner.add_provider(
            p["provider_name"],
            p["model"],
            label=p.get("label"),
            api_key=p.get("api_key"),
            base_url=p.get("base_url"),
        )
    for m in metrics:
        runner.add_metric(m)

    results = runner.run(injected_dataset, show_progress=True)

    # Collect samples with their scores
    scored_samples: list[dict[str, Any]] = []
    for _model_label, model_result in results.model_results.items():
        for idx in range(model_result.num_samples):
            score = _mean_score(model_result.scores[idx])
            convo = dataset.conversations[idx]
            user_input = convo.last_user_message or ""
            completion = model_result.completions[idx]
            # Use the ideal answer as the exemplar output (more reliable than model output)
            output = convo.ideal if convo.ideal else (
                completion.text if completion else ""
            )
            if user_input and output:
                scored_samples.append({
                    "input": user_input,
                    "output": output,
                    "score": score,
                })

    # Sort by score descending — top scorers make the best exemplars
    scored_samples.sort(key=lambda s: s["score"], reverse=True)

    # Deduplicate by input
    seen_inputs: set[str] = set()
    unique: list[dict[str, str]] = []
    for s in scored_samples:
        if s["input"] not in seen_inputs:
            seen_inputs.add(s["input"])
            unique.append(s)
        if len(unique) >= pool_size:
            break

    return unique


def _build_fewshot_prompt(instruction: str, examples: list[dict[str, str]]) -> str:
    """Assemble an instruction + few-shot examples into a single system prompt."""
    parts = [instruction.strip()]

    if examples:
        parts.append("")
        parts.append("## Examples")
        parts.append("")
        for i, ex in enumerate(examples, 1):
            parts.append(f"### Example {i}")
            parts.append(f"**Input:** {ex['input']}")
            parts.append(f"**Output:** {ex['output']}")
            parts.append("")

    return "\n".join(parts)


def _mean_score(scores: dict[str, Any]) -> float:
    """Compute mean score from a dict of metric_name → ScoreResult."""
    values = []
    for score_result in scores.values():
        if hasattr(score_result, "score"):
            values.append(score_result.score)
        elif isinstance(score_result, (int, float)):
            values.append(float(score_result))
    return sum(values) / len(values) if values else 0.0
