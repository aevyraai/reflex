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

"""Iterative strategy: diagnose failures → revise prompt → repeat.

The simplest optimization loop. Each iteration:
  1. Run the current prompt through verdict's EvalRunner
  2. Collect the worst-scoring samples
  3. Send them to Claude with the current prompt for diagnosis
  4. Claude proposes a revised prompt
  5. If the score meets the threshold, stop; otherwise repeat
"""

from __future__ import annotations

import logging
from typing import Any

from aevyra_reflex.agent import Agent
from aevyra_reflex.result import IterationRecord, OptimizationResult
from aevyra_reflex.strategies.base import Strategy

logger = logging.getLogger(__name__)


class IterativeStrategy(Strategy):
    """Diagnose-and-revise loop powered by Claude."""

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
        resume_state: dict | None = None,
        update_strategy_state: Any | None = None,
    ) -> OptimizationResult:
        from aevyra_verdict.runner import RunConfig

        # Restore cross-iteration state on resume
        rs = resume_state or {}
        iters_done: int = rs.get("iters_done", 0)
        current_prompt: str = rs.get("current_prompt", initial_prompt)
        previous_reasoning: str = rs.get("previous_reasoning", "")
        rewrite_log: list[dict] = list(rs.get("rewrite_log", []))
        score_trajectory: list[float] = list(rs.get("score_trajectory", []))

        iterations: list[IterationRecord] = []
        label_free = not dataset.has_ideals()

        run_config = RunConfig(
            temperature=config.eval_temperature,
            max_tokens=config.max_tokens,
        )

        _batch_size = getattr(config, "batch_size", 0)
        _batch_seed = getattr(config, "batch_seed", 42)
        _full_eval_steps = getattr(config, "full_eval_steps", 0)

        # Mid-iteration checkpoint: saves eval result so a crash during the
        # (cheap) agent call doesn't force re-running the (expensive) eval.
        _iter_state: dict = rs.get("iterative_iter_state", {})

        def _save_iter_state(state: dict, iters_done_override: int | None = None) -> None:
            if update_strategy_state:
                update_strategy_state({
                    "iters_done": iters_done_override if iters_done_override is not None else i,
                    "current_prompt": current_prompt,
                    "previous_reasoning": previous_reasoning,
                    "rewrite_log": rewrite_log,
                    "score_trajectory": score_trajectory,
                    "iterative_iter_state": state,
                })

        if iters_done > 0:
            logger.info(f"[iterative] Resuming from iteration {iters_done + 1} (prompt: {len(current_prompt)} chars)")

        for i in range(iters_done, config.max_iterations):
            tag = f"[iterative][iter {i + 1}/{config.max_iterations}]"
            logger.info(f"{tag} Starting")

            # Check for saved mid-iteration state for this specific iteration
            _saved = _iter_state if _iter_state.get("iter") == i else {}
            _saved_stage = _saved.get("stage", "")

            # Determine whether this iteration should use the full training set
            # (periodic full-eval checkpoint) or a mini-batch sample.
            is_full_eval = (
                _batch_size > 0
                and _full_eval_steps > 0
                and (i + 1) % _full_eval_steps == 0
            )
            effective_batch = 0 if is_full_eval else _batch_size

            # 1. Run eval with current prompt (or restore from mid-iter checkpoint)
            reasoning_before = getattr(agent, "tokens_used", 0)
            if _saved_stage == "eval_done":
                score = _saved["score"]
                eval_tokens = _saved["eval_tokens"]
                failing_samples = _saved["failing_samples"]
                logger.info(f"{tag} Resuming — reusing saved eval (score: {score:.4f})")
            else:
                eval_label = "full-eval checkpoint" if is_full_eval else "eval"
                logger.info(f"{tag} Running {eval_label}...")
                score, failing_samples, eval_tokens = _run_eval(
                    prompt=current_prompt,
                    dataset=dataset,
                    providers=providers,
                    metrics=metrics,
                    run_config=run_config,
                    bottom_k=config.extra_kwargs.get("bottom_k", 10),
                    batch_size=effective_batch,
                    iteration_seed=_batch_seed + i,
                )
                _save_iter_state({"iter": i, "stage": "eval_done",
                                  "score": score, "eval_tokens": eval_tokens,
                                  "failing_samples": failing_samples})

            # 2. Record this iteration
            record = IterationRecord(
                iteration=i + 1,
                system_prompt=current_prompt,
                score=score,
                reasoning=previous_reasoning,  # reasoning that produced this prompt
                eval_tokens=eval_tokens,
                is_full_eval=is_full_eval,
            )
            iterations.append(record)

            score_trajectory.append(score)
            logger.info(f"{tag} Score: {score:.4f}  target: {config.score_threshold:.4f}  failing samples: {len(failing_samples)}")

            # 3. Check convergence
            if score >= config.score_threshold:
                logger.info(f"{tag} Score threshold met — stopping.")
                return OptimizationResult(
                    best_prompt=current_prompt,
                    best_score=score,
                    iterations=iterations,
                    converged=True,
                )

            # 4. Ask reasoning model to diagnose and revise
            reasoning_before = getattr(agent, "tokens_used", 0)
            if i == 0 and not rewrite_log:
                logger.info(f"{tag} Diagnosing failures (first iteration)...")
                revised, reasoning, change_summary = agent.diagnose_and_revise(
                    system_prompt=current_prompt,
                    failing_samples=failing_samples,
                    label_free=label_free,
                )
            else:
                logger.info(f"{tag} Refining prompt (rewrite log: {len(rewrite_log)} entries)...")
                revised, reasoning, change_summary = agent.refine(
                    system_prompt=current_prompt,
                    iteration=i + 1,
                    score_trajectory=score_trajectory,
                    mean_score=score,
                    target_score=config.score_threshold,
                    failing_samples=failing_samples,
                    previous_reasoning=previous_reasoning,
                    rewrite_log=rewrite_log,
                    label_free=label_free,
                )
            record.reasoning_tokens = getattr(agent, "tokens_used", 0) - reasoning_before
            record.change_summary = change_summary

            # Update the causal rewrite log with this iteration's outcome
            prev_score = score_trajectory[-2] if len(score_trajectory) >= 2 else score
            delta = score - prev_score
            rewrite_log.append({
                "iteration": i + 1,
                "score": score,
                "delta": delta,
                "change_summary": change_summary,
            })

            # Update the record with the reasoning generated this iteration
            # (what Claude said after seeing this score — produces next prompt)
            record.reasoning = reasoning
            if on_iteration:
                on_iteration(record)

            previous_reasoning = reasoning
            current_prompt = revised
            logger.info(f"{tag} Revised prompt: {len(revised)} chars")
            if change_summary:
                logger.info(f"{tag} Change: {change_summary}")

            # Clear mid-iter checkpoint and persist final cross-iteration state.
            # current_prompt / previous_reasoning are already updated above.
            _iter_state = {}
            _save_iter_state({}, iters_done_override=i + 1)

        # Exhausted max_iterations
        best = max(iterations, key=lambda r: r.score)
        return OptimizationResult(
            best_prompt=best.system_prompt,
            best_score=best.score,
            iterations=iterations,
            converged=False,
        )


def _run_eval(
    *,
    prompt: str,
    dataset: Any,
    providers: list[dict[str, Any]],
    metrics: list[Any],
    run_config: Any,
    bottom_k: int = 10,
    batch_size: int = 0,
    iteration_seed: int = 0,
) -> tuple[float, list[dict[str, Any]], int]:
    """Run a verdict eval with the given system prompt and return (mean_score, failing_samples, total_tokens).

    Injects the system prompt into every conversation in the dataset, runs
    the eval, and extracts the bottom-k scoring samples for diagnosis.

    Args:
        batch_size: If > 0, randomly sample this many examples from the dataset
            before running the eval (mini-batch mode). 0 = use the full dataset.
        iteration_seed: Random seed for the mini-batch sample. Pass a different
            value each iteration to ensure each batch is distinct.
    """
    import random as _random

    from aevyra_verdict import Dataset, EvalRunner
    from aevyra_verdict.dataset import Conversation, Message

    # Apply mini-batch sampling if requested
    all_convos = list(dataset.conversations)
    if batch_size > 0 and batch_size < len(all_convos):
        rng = _random.Random(iteration_seed)
        indices = sorted(rng.sample(range(len(all_convos)), batch_size))
        all_convos = [all_convos[i] for i in indices]

    # Inject system prompt into each conversation
    injected_convos = []
    for convo in all_convos:
        messages = list(convo.messages)
        # Replace or prepend system message
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

    # Build runner
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

    # Aggregate scores across all models and metrics
    all_scores: list[tuple[float, int]] = []  # (score, sample_index)
    sample_details: dict[int, dict[str, Any]] = {}

    for model_label, model_result in results.model_results.items():
        for idx in range(model_result.num_samples):
            mean_sample_score = _mean_metric_score(model_result.scores[idx])
            all_scores.append((mean_sample_score, idx))

            if idx not in sample_details or mean_sample_score < sample_details[idx].get("score", 1.0):
                convo = injected_dataset.conversations[idx]
                completion = model_result.completions[idx]
                sample_details[idx] = {
                    "input": convo.last_user_message or "(no user message)",
                    "response": completion.text if completion else "(no response)",
                    "ideal": convo.ideal,
                    "score": mean_sample_score,
                }

    if not all_scores:
        return 0.0, [], 0

    mean_score = sum(s for s, _ in all_scores) / len(all_scores)
    total_tokens = sum(mr.total_tokens() for mr in results.model_results.values())

    # Get the bottom-k failing samples
    sorted_details = sorted(sample_details.values(), key=lambda d: d["score"])
    failing = sorted_details[:bottom_k]

    return mean_score, failing, total_tokens


def _mean_metric_score(scores: dict[str, Any]) -> float:
    """Compute mean score from a dict of metric_name → ScoreResult."""
    values = []
    for score_result in scores.values():
        if hasattr(score_result, "score"):
            values.append(score_result.score)
        elif isinstance(score_result, (int, float)):
            values.append(float(score_result))
    return sum(values) / len(values) if values else 0.0
