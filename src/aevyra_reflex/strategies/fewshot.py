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
        resume_state: dict | None = None,
        update_strategy_state: Any | None = None,
        eval_fn: Any | None = None,
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
            max_workers=config.max_workers,
        )

        # Restore state on resume
        rs = resume_state or {}
        iters_done: int = rs.get("iters_done", 0)
        current_instruction: str = rs.get("current_instruction", initial_prompt)
        current_examples: list[dict[str, str]] = rs.get("current_examples", [])

        # Phase 1: Bootstrap — skip if we already have enough candidates from a
        # previous run (including partial bootstrap that was interrupted mid-way).
        _saved_candidates: list = rs.get("candidate_exemplars", [])
        _bootstrap_done: bool = rs.get("bootstrap_done", False)
        # Bootstrap eval tokens are persisted in state so they survive a crash
        # between bootstrap completion and iteration 1 finishing.
        _bootstrap_eval_tokens: int = rs.get("bootstrap_eval_tokens", 0)
        if _bootstrap_done and _saved_candidates:
            candidate_exemplars = _saved_candidates
            logger.info(f"Phase 1: Resuming with {len(candidate_exemplars)} saved exemplar candidates (skipping bootstrap)")
        else:
            # Evaluate samples one at a time so partial progress can survive an
            # interruption — each scored sample is checkpointed immediately.
            already_scored = len(_saved_candidates)
            if already_scored:
                logger.info(f"Phase 1: Resuming bootstrap ({already_scored}/{len(dataset.conversations)} already scored)...")
            else:
                logger.info("Phase 1: Bootstrapping exemplar candidates...")
            new_tokens: int  # tokens from samples scored in this session only
            candidate_exemplars, new_tokens = _bootstrap_exemplars(
                prompt=initial_prompt,
                dataset=dataset,
                providers=providers,
                metrics=metrics,
                run_config=run_config,
                pool_size=fs_config.candidate_pool_size,
                partial_candidates=_saved_candidates,
                on_sample=lambda candidates: update_strategy_state({
                    "candidate_exemplars": candidates,
                    "bootstrap_done": False,
                    "bootstrap_eval_tokens": _bootstrap_eval_tokens,
                    "iters_done": 0,
                    "current_instruction": current_instruction,
                    "current_examples": current_examples,
                }) if update_strategy_state else None,
            )
            _bootstrap_eval_tokens += new_tokens
            _fmt_bt = f"{_bootstrap_eval_tokens / 1000:.1f}K" if _bootstrap_eval_tokens >= 1000 else str(_bootstrap_eval_tokens)
            logger.info(f"  Collected {len(candidate_exemplars)} candidate exemplars (bootstrap eval tokens: {_fmt_bt})")
            if update_strategy_state:
                update_strategy_state({
                    "candidate_exemplars": candidate_exemplars,
                    "bootstrap_done": True,
                    "bootstrap_eval_tokens": _bootstrap_eval_tokens,
                    "iters_done": 0,
                    "current_instruction": current_instruction,
                    "current_examples": current_examples,
                })

        # Phase 2: Iterative example selection and formatting
        failing_samples: list = []
        iterations: list[IterationRecord] = []

        _batch_size = getattr(config, "batch_size", 0)
        _batch_seed = getattr(config, "batch_seed", 42)
        _full_eval_steps = getattr(config, "full_eval_steps", 0)

        # Mid-iteration checkpoint: saves agent selection + eval result so a
        # crash during the (cheap) refine call doesn't force re-running the
        # (expensive) eval.
        _iter_state: dict = rs.get("fewshot_iter_state", {})

        def _save_iter_state(state: dict) -> None:
            if update_strategy_state:
                update_strategy_state({
                    "candidate_exemplars": candidate_exemplars,
                    "bootstrap_done": True,
                    "bootstrap_eval_tokens": _bootstrap_eval_tokens,
                    # mid-iter: save i so resume restarts this iteration
                    # (not i+1, which would skip it)
                    "iters_done": i,
                    "current_instruction": current_instruction,
                    "current_examples": current_examples,
                    "fewshot_iter_state": state,
                })

        for i in range(iters_done, config.max_iterations):
            logger.info(f"Iteration {i + 1}/{config.max_iterations}")

            # Check for saved mid-iteration state for this specific iteration
            _saved = _iter_state if _iter_state.get("iter") == i else {}
            _saved_stage = _saved.get("stage", "")

            # Determine full-eval checkpoint or mini-batch for this iteration
            is_full_eval = (
                _batch_size > 0
                and _full_eval_steps > 0
                and (i + 1) % _full_eval_steps == 0
            )
            effective_batch = 0 if is_full_eval else _batch_size

            # Ask Claude to select/refine examples (or restore from checkpoint)
            reasoning_before = getattr(agent, "tokens_used", 0)
            # Reasoning tokens from a previous session (saved in mid-iter checkpoint)
            # are carried forward so a resume doesn't zero out the spent tokens.
            _saved_reasoning_tokens: int = _saved.get("reasoning_tokens_so_far", 0)
            if _saved_stage == "eval_done":
                # Restore everything from the mid-iter checkpoint
                composite_prompt = _saved["composite_prompt"]
                selected = _saved["selected"]
                instruction = _saved["instruction"]
                score = _saved["score"]
                eval_tokens = _saved["eval_tokens"]
                failing_samples = _saved["failing_samples"]
                current_examples = selected
                current_instruction = instruction
                logger.info(f"  Resuming — reusing saved selection + eval (score: {score:.4f})")
            else:
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
                    eval_fn=eval_fn,
                )

                # Fold initial bootstrap tokens into the first iteration so they
                # flow through the IterationRecord machinery and survive resume.
                if i == 0 and _bootstrap_eval_tokens:
                    logger.debug(f"  Folding {_bootstrap_eval_tokens} bootstrap eval tokens into iteration 1")
                    eval_tokens += _bootstrap_eval_tokens
                    _bootstrap_eval_tokens = 0

                # Checkpoint after eval so a crash during the next refine call
                # doesn't force re-running this expensive eval. Also save
                # reasoning tokens spent this iteration (agent selection) so
                # they survive a resume that skips the selection on replay.
                _save_iter_state({"iter": i, "stage": "eval_done",
                                  "composite_prompt": composite_prompt,
                                  "selected": selected, "instruction": instruction,
                                  "score": score, "eval_tokens": eval_tokens,
                                  "failing_samples": failing_samples,
                                  "reasoning_tokens_so_far": getattr(agent, "tokens_used", 0) - reasoning_before})

            record = IterationRecord(
                iteration=i + 1,
                system_prompt=composite_prompt,
                score=score,
                reasoning=f"Selected {len(selected)} examples",
                eval_tokens=eval_tokens,
                is_full_eval=is_full_eval,
            )
            record.reasoning_tokens = (getattr(agent, "tokens_used", 0) - reasoning_before) + _saved_reasoning_tokens
            iterations.append(record)

            logger.info(f"  Score: {score:.4f} (target: {config.score_threshold:.4f})")
            logger.info(f"  Examples in prompt: {len(selected)}")

            if score >= config.score_threshold:
                if on_iteration:
                    on_iteration(record)
                logger.info("Score threshold met — stopping.")
                return OptimizationResult(
                    best_prompt=composite_prompt,
                    best_score=score,
                    iterations=iterations,
                    converged=True,
                )

            # Re-bootstrap periodically to discover new exemplars.
            # Tokens are folded into THIS iteration's record BEFORE on_iteration fires
            # so the checkpoint and running total both see the full cost.
            if (i + 1) % fs_config.bootstrap_rounds == 0 and i + 1 < config.max_iterations:
                logger.info("  Re-bootstrapping exemplar pool...")
                new_candidates, reboot_tokens = _bootstrap_exemplars(
                    prompt=composite_prompt,
                    dataset=dataset,
                    providers=providers,
                    metrics=metrics,
                    run_config=run_config,
                    pool_size=fs_config.candidate_pool_size,
                )
                record.eval_tokens += reboot_tokens
                # Merge pools, dedup by input
                existing_inputs = {e["input"] for e in candidate_exemplars}
                for c in new_candidates:
                    if c["input"] not in existing_inputs:
                        candidate_exemplars.append(c)
                        existing_inputs.add(c["input"])

            if update_strategy_state:
                update_strategy_state({
                    "candidate_exemplars": candidate_exemplars,
                    "bootstrap_done": True,
                    "bootstrap_eval_tokens": _bootstrap_eval_tokens,
                    "iters_done": i + 1,
                    "current_instruction": current_instruction,
                    "current_examples": current_examples,
                    "fewshot_iter_state": {},
                })
            _iter_state = {}
            if on_iteration:
                on_iteration(record)

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
    partial_candidates: list | None = None,
    on_sample: Any | None = None,
) -> tuple[list[dict[str, str]], int]:
    """Run eval sample-by-sample and extract top-scoring samples as candidate exemplars.

    Evaluates one sample at a time so partial results survive interruption.
    Already-scored samples (partial_candidates) are skipped on resume.

    Returns (candidates, total_eval_tokens) so callers can account for inference cost.
    """
    from aevyra_verdict import Dataset, EvalRunner
    from aevyra_verdict.dataset import Conversation, Message

    # Build index of already-scored inputs so we can skip them on resume
    already_scored_inputs: set[str] = {c["input"] for c in (partial_candidates or [])}
    scored_samples: list[dict[str, Any]] = list(partial_candidates or [])
    total_eval_tokens: int = 0

    def _make_runner() -> EvalRunner:
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
        return runner

    # Collect remaining (not-yet-scored) conversations in order
    remaining: list[Any] = []
    for convo in dataset.conversations:
        if (convo.last_user_message or "") not in already_scored_inputs:
            messages = list(convo.messages)
            if messages and messages[0].role == "system":
                messages[0] = Message(role="system", content=prompt)
            else:
                messages.insert(0, Message(role="system", content=prompt))
            remaining.append(Conversation(
                messages=messages,
                ideal=convo.ideal,
                metadata=convo.metadata,
            ))

    n_total = len(dataset.conversations)
    n_done = n_total - len(remaining)

    from tqdm import tqdm

    if not remaining:
        # All samples already scored (full resume) — nothing to do
        pass
    else:
        # Run all remaining samples in parallel batches.
        # Batch size = max_workers so we checkpoint after every batch while
        # still keeping requests concurrent within each batch.
        batch_size = run_config.max_workers or 4
        pbar = tqdm(total=n_total, initial=n_done)

        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start: batch_start + batch_size]
            batch_ds = Dataset(conversations=batch)

            runner = _make_runner()
            _judge_before = [getattr(m, "judge_tokens_used", 0) for m in metrics]
            results = runner.run(batch_ds, show_progress=False)
            total_eval_tokens += sum(mr.total_tokens() for mr in results.model_results.values())
            for _mi, _m in enumerate(metrics):
                total_eval_tokens += getattr(_m, "judge_tokens_used", 0) - _judge_before[_mi]

            for _label, model_result in results.model_results.items():
                for i, convo in enumerate(batch):
                    user_input = convo.last_user_message or ""
                    score = _mean_score(model_result.scores[i])
                    completion = model_result.completions[i]
                    output = convo.ideal if convo.ideal else (
                        completion.text if completion else ""
                    )
                    if user_input and output:
                        scored_samples.append({
                            "input": user_input,
                            "output": output,
                            "score": score,
                        })
                    already_scored_inputs.add(user_input)

            pbar.update(len(batch))

            # Checkpoint after each batch so resume can skip already-scored ones
            if on_sample:
                on_sample(list(scored_samples))

        pbar.close()

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

    return unique, total_eval_tokens


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
