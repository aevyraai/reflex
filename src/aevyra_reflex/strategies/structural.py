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

"""Structural optimization strategy.

Inspired by Microsoft's SAMMO framework. Instead of only tweaking the wording,
this strategy optimizes the *structure* of the prompt — section ordering,
formatting conventions, use of markdown vs. plain text, section headers,
constraint placement, and information hierarchy.

Each iteration:
  1. Run the current prompt through verdict
  2. Identify failures and their patterns
  3. Ask Claude to analyze the structural weaknesses of the prompt
  4. Generate structural variants (different organizations of the same content)
  5. Evaluate each variant **in parallel**; keep the best
  6. If the score meets the threshold, stop; otherwise repeat
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from aevyra_reflex.agent import Agent
from aevyra_reflex.result import IterationRecord, OptimizationResult
from aevyra_reflex.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class _StructuralConfig:
    """Extracted config for the structural strategy."""

    variants_per_round: int = 4
    bottom_k: int = 10


# Structural transformation axes — each represents a different way to
# reorganize a prompt while preserving its semantic content.
STRUCTURAL_TRANSFORMS = {
    "section_reorder": (
        "Reorder the sections of this prompt so that the most important "
        "constraints and instructions come first. Models attend more strongly "
        "to the beginning and end of prompts."
    ),
    "markdown_structure": (
        "Restructure this prompt using clear markdown formatting: headers for "
        "distinct sections, bullet points for lists of constraints, bold for "
        "critical instructions. Make the hierarchy of information explicit."
    ),
    "minimal_flat": (
        "Flatten this prompt into a single focused paragraph with no headers "
        "or formatting. Some models perform better with concise, flowing "
        "instructions rather than structured layouts."
    ),
    "xml_tags": (
        "Restructure this prompt using XML-style tags to delineate sections "
        "(e.g., <role>, <task>, <constraints>, <format>). This gives the model "
        "explicit structural markers to anchor on."
    ),
    "constraint_emphasis": (
        "Reorganize this prompt so that constraints and negative instructions "
        "(what NOT to do) are separated into their own clearly labeled section "
        "placed near the end. Positive instructions come first."
    ),
    "task_decomposition": (
        "Break the prompt's task into explicit numbered steps. Convert any "
        "implicit multi-step reasoning into an explicit step-by-step procedure "
        "the model should follow."
    ),
    "role_task_format": (
        "Restructure into three clear sections: (1) Role — who the model is, "
        "(2) Task — what it should do, (3) Format — how the output should look. "
        "Each section should be a separate paragraph or headed section."
    ),
    "input_anchored": (
        "Restructure the prompt so it explicitly references the structure of "
        "the user input. Add a section that describes what the model should "
        "expect as input and how to parse it before generating output."
    ),
}


class StructuralStrategy(Strategy):
    """Optimize prompt structure — section ordering, formatting, hierarchy.

    Generates multiple structural variants of the prompt each round, evaluates
    them, and keeps the best-performing structure. Combines structural search
    with Claude's analysis of which structural patterns help or hurt.
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
        from aevyra_reflex.strategies.iterative import _run_eval

        from aevyra_verdict.runner import RunConfig

        extra = config.extra_kwargs or {}
        s_config = _StructuralConfig(
            variants_per_round=extra.get("variants_per_round", 4),
            bottom_k=extra.get("bottom_k", 10),
        )

        run_config = RunConfig(
            temperature=config.eval_temperature,
            max_tokens=config.max_tokens,
            max_workers=config.max_workers,
        )

        current_prompt = initial_prompt
        iterations: list[IterationRecord] = []
        transforms_tried: set[str] = set()
        structural_history: list[dict[str, Any]] = []

        # Track which transforms have been available
        transform_keys = list(STRUCTURAL_TRANSFORMS.keys())

        _batch_size = getattr(config, "batch_size", 0)
        _batch_seed = getattr(config, "batch_seed", 42)
        _full_eval_steps = getattr(config, "full_eval_steps", 0)

        # Restore cross-iteration state on resume (transforms tried, history, current prompt)
        rs = resume_state or {}
        if rs.get("current_prompt"):
            current_prompt = rs["current_prompt"]
        if rs.get("transforms_tried"):
            transforms_tried = set(rs["transforms_tried"])
        if rs.get("structural_history"):
            structural_history = list(rs["structural_history"])

        # Mid-iteration checkpoint: stores stage results so a crash during
        # variant eval doesn't force re-running the base eval and analysis.
        _iter_state: dict = rs.get("structural_iter_state", {})

        def _save_iter_state(state: dict) -> None:
            if update_strategy_state:
                update_strategy_state({
                    "current_prompt": current_prompt,
                    "transforms_tried": list(transforms_tried),
                    "structural_history": structural_history,
                    "structural_iter_state": state,
                })

        for i in range(config.max_iterations):
            tag = f"[structural][iter {i + 1}/{config.max_iterations}]"

            # Check if we have saved mid-iteration state for this iteration
            _saved = _iter_state if _iter_state.get("iter") == i else {}
            _saved_stage = _saved.get("stage", "")

            logger.info(f"{tag} Starting")

            # Determine full-eval checkpoint or mini-batch for this iteration
            is_full_eval = (
                _batch_size > 0
                and _full_eval_steps > 0
                and (i + 1) % _full_eval_steps == 0
            )
            effective_batch = 0 if is_full_eval else _batch_size

            # 1. Evaluate current prompt (skip if already done in a previous session)
            if _saved_stage in ("base_eval_done", "analyzed", "variants_generated"):
                current_score = _saved["score"]
                eval_tokens = _saved["eval_tokens"]
                failing_samples = _saved["failing_samples"]
                logger.info(f"{tag} Resuming — reusing saved base eval (score: {current_score:.4f})")
            else:
                eval_label = "full-eval checkpoint" if is_full_eval else "eval"
                logger.info(f"{tag} Running {eval_label}...")
                current_score, failing_samples, eval_tokens = _run_eval(
                    prompt=current_prompt,
                    dataset=dataset,
                    providers=providers,
                    metrics=metrics,
                    run_config=run_config,
                    bottom_k=s_config.bottom_k,
                    batch_size=effective_batch,
                    iteration_seed=_batch_seed + i,
                    eval_fn=eval_fn,
                )
                _save_iter_state({"iter": i, "stage": "base_eval_done",
                                  "score": current_score, "eval_tokens": eval_tokens,
                                  "failing_samples": failing_samples})

            record = IterationRecord(
                iteration=i + 1,
                system_prompt=current_prompt,
                score=current_score,
                eval_tokens=eval_tokens,
                is_full_eval=is_full_eval,
            )
            iterations.append(record)
            # NOTE: on_iteration is called AFTER reasoning/variants are done below,
            # so the dashboard and checkpoint get reasoning_tokens + reasoning text.

            logger.info(f"{tag} Score: {current_score:.4f}  target: {config.score_threshold:.4f}  failing samples: {len(failing_samples)}")

            if current_score >= config.score_threshold:
                logger.info(f"{tag} Score threshold met — stopping.")
                return OptimizationResult(
                    best_prompt=current_prompt,
                    best_score=current_score,
                    iterations=iterations,
                    converged=True,
                )

            # 2. Ask reasoning model to analyze structural weaknesses
            reasoning_before = getattr(agent, "tokens_used", 0)
            # Reasoning tokens from a previous session (saved in mid-iter checkpoint)
            # are carried forward so a resume doesn't zero out the spent tokens.
            _saved_reasoning_tokens: int = _saved.get("reasoning_tokens_so_far", 0)
            if _saved_stage in ("analyzed", "variants_generated"):
                analysis = _saved["analysis"]
                round_transforms = _saved["round_transforms"]
                for t in round_transforms:
                    transforms_tried.add(t)
                logger.info(f"{tag} Resuming — reusing saved analysis")
            else:
                logger.info(f"{tag} Analyzing structural weaknesses (transform history: {len(structural_history)} entries)...")
                analysis = agent.analyze_prompt_structure(
                    system_prompt=current_prompt,
                    failing_samples=failing_samples,
                    structural_history=structural_history,
                )

                # 3. Pick transforms we haven't tried yet, plus agent-guided variant
                untried = [k for k in transform_keys if k not in transforms_tried]
                if not untried:
                    # All tried — reset and let the agent guide selection
                    transforms_tried.clear()
                    untried = transform_keys[:]

                # Select a subset for this round
                round_transforms = untried[: s_config.variants_per_round - 1]
                for t in round_transforms:
                    transforms_tried.add(t)
                _save_iter_state({"iter": i, "stage": "analyzed",
                                  "score": current_score, "eval_tokens": eval_tokens,
                                  "failing_samples": failing_samples,
                                  "analysis": analysis, "round_transforms": round_transforms,
                                  "reasoning_tokens_so_far": getattr(agent, "tokens_used", 0) - reasoning_before})

            # 4. Generate structural variants (in parallel)
            variants: list[tuple[str, str]] = []  # (transform_name, prompt_text)

            max_workers = min(len(round_transforms) + 1, config.max_workers or 4)
            num_variants = len(round_transforms) + 1  # transforms + freeform

            if _saved_stage == "variants_generated":
                variants = [tuple(v) for v in _saved["variants"]]  # type: ignore[misc]
                logger.info(f"{tag} Resuming — reusing {len(variants)} saved variants")
            else:
                def _gen_variant(t_name: str) -> tuple[str, str]:
                    t_instruction = STRUCTURAL_TRANSFORMS[t_name]
                    variant = agent.restructure_prompt(
                        current_prompt=current_prompt,
                        transform_instruction=t_instruction,
                        analysis=analysis,
                    )
                    return (t_name, variant)

                logger.info(f"{tag} Generating {num_variants} variants ({max_workers} workers)...")
                t0 = time.time()

                with ThreadPoolExecutor(max_workers=max_workers) as gen_pool:
                    gen_futures = {
                        gen_pool.submit(_gen_variant, t_name): t_name
                        for t_name in round_transforms
                    }
                    freeform_future = gen_pool.submit(
                        lambda: (
                            "agent_guided",
                            agent.freeform_restructure(
                                current_prompt=current_prompt,
                                failing_samples=failing_samples,
                                analysis=analysis,
                                score_trajectory=[r.score for r in iterations],
                                structural_history=structural_history,
                            ),
                        )
                    )

                    for future in as_completed(gen_futures):
                        t_name, variant = future.result()
                        variants.append((t_name, variant))

                    freeform_name, freeform_text = freeform_future.result()
                    variants.append((freeform_name, freeform_text))

                gen_elapsed = time.time() - t0
                logger.info(f"{tag} Generated {num_variants} variants in {gen_elapsed:.1f}s")
                _save_iter_state({"iter": i, "stage": "variants_generated",
                                  "score": current_score, "eval_tokens": eval_tokens,
                                  "failing_samples": failing_samples,
                                  "analysis": analysis, "round_transforms": round_transforms,
                                  "variants": [[n, p] for n, p in variants],
                                  "reasoning_tokens_so_far": getattr(agent, "tokens_used", 0) - reasoning_before})

            # 5. Evaluate all variants in parallel and pick the best
            best_variant_score = current_score
            best_variant_prompt = current_prompt
            best_variant_name = "current"
            variant_scores: dict[str, float] = {}

            # Variants use the same batch decision as the base eval for this
            # iteration so comparisons are fair (all on the same examples).
            _iter_batch_size = effective_batch
            _iter_seed = _batch_seed + i

            def _eval_variant(v_name: str, v_prompt: str) -> tuple[str, str, float, int]:
                v_score, _, v_toks = _run_eval(
                    prompt=v_prompt,
                    dataset=dataset,
                    providers=providers,
                    metrics=metrics,
                    run_config=run_config,
                    bottom_k=s_config.bottom_k,
                    batch_size=_iter_batch_size,
                    iteration_seed=_iter_seed,
                    eval_fn=eval_fn,
                )
                return v_name, v_prompt, v_score, v_toks

            logger.info(f"{tag} Evaluating {len(variants)} variants ({max_workers} workers)...")
            t0 = time.time()

            with ThreadPoolExecutor(max_workers=max_workers) as eval_pool:
                eval_futures = [
                    eval_pool.submit(_eval_variant, v_name, v_prompt)
                    for v_name, v_prompt in variants
                ]
                for future in as_completed(eval_futures):
                    v_name, v_prompt, v_score, v_toks = future.result()
                    record.eval_tokens += v_toks
                    variant_scores[v_name] = v_score
                    delta = v_score - current_score
                    sign = "+" if delta >= 0 else ""
                    effect = "✓ helped" if delta > 0.005 else ("✗ no effect" if abs(delta) <= 0.005 else "✗ hurt")
                    logger.info(f"{tag}   {v_name}: {v_score:.4f} (Δ{sign}{delta:.4f} — {effect})")

                    if v_score > best_variant_score:
                        best_variant_score = v_score
                        best_variant_prompt = v_prompt
                        best_variant_name = v_name

            eval_elapsed = time.time() - t0
            logger.info(
                f"{tag} Evaluated {len(variants)} variants in {eval_elapsed:.1f}s "
                f"({eval_elapsed / len(variants):.1f}s/variant)"
            )

            delta = best_variant_score - current_score
            sign = "+" if delta >= 0 else ""
            structural_history.append({
                "iteration": i + 1,
                "transforms_tried": [v[0] for v in variants],
                "best_transform": best_variant_name,
                "score_before": current_score,
                "score_after": best_variant_score,
                "delta": delta,
            })

            record.change_summary = best_variant_name if best_variant_name != "current" else ""

            # Store reasoning so the dashboard can display it
            variant_lines = "\n".join(
                f"  {name}: {score:.4f} ({'winner' if name == best_variant_name else ('Δ' + ('+' if score >= current_score else '') + f'{score - current_score:.4f}')})"
                for name, score in sorted(variant_scores.items(), key=lambda x: -x[1])
            )
            record.reasoning = (
                f"{analysis}\n\n"
                f"Transforms tried:\n{variant_lines}\n\n"
                f"Winner: {best_variant_name}"
            )

            if best_variant_name != "current":
                logger.info(f"{tag} Winner: {best_variant_name} ({current_score:.4f} → {best_variant_score:.4f}, Δ{sign}{delta:.4f})")
                current_prompt = best_variant_prompt
            else:
                logger.info(f"{tag} No variant improved over current prompt.")

            # Capture all reasoning tokens for this iteration (analysis + all variant generation).
            # Add any tokens spent in a previous session (restored from mid-iter checkpoint)
            # so a resume doesn't lose the pre-crash reasoning spend.
            record.reasoning_tokens = (getattr(agent, "tokens_used", 0) - reasoning_before) + _saved_reasoning_tokens

            # Fire callback now that reasoning_tokens + reasoning text are populated
            if on_iteration:
                on_iteration(record)

            # Clear mid-iter state now that this iteration is fully committed,
            # and persist updated cross-iteration state (transforms tried, history).
            _iter_state = {}
            _save_iter_state({})

        best = max(iterations, key=lambda r: r.score)
        return OptimizationResult(
            best_prompt=best.system_prompt,
            best_score=best.score,
            iterations=iterations,
            converged=False,
        )
