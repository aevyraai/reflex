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

"""Auto strategy — adaptive multi-phase prompt optimization.

Instead of asking the user to pick a strategy, ``auto`` runs a diagnostic
after each phase and decides which optimization axis to apply next.  The
four axes are complementary and stack on top of each other:

  structural  → fix the organization / formatting  (foundation)
  iterative   → fix the wording based on failures  (targeted edits)
  fewshot     → add demonstrations to cover gaps    (additive)
  pdo         → tournament search to polish          (refinement)

The auto strategy works as a **budget-aware pipeline**:

1. Run baseline eval.
2. Claude analyzes failures and recommends an *optimization axis*.
3. Apply that axis for a few iterations (its "phase budget").
4. Re-evaluate.  If the threshold is met, stop.
5. Otherwise, Claude sees what changed and picks the next axis.
6. Repeat until the global budget runs out.

Each phase inherits the best prompt from the previous phase, so
structural improvements are preserved when iterative edits are layered
on, and few-shot examples are added on top of both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from aevyra_reflex.agent import Agent
from aevyra_reflex.result import IterationRecord, OptimizationResult
from aevyra_reflex.strategies.base import Strategy

logger = logging.getLogger(__name__)

# Axis names that Claude can recommend.
AXES = ("structural", "iterative", "fewshot", "pdo")

# Default iteration budgets *per phase* (can be overridden via extra_kwargs).
DEFAULT_PHASE_BUDGETS: dict[str, int] = {
    "structural": 3,
    "iterative": 4,
    "fewshot": 3,
    "pdo": 15,
}


@dataclass
class _AutoConfig:
    """Extracted config for the auto strategy."""

    max_phases: int = 4
    min_phases: int = 2
    phase_budgets: dict[str, int] | None = None
    # If True, always start with structural regardless of Claude's advice.
    start_structural: bool = True


class AutoStrategy(Strategy):
    """Adaptive multi-phase optimization that combines all four axes.

    The user doesn't need to choose a strategy — ``auto`` handles it.
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
        extra = config.extra_kwargs or {}
        a_config = _AutoConfig(
            max_phases=extra.get("max_phases", 4),
            min_phases=extra.get("min_phases", 2),
            phase_budgets=extra.get("phase_budgets"),
            start_structural=extra.get("start_structural", True),
        )
        phase_budgets = a_config.phase_budgets or dict(DEFAULT_PHASE_BUDGETS)

        # Global state across all phases
        current_prompt = initial_prompt
        all_iterations: list[IterationRecord] = []
        axes_used: list[str] = []
        phase_history: list[dict[str, Any]] = []
        global_iter = 0
        last_phase_score = 0.0  # tracks the best score entering each phase

        # fewshot requires labeled examples — exclude it for label-free datasets
        available_axes = list(AXES)
        if not dataset.has_ideals() and "fewshot" in available_axes:
            available_axes.remove("fewshot")
            logger.info("Label-free dataset — excluding fewshot axis (requires ideal answers)")

        for phase_idx in range(a_config.max_phases):
            # ----------------------------------------------------------
            # 1. Decide which axis to use
            # ----------------------------------------------------------
            if phase_idx == 0 and a_config.start_structural:
                axis = "structural"
            else:
                axis = agent.recommend_axis(
                    current_prompt=current_prompt,
                    dataset_sample=_sample_inputs(dataset, n=5),
                    phase_history=phase_history,
                    axes_available=[a for a in available_axes if a not in axes_used],
                    axes_used=axes_used,
                )
                # Validate
                if axis not in AXES:
                    logger.warning(f"Agent recommended unknown axis {axis!r}, falling back to iterative")
                    axis = "iterative"

            axes_used.append(axis)
            phase_budget = min(
                phase_budgets.get(axis, 3),
                config.max_iterations - global_iter,
            )
            if phase_budget <= 0:
                logger.info("Global iteration budget exhausted.")
                break

            logger.info(f"Phase {phase_idx + 1}: applying '{axis}' for up to {phase_budget} iterations")

            # ----------------------------------------------------------
            # 2. Build a sub-config for this phase
            # ----------------------------------------------------------
            from aevyra_reflex.optimizer import OptimizerConfig

            sub_config = OptimizerConfig(
                strategy=axis,
                max_iterations=phase_budget,
                score_threshold=config.score_threshold,
                reasoning_model=config.reasoning_model,
                eval_temperature=config.eval_temperature,
                max_tokens=config.max_tokens,
                max_workers=config.max_workers,
                extra_kwargs=extra.get(f"{axis}_kwargs", {}),
            )

            # ----------------------------------------------------------
            # 3. Run the sub-strategy
            # ----------------------------------------------------------
            from aevyra_reflex.strategies import get_strategy

            sub_strategy = get_strategy(axis)()

            def _phase_callback(record: IterationRecord) -> None:
                # Re-number iterations globally
                nonlocal global_iter
                global_iter += 1
                record = IterationRecord(
                    iteration=global_iter,
                    system_prompt=record.system_prompt,
                    score=record.score,
                    scores_by_metric=record.scores_by_metric,
                    reasoning=f"[{axis}] {record.reasoning}",
                )
                all_iterations.append(record)
                if on_iteration:
                    on_iteration(record)

            sub_result = sub_strategy.run(
                initial_prompt=current_prompt,
                dataset=dataset,
                providers=providers,
                metrics=metrics,
                agent=agent,
                config=sub_config,
                on_iteration=_phase_callback,
            )

            # If sub-strategy didn't report via callback, absorb its iterations
            if not any(r.reasoning.startswith(f"[{axis}]") for r in all_iterations):
                for rec in sub_result.iterations:
                    global_iter += 1
                    adjusted = IterationRecord(
                        iteration=global_iter,
                        system_prompt=rec.system_prompt,
                        score=rec.score,
                        scores_by_metric=rec.scores_by_metric,
                        reasoning=f"[{axis}] {rec.reasoning}",
                    )
                    all_iterations.append(adjusted)
                    if on_iteration:
                        on_iteration(adjusted)

            # ----------------------------------------------------------
            # 4. Record phase outcome
            # ----------------------------------------------------------
            # For the first phase, use the sub-strategy's first iteration
            # score as the "before" (since no baseline was run inside auto).
            if phase_idx == 0 and sub_result.iterations:
                last_phase_score = sub_result.iterations[0].score

            phase_history.append({
                "phase": phase_idx + 1,
                "axis": axis,
                "iterations_used": len(sub_result.iterations),
                "score_before": last_phase_score,
                "score_after": sub_result.best_score,
                "improvement": sub_result.best_score - last_phase_score,
                "converged": sub_result.converged,
            })
            last_phase_score = sub_result.best_score

            current_prompt = sub_result.best_prompt

            logger.info(
                f"  Phase {phase_idx + 1} ({axis}) done: "
                f"score={sub_result.best_score:.4f}, "
                f"converged={sub_result.converged}"
            )

            # ----------------------------------------------------------
            # 5. Check if we've hit the threshold
            # ----------------------------------------------------------
            # Don't stop early until we've run at least min_phases —
            # LLM judge scores are noisy, so a single high score may not hold.
            phases_done = phase_idx + 1
            if phases_done >= a_config.min_phases and (
                sub_result.converged or sub_result.best_score >= config.score_threshold
            ):
                logger.info("Score threshold met — stopping auto strategy.")
                return OptimizationResult(
                    best_prompt=current_prompt,
                    best_score=sub_result.best_score,
                    iterations=all_iterations,
                    converged=True,
                    strategy_name="auto",
                    phase_history=phase_history,
                )

            if global_iter >= config.max_iterations:
                logger.info("Global iteration budget exhausted.")
                break

        # Exhausted all phases
        best = max(all_iterations, key=lambda r: r.score) if all_iterations else IterationRecord(
            iteration=0, system_prompt=initial_prompt, score=0.0
        )
        return OptimizationResult(
            best_prompt=best.system_prompt,
            best_score=best.score,
            iterations=all_iterations,
            converged=False,
            strategy_name="auto",
            phase_history=phase_history,
        )


def _sample_inputs(dataset: Any, n: int = 5) -> list[str]:
    """Get a few sample user inputs from the dataset."""
    inputs = []
    for convo in dataset.conversations[:n]:
        msg = convo.last_user_message
        if msg:
            inputs.append(msg)
    return inputs
