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
        resume_state: dict[str, Any] | None = None,
        update_strategy_state: Any | None = None,
    ) -> OptimizationResult:
        extra = config.extra_kwargs or {}
        a_config = _AutoConfig(
            max_phases=extra.get("max_phases", 4),
            min_phases=extra.get("min_phases", 2),
            phase_budgets=extra.get("phase_budgets"),
            start_structural=extra.get("start_structural", True),
        )
        phase_budgets = a_config.phase_budgets or dict(DEFAULT_PHASE_BUDGETS)

        # Restore phase state from checkpoint if resuming
        rs = resume_state or {}
        resume_phase_idx: int = rs.get("phase_idx", 0)
        resume_phase_iters_done: int = rs.get("phase_iters_done", 0)
        is_resuming = bool(rs)

        # Global state across all phases
        current_prompt = initial_prompt
        all_iterations: list[IterationRecord] = []
        axes_used: list[str] = list(rs.get("axes_used", []))
        phase_history: list[dict[str, Any]] = list(rs.get("phase_history", []))
        global_iter: int = rs.get("global_iter", 0)
        last_phase_score: float = rs.get("last_phase_score", 0.0)

        if is_resuming:
            logger.info(
                f"[auto] Resuming from phase {resume_phase_idx + 1}, "
                f"{resume_phase_iters_done} iteration(s) already done in that phase"
            )

        # fewshot requires labeled examples — exclude it for label-free datasets
        available_axes = list(AXES)
        if not dataset.has_ideals() and "fewshot" in available_axes:
            available_axes.remove("fewshot")
            logger.info("Label-free dataset — excluding fewshot axis (requires ideal answers)")

        for phase_idx in range(a_config.max_phases):
            # ----------------------------------------------------------
            # 0. Skip phases already completed before the crash
            # ----------------------------------------------------------
            if is_resuming and phase_idx < resume_phase_idx:
                logger.info(f"[auto] Skipping phase {phase_idx + 1} (already completed before crash)")
                continue

            # ----------------------------------------------------------
            # 1. Decide which axis to use
            # ----------------------------------------------------------
            if is_resuming and phase_idx == resume_phase_idx and axes_used:
                # Resume mid-phase: axis was already decided, reuse the saved one
                axis = axes_used[-1]
                logger.info(f"[auto] Resuming phase {phase_idx + 1} with axis '{axis}'")
            elif phase_idx == 0 and a_config.start_structural:
                axis = "structural"
                axes_used.append(axis)
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

            full_budget = phase_budgets.get(axis, 3)
            # For the phase that was in-progress, subtract already-done iterations
            iters_already_done = resume_phase_iters_done if (is_resuming and phase_idx == resume_phase_idx) else 0
            remaining_in_phase = full_budget - iters_already_done
            remaining_global = config.max_iterations - global_iter

            if remaining_in_phase <= 0:
                # This phase was already fully completed — skip to the next one
                logger.info(f"[auto] Phase {phase_idx + 1} ({axis}) already complete, advancing to next phase.")
                if is_resuming and phase_idx == resume_phase_idx:
                    is_resuming = False
                continue

            if remaining_global <= 0:
                logger.info("Global iteration budget exhausted.")
                break

            phase_budget = min(remaining_in_phase, remaining_global)

            if iters_already_done > 0:
                logger.info(
                    f"Phase {phase_idx + 1}: resuming '{axis}' — "
                    f"{iters_already_done}/{full_budget} done, {phase_budget} remaining"
                )
            else:
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
            _phase_iters_this_run = 0

            # Sub-strategy state (e.g. fewshot bootstrap candidates) — persisted
            # alongside auto's own state so resume can skip completed sub-work.
            # Seed from checkpoint if this is the phase we're resuming into.
            _sub_state_box: list[dict] = [
                rs.get("sub_strategy_state", {})
                if (is_resuming and phase_idx == resume_phase_idx) else {}
            ]

            def _save_phase_checkpoint(_phase_idx: int = phase_idx) -> None:
                if update_strategy_state:
                    state: dict[str, Any] = {
                        "phase_idx": _phase_idx,
                        "phase_iters_done": iters_already_done + _phase_iters_this_run,
                        "axes_used": list(axes_used),
                        "phase_history": list(phase_history),
                        "global_iter": global_iter,
                        "last_phase_score": last_phase_score,
                    }
                    if _sub_state_box[0]:
                        state["sub_strategy_state"] = _sub_state_box[0]
                    update_strategy_state(state)

            def _sub_update_strategy_state(sub_state: dict) -> None:
                """Relay sub-strategy state updates into the checkpoint."""
                _sub_state_box[0] = sub_state
                _save_phase_checkpoint()

            def _phase_callback(record: IterationRecord, _axis: str = axis, _phase_idx: int = phase_idx) -> None:
                # Re-number iterations globally
                nonlocal global_iter, _phase_iters_this_run
                global_iter += 1
                _phase_iters_this_run += 1
                record = IterationRecord(
                    iteration=global_iter,
                    system_prompt=record.system_prompt,
                    score=record.score,
                    scores_by_metric=record.scores_by_metric,
                    reasoning=f"[{_axis}] {record.reasoning}",
                    eval_tokens=getattr(record, "eval_tokens", 0),
                    reasoning_tokens=getattr(record, "reasoning_tokens", 0),
                    val_score=getattr(record, "val_score", None),
                    is_full_eval=getattr(record, "is_full_eval", False),
                    change_summary=getattr(record, "change_summary", ""),
                )
                all_iterations.append(record)
                # Persist phase state (including sub-strategy state) so resume
                # can skip completed work at both the auto and sub-strategy level.
                _save_phase_checkpoint(_phase_idx)
                if on_iteration:
                    on_iteration(record)

            # Pass saved sub-strategy state back on resume so fewshot can skip
            # bootstrap, PDO can restore its pool, etc.
            sub_resume_state = _sub_state_box[0] if _sub_state_box[0] else None

            sub_result = sub_strategy.run(
                initial_prompt=current_prompt,
                dataset=dataset,
                providers=providers,
                metrics=metrics,
                agent=agent,
                config=sub_config,
                on_iteration=_phase_callback,
                resume_state=sub_resume_state,
                update_strategy_state=_sub_update_strategy_state,
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
            # After completing the resumed phase, clear resume state so
            # subsequent phases run normally.
            if is_resuming and phase_idx == resume_phase_idx:
                is_resuming = False

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

            # Advance the checkpoint to the next phase so a crash between
            # phases doesn't re-run this one.
            # Also reset val history so early stopping starts fresh — one
            # phase's plateau shouldn't penalize the next strategy.
            if update_strategy_state:
                update_strategy_state({
                    "phase_idx": phase_idx + 1,
                    "phase_iters_done": 0,
                    "axes_used": list(axes_used),
                    "phase_history": list(phase_history),
                    "global_iter": global_iter,
                    "last_phase_score": last_phase_score,
                    "_reset_val_history": True,
                })

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
