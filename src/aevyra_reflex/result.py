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

"""Optimization results returned by PromptOptimizer.run()."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class IterationRecord:
    """Score and prompt from a single optimization iteration."""

    iteration: int
    system_prompt: str
    score: float
    scores_by_metric: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    eval_tokens: int = 0       # tokens used by the eval model this iteration
    reasoning_tokens: int = 0  # tokens used by the reasoning model this iteration
    change_summary: str = ""   # one-liner: what the reasoning model changed this iteration
    val_score: float | None = None  # validation set score (None when val_ratio=0)
    is_full_eval: bool = False  # True when scored on the full training set (periodic checkpoint), not a mini-batch


@dataclass
class SampleSnapshot:
    """A single sample's input, model output, ideal output, and score."""

    input: str
    response: str
    ideal: str
    score: float


@dataclass
class EvalSnapshot:
    """Scores from a single eval run (baseline or final)."""

    mean_score: float
    scores_by_metric: dict[str, float] = field(default_factory=dict)
    system_prompt: str = ""
    samples: list[SampleSnapshot] = field(default_factory=list)
    total_tokens: int = 0
    std_score: float = 0.0   # std dev of mean scores across eval_runs (0 when eval_runs=1)
    n_runs: int = 1          # number of eval passes averaged to produce mean_score


@dataclass
class OptimizationResult:
    """The full outcome of an optimization run: baseline → optimize → verify."""

    best_prompt: str
    best_score: float
    iterations: list[IterationRecord]
    converged: bool

    # Baseline and final verification scores (filled by optimizer)
    baseline: EvalSnapshot | None = None
    final: EvalSnapshot | None = None

    # Token usage (filled by optimizer)
    total_eval_tokens: int = 0       # all eval-model tokens across the run
    total_reasoning_tokens: int = 0  # all reasoning-model tokens across the run

    # Dataset split info (filled by optimizer when train_ratio < 1.0)
    train_size: int = 0  # examples used during optimization
    test_size: int = 0   # held-out examples used for baseline and final eval
    batch_size: int = 0  # per-iteration mini-batch size (0 = full training set)

    # Validation split info (filled by optimizer when val_ratio > 0)
    val_size: int = 0                     # examples in the validation set
    val_trajectory: list[float] = field(default_factory=list)  # val score per iteration
    early_stopped: bool = False           # True when early stopping triggered on val plateau

    # Statistical significance (filled by optimizer after run)
    p_value: float | None = None        # p-value from paired Wilcoxon/t-test (None if n < 2 or scipy missing)
    is_significant: bool | None = None  # True if p_value < 0.05

    # Strategy metadata (filled by auto strategy or optimizer)
    strategy_name: str = ""
    phase_history: list[dict] = field(default_factory=list)  # auto strategy phases

    # Timing (filled by optimizer)
    duration_seconds: float = 0.0

    @property
    def score_trajectory(self) -> list[float]:
        return [r.score for r in self.iterations]

    @property
    def improvement(self) -> float | None:
        """Absolute score improvement from baseline to final."""
        if self.baseline and self.final:
            return self.final.mean_score - self.baseline.mean_score
        return None

    @property
    def improvement_pct(self) -> float | None:
        """Percentage improvement from baseline to final."""
        if self.baseline and self.final and self.baseline.mean_score > 0:
            return (self.final.mean_score - self.baseline.mean_score) / self.baseline.mean_score * 100
        return None

    def summary(self) -> str:
        lines = []

        if self.baseline and self.final:
            imp = self.improvement or 0
            pct = self.improvement_pct or 0
            sign = "+" if imp >= 0 else ""

            def _fmt_score(snap: "EvalSnapshot") -> str:
                s = f"{snap.mean_score:.4f}"
                if snap.std_score > 0:
                    s += f" ± {snap.std_score:.4f}"
                if snap.n_runs > 1:
                    s += f"  ({snap.n_runs} runs)"
                return s

            lines.append("=" * 52)
            lines.append("  OPTIMIZATION RESULTS")
            lines.append("=" * 52)
            if self.train_size and self.test_size:
                if self.val_size:
                    lines.append(
                        f"  Train/val/test   : {self.train_size} / {self.val_size} / {self.test_size} samples"
                    )
                else:
                    lines.append(f"  Train / test     : {self.train_size} / {self.test_size} samples")
                if self.batch_size:
                    lines.append(f"  Batch size       : {self.batch_size} examples/iter  (mini-batch mode)")
                lines.append(f"  Baseline score   : {_fmt_score(self.baseline)}  (on {self.test_size}-sample test set)")
                lines.append(f"  Final score      : {_fmt_score(self.final)}  (on {self.test_size}-sample test set)")
            else:
                if self.batch_size:
                    lines.append(f"  Batch size       : {self.batch_size} examples/iter  (mini-batch mode)")
                lines.append(f"  Baseline score   : {_fmt_score(self.baseline)}")
                lines.append(f"  Final score      : {_fmt_score(self.final)}")
            lines.append(f"  Improvement      : {sign}{imp:.4f} ({sign}{pct:.1f}%)")
            if self.p_value is not None:
                sig_mark = "✓ significant" if self.is_significant else "✗ not significant"
                lines.append(f"  Significance     : p={self.p_value:.4f}  {sig_mark} (α=0.05, paired test)")
            elif self.baseline.samples and len(self.baseline.samples) < 2:
                lines.append("  Significance     : n/a (need ≥2 samples)")
            else:
                lines.append("  Significance     : install scipy for p-values")
            lines.append(f"  Iterations       : {len(self.iterations)}")
            if self.early_stopped:
                lines.append("  Early stopped    : Yes (val score plateaued)")
            lines.append(f"  Converged        : {self.converged}")
            if self.total_eval_tokens or self.total_reasoning_tokens:
                def _fmt_tok(n): return f"{n/1e6:.2f}M" if n >= 1_000_000 else (f"{n/1000:.1f}K" if n >= 1000 else str(n))
                lines.append(f"  Eval tokens      : {_fmt_tok(self.total_eval_tokens)}")
                lines.append(f"  Reasoning tokens : {_fmt_tok(self.total_reasoning_tokens)}")
            if self.duration_seconds > 0:
                _dm, _ds = divmod(int(self.duration_seconds), 60)
                _dh, _dm = divmod(_dm, 60)
                _dlabel = (
                    f"{_dh}h {_dm}m {_ds}s" if _dh
                    else f"{_dm}m {_ds}s" if _dm
                    else f"{_ds}s"
                )
                lines.append(f"  Duration         : {_dlabel}")
            lines.append("-" * 52)

            if self.baseline.scores_by_metric or self.final.scores_by_metric:
                lines.append("  Per-metric breakdown:")
                all_metrics = set(self.baseline.scores_by_metric) | set(self.final.scores_by_metric)
                for m in sorted(all_metrics):
                    b = self.baseline.scores_by_metric.get(m, 0)
                    f = self.final.scores_by_metric.get(m, 0)
                    d = f - b
                    s = "+" if d >= 0 else ""
                    lines.append(f"    {m:30s}  {b:.4f} → {f:.4f}  ({s}{d:.4f})")
                lines.append("-" * 52)

            lines.append(f"  Train traj : {' → '.join(f'{s:.3f}' for s in self.score_trajectory)}")
            if self.val_trajectory:
                lines.append(f"  Val traj   : {' → '.join(f'{s:.3f}' for s in self.val_trajectory)}")
            lines.append("=" * 52)

            # What happened (trajectory + strategy)
            analysis = self._analyze_trajectory()
            strategy_insight = self._analyze_strategy()
            prompt_insight = self._analyze_prompt_changes()

            if analysis or strategy_insight or prompt_insight:
                lines.append("")
                lines.append("  WHAT HAPPENED")
                lines.append("-" * 52)
                for line in analysis:
                    lines.append(f"  {line}")
                if strategy_insight:
                    if analysis:
                        lines.append("")
                    for line in strategy_insight:
                        lines.append(f"  {line}")
                if prompt_insight:
                    lines.append("")
                    for line in prompt_insight:
                        lines.append(f"  {line}")
                lines.append("=" * 52)

            # Before/after example
            example = self._before_after_example()
            if example:
                lines.append("")
                lines.append("  BEFORE / AFTER EXAMPLE")
                lines.append("-" * 52)
                for line in example:
                    lines.append(line)
                lines.append("=" * 52)
        else:
            lines.append(f"Optimization complete — {len(self.iterations)} iteration(s)")
            lines.append(f"  Best score : {self.best_score:.4f}")
            lines.append(f"  Converged  : {self.converged}")
            lines.append(f"  Trajectory : {' → '.join(f'{s:.3f}' for s in self.score_trajectory)}")

        return "\n".join(lines)

    def _analyze_trajectory(self) -> list[str]:
        """Explain what the trajectory means and what to do next."""
        traj = self.score_trajectory
        if len(traj) < 2:
            return []

        lines: list[str] = []
        baseline = self.baseline.mean_score if self.baseline else traj[0]
        final = self.final.mean_score if self.final else traj[-1]
        peak_score = max(traj)
        peak_iter = traj.index(peak_score) + 1
        n = len(traj)

        eps = 0.015  # tolerance for "flat" — ±1.5% noise is not a real change
        improving_steps = sum(1 for i in range(1, n) if traj[i] > traj[i - 1] + eps)
        declining_steps = sum(1 for i in range(1, n) if traj[i] < traj[i - 1] - eps)
        flat_steps = n - 1 - improving_steps - declining_steps

        # Detect over-optimization: peak in first half and significant tail drop
        over_optimized = (
            peak_iter <= n // 2
            and peak_iter < n
            and traj[-1] < peak_score * 0.85
        )

        # --- What happened ---
        if self.converged:
            lines.append(
                f"The optimizer found a prompt that meets the target "
                f"threshold in {peak_iter} iterations. The model "
                f"responds well to the optimized instructions."
            )
        elif over_optimized:
            lines.append(
                f"The optimizer found a strong prompt at iteration "
                f"{peak_iter}, but continued revisions after that "
                f"point made things worse. This is "
                f"over-optimization — the agent kept trying to "
                f"improve what was already working and introduced "
                f"regressions on some samples. The best prompt "
                f"(from iteration {peak_iter}) was saved, not the "
                f"last one."
            )
        elif flat_steps > n // 2:
            lines.append(
                "The score plateaued — it jumped initially then "
                "stopped improving despite further prompt changes. "
                "This usually means the model has hit its capability "
                "ceiling for this task. A larger model or adding "
                "few-shot examples may be needed to go higher."
            )
        elif improving_steps > declining_steps * 2:
            lines.append(
                "The score improved steadily across iterations, "
                "meaning the model kept benefiting from each prompt "
                "revision. This suggests the model is responsive to "
                "prompt changes and more iterations could help "
                "further."
            )
        else:
            lines.append(
                "The score fluctuated between iterations, meaning "
                "some prompt changes helped certain samples but hurt "
                "others. With a small dataset, this is common — "
                "individual samples have outsized impact on the mean "
                "score."
            )

        # --- What does the improvement mean ---
        if baseline > 0 and final > baseline:
            gap_closed = (final - baseline) / (1.0 - baseline) * 100 if baseline < 1.0 else 0
            lines.append(
                f"The optimization closed {gap_closed:.0f}% of the gap "
                f"between the baseline ({baseline:.3f}) and a perfect "
                f"score (1.0)."
            )

        # --- What to try next ---
        suggestions: list[str] = []

        if not self.converged:
            if improving_steps > declining_steps and traj[-1] >= traj[-2]:
                suggestions.append(
                    "increase --max-iterations (the score was still "
                    "climbing)"
                )
            if over_optimized:
                suggestions.append(
                    f"use --max-iterations {peak_iter + 2} to stop "
                    f"before over-optimization kicks in"
                )
            if flat_steps > n // 2:
                suggestions.append(
                    "try a larger model — the current one may have "
                    "hit its capability limit"
                )
            if n <= 5:
                suggestions.append(
                    "add more samples to the dataset for more stable "
                    "scores"
                )
            if self.baseline and self.final:
                score_gap = 1.0 - final
                if score_gap > 0.3:
                    suggestions.append(
                        "try the 'fewshot' strategy — adding examples "
                        "often helps smaller models"
                    )

        if suggestions:
            lines.append("")
            lines.append("To improve further:")
            for s in suggestions:
                lines.append(f"  - {s}")

        return lines

    def _analyze_strategy(self) -> list[str]:
        """Explain which strategies were used, what they contributed,
        and teach the user prompt-engineering lessons from each phase."""
        lines: list[str] = []

        # Per-axis educational explanations (helped / hurt / no change)
        _lessons_helped = {
            "structural": (
                "Structure matters. The model responded much better "
                "when the same instructions were organized with clear "
                "sections, headers, and hierarchy. Lesson: even if "
                "your prompt has the right content, poor layout can "
                "hide it from the model. Use markdown headers, bullet "
                "points, and logical ordering."
            ),
            "iterative": (
                "Specificity matters. The model improved when vague "
                "instructions were replaced with precise ones — "
                "explicit constraints, concrete examples of what to "
                "avoid, and clearer success criteria. Lesson: don't "
                "assume the model knows what you mean. Spell out "
                "edge cases, output requirements, and common mistakes."
            ),
            "fewshot": (
                "Examples matter. The model understood the task "
                "description but couldn't produce the right output "
                "format until it saw concrete examples. Lesson: when "
                "your expected output has a specific structure (tables, "
                "headers, bullet patterns), showing 2-3 examples is "
                "often more effective than describing the format in "
                "words."
            ),
            "pdo": (
                "Exploration matters. The best prompt emerged from "
                "testing many variants, not from a single chain of "
                "revisions. Lesson: when one approach to prompting "
                "isn't working, try a fundamentally different angle "
                "rather than tweaking the same prompt repeatedly."
            ),
        }
        _lessons_hurt = {
            "structural": (
                "Restructuring made things worse — the original "
                "layout was fine, and reorganizing introduced "
                "confusion. Lesson: don't over-format a prompt that "
                "already works. Unnecessary headers or sections can "
                "distract the model from the actual task."
            ),
            "iterative": (
                "Further wording revisions hurt — the prompt was "
                "already clear enough, and additional constraints "
                "over-constrained the model. Lesson: there's a sweet "
                "spot for specificity. Too many rules can make the "
                "model focus on following constraints rather than "
                "doing the task well."
            ),
            "fewshot": (
                "Adding examples made things worse — the model may "
                "have over-fitted to the examples instead of "
                "generalizing. Lesson: few-shot examples should be "
                "diverse. If they're too similar, the model may "
                "copy patterns that don't apply to other inputs."
            ),
            "pdo": (
                "Tournament search made things worse — exploring too "
                "many variants introduced instability. Lesson: if the "
                "prompt is already performing well, broad search can "
                "stumble into worse regions of the prompt space."
            ),
        }
        _lessons_neutral = {
            "structural": (
                "Restructuring had no effect — the model doesn't "
                "seem sensitive to layout for this task. Lesson: "
                "structure mainly helps when prompts are long or "
                "multi-part. Short, focused prompts may not benefit."
            ),
            "iterative": (
                "Wording revisions had no effect — the instructions "
                "were already clear enough. Lesson: if the model "
                "understands the task but still scores low, the "
                "bottleneck may be the model's ability, not the "
                "prompt's clarity."
            ),
            "fewshot": (
                "Adding examples had no effect — the model already "
                "knew the expected format. Lesson: few-shot examples "
                "are most useful when the output format is unusual or "
                "complex. For straightforward tasks, they add tokens "
                "without improving quality."
            ),
            "pdo": (
                "Tournament search found nothing better — the current "
                "prompt may be near-optimal for this model. Lesson: "
                "when multiple diverse variants all score the same, "
                "the ceiling is likely the model's capability, not "
                "the prompt."
            ),
        }

        if self.phase_history:
            lines.append("Strategy breakdown (auto mode):")
            for phase in self.phase_history:
                axis = phase.get("axis", "?")
                before = phase.get("score_before", 0)
                after = phase.get("score_after", 0)
                delta = after - before
                sign = "+" if delta >= 0 else ""

                label = {
                    "structural": "restructured the prompt's formatting and layout",
                    "iterative": "diagnosed failures and revised the wording",
                    "fewshot": "added few-shot examples to the prompt",
                    "pdo": "searched over prompt variants via tournament",
                }.get(axis, axis)

                if delta > 0.02:
                    verdict = "helped"
                elif delta < -0.02:
                    verdict = "hurt"
                else:
                    verdict = "no change"

                lines.append(
                    f"  Phase {phase.get('phase', '?')} ({axis}): "
                    f"{label} — {sign}{delta:.3f} ({verdict})"
                )

            # Educational lessons for every phase
            lines.append("")
            lines.append("What each phase teaches about prompt writing:")
            for phase in self.phase_history:
                axis = phase.get("axis", "?")
                delta = phase.get("score_after", 0) - phase.get("score_before", 0)

                if delta > 0.02:
                    lesson = _lessons_helped.get(axis, "")
                elif delta < -0.02:
                    lesson = _lessons_hurt.get(axis, "")
                else:
                    lesson = _lessons_neutral.get(axis, "")

                if lesson:
                    lines.append(f"  [{axis}] {lesson}")
                    lines.append("")

            # Remove trailing blank line
            if lines and lines[-1] == "":
                lines.pop()

        elif self.strategy_name:
            # Single strategy — give the relevant lesson
            traj = self.score_trajectory
            if len(traj) >= 2:
                delta = traj[-1] - traj[0]
            else:
                delta = 0.0
            axis = self.strategy_name

            if delta > 0.02:
                lesson = _lessons_helped.get(axis, "")
            elif delta < -0.02:
                lesson = _lessons_hurt.get(axis, "")
            else:
                lesson = _lessons_neutral.get(axis, "")

            if lesson:
                lines.append(lesson)

        return lines

    def _analyze_prompt_changes(self) -> list[str]:
        """Explain what changed between the original and optimized prompt."""
        if not self.baseline or not self.baseline.system_prompt:
            return []
        original = self.baseline.system_prompt
        optimized = self.best_prompt
        if not optimized or original == optimized:
            return []

        lines: list[str] = []
        lines.append("What changed in the prompt:")

        # Length comparison
        orig_words = len(original.split())
        opt_words = len(optimized.split())
        if opt_words > orig_words * 2:
            lines.append(
                f"  - Much longer ({orig_words} → {opt_words} words). "
                f"The model needed more detailed instructions."
            )
        elif opt_words > orig_words * 1.3:
            lines.append(
                f"  - Longer ({orig_words} → {opt_words} words). "
                f"Added specificity helped the model."
            )
        elif opt_words < orig_words * 0.7:
            lines.append(
                f"  - Shorter ({orig_words} → {opt_words} words). "
                f"Removing noise improved focus."
            )

        # Structural markers
        opt_lower = optimized.lower()
        orig_lower = original.lower()
        new_features = []

        if "##" in optimized and "##" not in original:
            new_features.append("markdown headers for clear sections")
        if "**" in optimized and "**" not in original:
            new_features.append("bold emphasis on key instructions")
        if "<" in optimized and ">" in optimized and "<" not in original:
            new_features.append("XML tags for structural clarity")
        if optimized.count("\n-") > original.count("\n-") + 1:
            new_features.append("bullet points for constraints/rules")
        if optimized.count("example") > original.count("example") + 1:
            new_features.append("examples showing expected behavior")
        if ("step" in opt_lower and "step" not in orig_lower) or \
           (optimized.count("1.") > original.count("1.")):
            new_features.append("step-by-step instructions")
        if any(w in opt_lower for w in ["do not", "never", "avoid", "don't"]) and \
           not any(w in orig_lower for w in ["do not", "never", "avoid", "don't"]):
            new_features.append("explicit constraints on what to avoid")
        if any(w in opt_lower for w in ["format", "structure", "## summary", "key facts"]) and \
           not any(w in orig_lower for w in ["format", "structure", "## summary", "key facts"]):
            new_features.append("output format specification")

        if new_features:
            lines.append("  - Added: " + ", ".join(new_features))

        return lines

    def _before_after_example(self) -> list[str]:
        """Show a side-by-side comparison of one sample before and after optimization."""
        if not self.baseline or not self.final:
            return []
        if not self.baseline.samples or not self.final.samples:
            return []

        # Find the sample with the biggest improvement
        best_delta = -float("inf")
        best_idx = 0
        for i in range(min(len(self.baseline.samples), len(self.final.samples))):
            delta = self.final.samples[i].score - self.baseline.samples[i].score
            if delta > best_delta:
                best_delta = delta
                best_idx = i

        before = self.baseline.samples[best_idx]
        after = self.final.samples[best_idx]

        lines: list[str] = []
        lines.append("")

        # Truncate long text for display
        input_text = before.input
        if len(input_text) > 300:
            input_text = input_text[:300] + "..."

        lines.append(f"  Input (sample {best_idx + 1}):")
        lines.append(f"    {input_text}")
        lines.append("")

        if before.ideal:
            ideal_text = before.ideal
            if len(ideal_text) > 300:
                ideal_text = ideal_text[:300] + "..."
            lines.append("  Expected:")
            lines.append(f"    {ideal_text}")
            lines.append("")

        lines.append(f"  BEFORE (score: {before.score:.3f}):")
        before_text = before.response
        if len(before_text) > 500:
            before_text = before_text[:500] + "..."
        for line in before_text.split("\n"):
            lines.append(f"    {line}")
        lines.append("")

        lines.append(f"  AFTER (score: {after.score:.3f}):")
        after_text = after.response
        if len(after_text) > 500:
            after_text = after_text[:500] + "..."
        for line in after_text.split("\n"):
            lines.append(f"    {line}")
        lines.append("")

        lines.append(f"  Score change: {before.score:.3f} → {after.score:.3f} ({'+' if best_delta >= 0 else ''}{best_delta:.3f})")

        return lines

    def to_dict(self) -> dict:
        d = {
            "best_prompt": self.best_prompt,
            "best_score": self.best_score,
            "converged": self.converged,
            "iterations": [
                {
                    "iteration": r.iteration,
                    "score": r.score,
                    "scores_by_metric": r.scores_by_metric,
                    "system_prompt": r.system_prompt,
                    "reasoning": r.reasoning,
                    "eval_tokens": r.eval_tokens,
                    "reasoning_tokens": r.reasoning_tokens,
                    **({"val_score": r.val_score} if r.val_score is not None else {}),
                    **({"is_full_eval": True} if r.is_full_eval else {}),
                }
                for r in self.iterations
            ],
            "total_eval_tokens": self.total_eval_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
        }
        if self.baseline:
            d["baseline"] = {
                "mean_score": self.baseline.mean_score,
                "std_score": self.baseline.std_score,
                "n_runs": self.baseline.n_runs,
                "scores_by_metric": self.baseline.scores_by_metric,
                "system_prompt": self.baseline.system_prompt,
                "total_tokens": self.baseline.total_tokens,
            }
        if self.final:
            d["final"] = {
                "mean_score": self.final.mean_score,
                "std_score": self.final.std_score,
                "n_runs": self.final.n_runs,
                "scores_by_metric": self.final.scores_by_metric,
                "system_prompt": self.final.system_prompt,
                "total_tokens": self.final.total_tokens,
            }
        if self.train_size:
            d["train_size"] = self.train_size
        if self.test_size:
            d["test_size"] = self.test_size
        if self.val_size:
            d["val_size"] = self.val_size
        if self.val_trajectory:
            d["val_trajectory"] = self.val_trajectory
        if self.early_stopped:
            d["early_stopped"] = self.early_stopped
        if self.batch_size:
            d["batch_size"] = self.batch_size
        if self.improvement is not None:
            d["improvement"] = self.improvement
            d["improvement_pct"] = self.improvement_pct
        if self.p_value is not None:
            d["p_value"] = self.p_value
            d["is_significant"] = self.is_significant
        if self.strategy_name:
            d["strategy_name"] = self.strategy_name
        if self.phase_history:
            d["phase_history"] = self.phase_history
        return d

    def to_json(self, path: str | Path) -> None:
        """Save the full result to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def save_best_prompt(self, path: str | Path) -> None:
        """Save the best prompt to a markdown file."""
        Path(path).write_text(self.best_prompt)
