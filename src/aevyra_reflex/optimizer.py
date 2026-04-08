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

"""PromptOptimizer — single entry point for the full optimize workflow.

Runs baseline eval → optimization loop → final verification eval, then
presents a before/after comparison. All evaluation is done via verdict.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aevyra_reflex.agent import LLM
from aevyra_reflex.result import EvalSnapshot, OptimizationResult, SampleSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict results parsing — extract target scores from verdict output
# ---------------------------------------------------------------------------

def parse_verdict_results(
    path: str | Path,
    metric: str | None = None,
) -> dict[str, Any]:
    """Parse a verdict results JSON file and extract model scores.

    Returns a dict with:
        - models: dict mapping label → {provider, model, mean_scores: {metric: score}}
        - metrics: list of metric names
        - best_model: label of the highest-scoring model
        - best_score: the best model's mean score (for the given metric, or first metric)
        - target_model: same as best_model (the one we're aiming to match)
        - target_score: same as best_score

    Args:
        path: Path to a verdict results.json file.
        metric: Which metric to use for ranking. If None, uses the first metric.
    """
    data = json.loads(Path(path).read_text())

    metrics = data.get("metrics", [])
    if not metrics:
        raise ValueError(f"No metrics found in verdict results: {path}")

    # Pick the metric to rank by
    rank_metric = metric or metrics[0]

    models_summary = data.get("models", {})
    if not models_summary:
        raise ValueError(f"No model results found in verdict results: {path}")

    # Build structured model info
    models: dict[str, dict[str, Any]] = {}
    for label, info in models_summary.items():
        mean_scores = {}
        for m in metrics:
            val = info.get(f"{m}_mean")
            if val is not None:
                mean_scores[m] = val
        models[label] = {
            "provider": info.get("provider", ""),
            "model": info.get("model", ""),
            "mean_scores": mean_scores,
        }

    # Find best model by the ranking metric
    best_label = None
    best_score = -1.0
    for label, info in models.items():
        score = info["mean_scores"].get(rank_metric, 0.0)
        if score > best_score:
            best_score = score
            best_label = label

    return {
        "models": models,
        "metrics": metrics,
        "best_model": best_label,
        "best_score": best_score,
        "target_model": best_label,
        "target_score": best_score,
    }


# ---------------------------------------------------------------------------
# Provider aliases — OpenAI-compatible services that can be used with -m
# ---------------------------------------------------------------------------
# Maps alias → (base_url, env_var_for_api_key)
# When a user writes `-m openrouter/meta-llama/llama-3.1-8b-instruct`, we
# resolve "openrouter" to the openai provider with the right base_url and key.
PROVIDER_ALIASES: dict[str, dict[str, str]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_key": "FIREWORKS_API_KEY",
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "env_key": "DEEPINFRA_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
    },
    "lepton": {
        "base_url": "https://api.lepton.ai/v1",
        "env_key": "LEPTON_API_KEY",
    },
    "perplexity": {
        "base_url": "https://api.perplexity.ai",
        "env_key": "PERPLEXITY_API_KEY",
    },
}


def _resolve_provider(
    provider_name: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Resolve a provider alias to its actual provider_name, base_url, and api_key.

    If provider_name is a known alias (e.g., "openrouter"), it's mapped to the
    "openai" provider with the appropriate base_url and API key from env vars.
    Native providers ("openai", "anthropic", "google", "local") pass through unchanged.
    """
    alias = PROVIDER_ALIASES.get(provider_name)
    if alias:
        resolved_base_url = base_url or alias["base_url"]
        resolved_api_key = api_key or os.environ.get(alias["env_key"]) or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            logger.warning(
                f"No API key found for {provider_name}. "
                f"Set {alias['env_key']} or OPENAI_API_KEY."
            )
        return {
            "provider_name": "openai",
            "model": model,
            "api_key": resolved_api_key,
            "base_url": resolved_base_url,
        }

    return {
        "provider_name": provider_name,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }


@dataclass
class OptimizerConfig:
    """Configuration for a prompt optimization run."""

    max_iterations: int = 10
    """Stop after this many iterations (iterative) or rounds (pdo)."""

    score_threshold: float = 0.85
    """Stop early when mean score meets or exceeds this."""

    strategy: str = "iterative"
    """Optimization strategy: 'iterative' (diagnose-revise) or 'pdo' (dueling bandits)."""

    reasoning_model: str = "claude-sonnet-4-20250514"
    """LLM used for reasoning during optimization (analyzing failures, proposing
    rewrites, recommending strategies). Default: Claude Sonnet."""

    reasoning_provider: str | None = None
    """Provider for the reasoning model: 'anthropic', 'openai', 'ollama', or an alias.
    If None, resolved from the model name (claude-* → anthropic, etc.)."""

    reasoning_api_key: str | None = None
    """API key for the reasoning model. If None, uses the default env var for the provider."""

    reasoning_base_url: str | None = None
    """Base URL for the reasoning model (for self-hosted or OpenAI-compatible endpoints)."""

    source_model: str | None = None
    """The model the prompt was originally written for (e.g. 'claude-sonnet', 'gpt-4o').
    When set, the reasoning model is told which model family this prompt came from so it
    can make smarter migration decisions — e.g. converting XML tags to markdown headers
    when moving from Claude to GPT-4o, or adjusting system-prompt structure for Llama."""

    temperature: float = 1.0
    """Temperature for the agent's prompt proposals."""

    eval_temperature: float = 0.0
    """Temperature for eval completions. Keep at 0 for reproducibility."""

    max_tokens: int = 1024
    """Max tokens for eval completions."""

    max_workers: int = 4
    """Max parallel threads for variant evaluation and duel execution.
    For Ollama, match this to OLLAMA_NUM_PARALLEL (default 1)."""

    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    """Strategy-specific parameters. For PDO: total_rounds, duels_per_round,
    samples_per_duel, initial_pool_size, thompson_alpha, mutation_frequency,
    num_top_to_mutate, max_pool_size."""

    # --- Target from verdict ---
    target_model: str | None = None
    """Label of the model whose score we're trying to match (from verdict)."""

    target_source: str | None = None
    """How the target was set: 'verdict_json', 'verdict_run', 'manual', or None."""


class PromptOptimizer:
    """Agentic prompt optimizer — diagnose failures and rewrite until target is met.

    Single command, full workflow: baseline → optimize → verify.

    Usage:
        result = (
            PromptOptimizer()
            .set_dataset(dataset)
            .add_provider("openai", "gpt-5.4-nano")
            .add_metric(RougeScore())
            .run("You are a helpful assistant.")
        )
        print(result.summary())  # shows before/after comparison
    """

    def __init__(self, config: OptimizerConfig | None = None):
        self.config = config or OptimizerConfig()
        self._dataset = None
        self._providers: list[dict[str, Any]] = []
        self._metrics: list[Any] = []

    def set_dataset(self, dataset: Any) -> PromptOptimizer:
        """Set the evaluation dataset (a verdict Dataset instance)."""
        self._dataset = dataset
        return self

    def add_provider(
        self,
        provider_name: str,
        model: str,
        *,
        label: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> PromptOptimizer:
        """Add a model to evaluate prompts against.

        Supports native providers (openai, anthropic, google, local) and
        aliases for OpenAI-compatible services: openrouter, together,
        fireworks, deepinfra, groq, lepton, perplexity.

        Examples:
            .add_provider("openai", "gpt-4o-mini")
            .add_provider("openrouter", "meta-llama/llama-3.1-8b-instruct")
            .add_provider("together", "meta-llama/Llama-3.1-8B-Instruct")
            .add_provider("local", "llama3.2:1b")  # Ollama
        """
        resolved = _resolve_provider(provider_name, model, api_key, base_url)
        resolved["label"] = label
        resolved.update(kwargs)
        self._providers.append(resolved)
        return self

    def add_metric(self, metric: Any) -> PromptOptimizer:
        """Add a scoring metric (verdict Metric instance)."""
        self._metrics.append(metric)
        return self

    def set_target_from_verdict(
        self,
        path: str | Path,
        metric: str | None = None,
    ) -> PromptOptimizer:
        """Set the score threshold from a verdict results JSON file.

        Parses the file, finds the best model's score, and uses it as
        the optimization target. This answers: "make my model as good as
        the best model in this benchmark."

        Args:
            path: Path to verdict's results.json.
            metric: Which metric to rank by. Defaults to the first metric
                    in the results file.
        """
        parsed = parse_verdict_results(path, metric=metric)
        self.config.score_threshold = parsed["best_score"]
        self.config.target_model = parsed["target_model"]
        self.config.target_source = "verdict_json"
        logger.info(
            f"Target set from verdict results: {parsed['best_score']:.4f} "
            f"({parsed['target_model']})"
        )
        return self

    def benchmark_and_set_target(
        self,
        prompt: str,
        providers: list[dict[str, Any]],
        metric: str | None = None,
    ) -> dict[str, Any]:
        """Run verdict with multiple models, then set the target from the best.

        This is the "run verdict first, then optimize" flow. Runs all
        providers against the dataset, finds the best-scoring model, and
        sets score_threshold to that score.

        The model to optimize should be the *weakest* provider, which is
        typically the one already added via add_provider().

        Args:
            prompt: The system prompt to benchmark.
            providers: All providers to benchmark (including the target model).
            metric: Which metric to rank by. Defaults to first metric.

        Returns:
            Dict with benchmark results (model scores, best model, etc.)
        """
        from aevyra_verdict import EvalRunner
        from aevyra_verdict.dataset import Conversation, Dataset, Message
        from aevyra_verdict.runner import RunConfig

        if self._dataset is None:
            raise ValueError("No dataset set. Call set_dataset() first.")
        if not self._metrics:
            raise ValueError("No metrics added. Call add_metric() first.")

        # Inject system prompt
        injected = []
        for convo in self._dataset.conversations:
            messages = list(convo.messages)
            if messages and messages[0].role == "system":
                messages[0] = Message(role="system", content=prompt)
            else:
                messages.insert(0, Message(role="system", content=prompt))
            injected.append(Conversation(
                messages=messages, ideal=convo.ideal, metadata=convo.metadata,
            ))

        ds = Dataset(conversations=injected)
        run_config = RunConfig(
            temperature=self.config.eval_temperature,
            max_tokens=self.config.max_tokens,
        )

        runner = EvalRunner(config=run_config)
        for p in providers:
            runner.add_provider(
                p["provider_name"], p["model"],
                label=p.get("label"), api_key=p.get("api_key"),
                base_url=p.get("base_url"),
            )
        for m in self._metrics:
            runner.add_metric(m)

        logger.info(f"Benchmarking {len(providers)} models...")
        results = runner.run(ds, show_progress=True)

        # Find best model
        metric_names = [m.name for m in self._metrics]
        rank_metric = metric or metric_names[0]

        best_label = None
        best_score = -1.0
        model_scores: dict[str, float] = {}

        for label, model_result in results.model_results.items():
            score = model_result.mean_score(rank_metric) or 0.0
            model_scores[label] = score
            if score > best_score:
                best_score = score
                best_label = label

        self.config.score_threshold = best_score
        self.config.target_model = best_label
        self.config.target_source = "verdict_run"

        logger.info(
            f"Benchmark complete. Target: {best_score:.4f} ({best_label})"
        )
        for label, score in sorted(model_scores.items(), key=lambda x: -x[1]):
            marker = " ← target" if label == best_label else ""
            logger.info(f"  {label:30s}  {score:.4f}{marker}")

        return {
            "model_scores": model_scores,
            "best_model": best_label,
            "best_score": best_score,
            "results": results,
        }

    def run(
        self,
        initial_prompt: str,
        on_iteration: Any | None = None,
        run_store: Any | None = None,
        resume_run: Any | None = None,
        callbacks: list[Any] | None = None,
        baseline_override: "EvalSnapshot | None" = None,
        branched_from: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Run the full optimization workflow: baseline → optimize → verify.

        Args:
            initial_prompt: The starting system prompt to improve.
            on_iteration: Optional callback(IterationRecord) for progress updates.
            run_store: Optional RunStore for checkpointing and persistence.
                       If provided, creates a new run (or resumes if resume_run is set).
            resume_run: Optional Run handle to resume from. Requires run_store.
            callbacks: Optional list of Callback objects (e.g. MLflowCallback()).
                       Each callback can implement on_run_start, on_iteration,
                       and/or on_run_end.
            baseline_override: If provided, skip the baseline eval and use this
                               EvalSnapshot instead. Used when branching from an
                               existing run to avoid re-running the baseline.
            branched_from: Optional dict with ``run_id`` and ``iteration`` keys
                           passed to RunStore.create_run() for lineage tracking.

        Returns:
            OptimizationResult with baseline/final scores and full history.
        """
        if self._dataset is None:
            raise ValueError("No dataset set. Call set_dataset() first.")
        if not self._providers:
            raise ValueError("No providers added. Call add_provider() first.")
        if not self._metrics:
            raise ValueError("No metrics added. Call add_metric() first.")

        # Check for Ollama and warn about parallel settings
        self._check_parallel_config()

        # Normalise callbacks list
        _callbacks = list(callbacks or [])

        # Fire on_run_start
        for cb in _callbacks:
            if hasattr(cb, "on_run_start"):
                try:
                    cb.on_run_start(self.config, initial_prompt)
                except Exception:
                    logger.exception(f"Callback {cb!r} raised in on_run_start")

        # --- Set up run persistence ---
        from aevyra_reflex.run_store import (
            CheckpointState,
            IterationState,
            Run,
        )

        run: Run | None = None
        checkpoint: CheckpointState | None = None

        if resume_run is not None:
            run = resume_run
            checkpoint = run.load_checkpoint()
            if checkpoint:
                logger.info(
                    f"Resuming run {run.run_id} from iteration "
                    f"{checkpoint.completed_iterations} "
                    f"(best score: {checkpoint.best_score:.4f})"
                )
        elif run_store is not None:
            from dataclasses import asdict
            config_dict = asdict(self.config)
            # Persist the eval model names so the dashboard and CLI can show them
            config_dict["_cli_models"] = [
                p.get("model", p.get("label", "")) for p in self._providers
            ]
            run = run_store.create_run(
                config=config_dict,
                dataset_path=getattr(self._dataset, '_source_path', 'unknown'),
                prompt_path="",
                initial_prompt=initial_prompt,
                branched_from=branched_from,
            )

        # --- Step 1: Baseline eval (skip if resuming or branching with a known baseline) ---
        run_tag = f"[run {run.run_id}]" if run else "[run ?]"
        strategy_tag = f"[{self.config.strategy}]"
        tag = f"{run_tag}{strategy_tag}"

        if baseline_override is not None:
            logger.info(f"{tag} Branch run — reusing parent baseline.")
            baseline = baseline_override
            if run:
                run.save_baseline({
                    "mean_score": baseline.mean_score,
                    "scores_by_metric": baseline.scores_by_metric,
                })
        elif checkpoint and checkpoint.baseline:
            logger.info(f"{tag} Resuming — using saved baseline.")
            baseline = EvalSnapshot(
                mean_score=checkpoint.baseline["mean_score"],
                scores_by_metric=checkpoint.baseline.get("scores_by_metric", {}),
            )
        else:
            logger.info(f"{tag} Running baseline evaluation...")
            baseline = self._run_eval(initial_prompt)
            logger.info(f"{tag} Baseline score: {baseline.mean_score:.4f}")
            if run:
                run.save_baseline({
                    "mean_score": baseline.mean_score,
                    "scores_by_metric": baseline.scores_by_metric,
                })

        # --- Step 2: Optimization loop ---
        llm = LLM(
            model=self.config.reasoning_model,
            max_tokens=4096,
            provider=self.config.reasoning_provider,
            api_key=self.config.reasoning_api_key,
            base_url=self.config.reasoning_base_url,
            source_model=self.config.source_model,
        )

        from aevyra_reflex.strategies import get_strategy
        strategy_cls = get_strategy(self.config.strategy)
        strategy = strategy_cls()

        # Wrap the on_iteration callback to save checkpoints
        def _checkpointing_callback(record):
            if on_iteration:
                on_iteration(record)
            for cb in _callbacks:
                if hasattr(cb, "on_iteration"):
                    try:
                        cb.on_iteration(record)
                    except Exception:
                        logger.exception(f"Callback {cb!r} raised in on_iteration")
            if run:
                # Save iteration
                run.save_iteration(IterationState(
                    iteration=record.iteration,
                    system_prompt=record.system_prompt,
                    score=record.score,
                    scores_by_metric=record.scores_by_metric,
                    reasoning=record.reasoning,
                    eval_tokens=getattr(record, "eval_tokens", 0),
                    reasoning_tokens=getattr(record, "reasoning_tokens", 0),
                    change_summary=getattr(record, "change_summary", ""),
                ))
                # Update checkpoint
                run.save_checkpoint(CheckpointState(
                    run_id=run.run_id,
                    initial_prompt=initial_prompt,
                    current_prompt=record.system_prompt,
                    completed_iterations=record.iteration,
                    best_prompt=record.system_prompt if record.score >= (
                        checkpoint.best_score if checkpoint else 0
                    ) else (checkpoint.best_prompt if checkpoint else initial_prompt),
                    best_score=max(
                        record.score,
                        checkpoint.best_score if checkpoint else 0,
                    ),
                    score_trajectory=(
                        (checkpoint.score_trajectory if checkpoint else [])
                        + [record.score]
                    ) if not hasattr(_checkpointing_callback, '_trajectory') else
                    _checkpointing_callback._trajectory + [record.score],
                    previous_reasoning=record.reasoning,
                    baseline={
                        "mean_score": baseline.mean_score,
                        "scores_by_metric": baseline.scores_by_metric,
                    },
                ))
                # Track trajectory across calls
                if not hasattr(_checkpointing_callback, '_trajectory'):
                    _checkpointing_callback._trajectory = checkpoint.score_trajectory[:] if checkpoint else []
                _checkpointing_callback._trajectory.append(record.score)

        result = strategy.run(
            initial_prompt=checkpoint.current_prompt if checkpoint else initial_prompt,
            dataset=self._dataset,
            providers=self._providers,
            metrics=self._metrics,
            agent=llm,
            config=self.config,
            on_iteration=_checkpointing_callback,
            **({"resume_state": checkpoint.strategy_state} if checkpoint and checkpoint.strategy_state else {}),
        )

        # --- Step 3: Final verification eval ---
        logger.info(f"{tag} Running final verification...")
        final = self._run_eval(result.best_prompt)
        logger.info(f"{tag} Final score: {final.mean_score:.4f}  (baseline: {baseline.mean_score:.4f}  delta: {final.mean_score - baseline.mean_score:+.4f})")

        # Attach baseline, final, and strategy info to result
        result.baseline = baseline
        result.baseline.system_prompt = initial_prompt
        result.final = final
        result.final.system_prompt = result.best_prompt
        result.best_score = final.mean_score
        if not result.strategy_name:
            result.strategy_name = self.config.strategy

        # Accumulate token counts across all phases
        result.total_eval_tokens = (
            baseline.total_tokens
            + sum(getattr(r, "eval_tokens", 0) for r in result.iterations)
            + final.total_tokens
        )
        result.total_reasoning_tokens = llm.tokens_used

        # Save final result
        if run:
            run.save_result(result.to_dict())
            logger.info(f"Run {run.run_id} saved to {run.run_dir}")

        # Fire on_run_end
        for cb in _callbacks:
            if hasattr(cb, "on_run_end"):
                try:
                    cb.on_run_end(result)
                except Exception:
                    logger.exception(f"Callback {cb!r} raised in on_run_end")

        return result

    def _check_parallel_config(self) -> None:
        """Detect Ollama usage and warn if parallel inference isn't configured."""
        import os

        is_ollama = any(
            p.get("provider_name") == "local"
            or "ollama" in (p.get("base_url") or "").lower()
            or "localhost:11434" in (p.get("base_url") or "")
            for p in self._providers
        )

        if not is_ollama:
            return

        ollama_parallel = os.environ.get("OLLAMA_NUM_PARALLEL", "")
        strategy = self.config.strategy
        needs_parallel = strategy in ("structural", "pdo", "auto")

        if needs_parallel and not ollama_parallel:
            logger.warning(
                "Ollama detected with %s strategy, but OLLAMA_NUM_PARALLEL "
                "is not set. Ollama defaults to processing 1 request at a "
                "time, so parallel variant evaluation won't speed things up. "
                "To enable parallel inference, restart Ollama with:\n"
                "  OLLAMA_NUM_PARALLEL=4 ollama serve\n"
                "Then set max_workers to match:\n"
                "  aevyra-reflex optimize ... --max-workers 4",
                self.config.strategy,
            )
            # Fall back to sequential to avoid pointless thread overhead
            self.config.max_workers = 1
        elif needs_parallel and ollama_parallel:
            num = int(ollama_parallel)
            if self.config.max_workers > num:
                logger.info(
                    f"Capping max_workers to OLLAMA_NUM_PARALLEL={num} "
                    f"(was {self.config.max_workers})"
                )
                self.config.max_workers = num
            else:
                logger.info(
                    f"Ollama parallel inference enabled "
                    f"(OLLAMA_NUM_PARALLEL={num}, max_workers={self.config.max_workers})"
                )

    def _run_eval(self, prompt: str) -> EvalSnapshot:
        """Run a full verdict eval with the given system prompt."""
        from aevyra_verdict import EvalRunner
        from aevyra_verdict.dataset import Conversation, Dataset, Message
        from aevyra_verdict.runner import RunConfig

        # Inject system prompt
        injected = []
        for convo in self._dataset.conversations:
            messages = list(convo.messages)
            if messages and messages[0].role == "system":
                messages[0] = Message(role="system", content=prompt)
            else:
                messages.insert(0, Message(role="system", content=prompt))
            injected.append(Conversation(
                messages=messages, ideal=convo.ideal, metadata=convo.metadata,
            ))

        ds = Dataset(conversations=injected)
        run_config = RunConfig(
            temperature=self.config.eval_temperature,
            max_tokens=self.config.max_tokens,
        )

        runner = EvalRunner(config=run_config)
        for p in self._providers:
            runner.add_provider(
                p["provider_name"], p["model"],
                label=p.get("label"), api_key=p.get("api_key"),
                base_url=p.get("base_url"),
            )
        for m in self._metrics:
            runner.add_metric(m)

        results = runner.run(ds, show_progress=True)

        # Aggregate scores and capture per-sample data
        all_scores: list[float] = []
        scores_by_metric: dict[str, list[float]] = {}
        sample_snapshots: list[SampleSnapshot] = []

        for model_label, model_result in results.model_results.items():
            for idx in range(model_result.num_samples):
                sample_scores = model_result.scores[idx]
                # Mean score for this sample across all metrics
                sample_vals = []
                for metric_name, score_result in sample_scores.items():
                    val = score_result.score if hasattr(score_result, "score") else float(score_result)
                    all_scores.append(val)
                    sample_vals.append(val)
                    scores_by_metric.setdefault(metric_name, []).append(val)
                    # Expand per-dimension sub_scores (e.g. multi-dimensional LLMJudge)
                    for sub_name, sub_val in (getattr(score_result, "sub_scores", None) or {}).items():
                        scores_by_metric.setdefault(sub_name, []).append(sub_val)

                # Capture sample snapshot for before/after comparison
                convo = ds.conversations[idx]
                completion = model_result.completions[idx]
                sample_snapshots.append(SampleSnapshot(
                    input=convo.last_user_message or "(no user message)",
                    response=completion.text if completion else "(no response)",
                    ideal=convo.ideal or "",
                    score=sum(sample_vals) / len(sample_vals) if sample_vals else 0.0,
                ))

        mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
        per_metric = {k: sum(v) / len(v) for k, v in scores_by_metric.items()}
        total_tokens = sum(
            mr.total_tokens() for mr in results.model_results.values()
        )

        return EvalSnapshot(
            mean_score=mean,
            scores_by_metric=per_metric,
            samples=sample_snapshots,
            total_tokens=total_tokens,
        )
