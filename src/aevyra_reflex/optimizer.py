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


class _EarlyStop(Exception):
    """Raised inside the iteration callback to interrupt optimization when the
    validation score has not improved for ``early_stopping_patience`` iterations."""


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
# When a user writes `-m together/meta-llama/llama-3.1-8b-instruct`, we
# resolve "together" to the openai provider with the right base_url and key.
#
# Note: "openrouter" is intentionally NOT listed here — verdict has a native
# OpenRouterProvider that reads OPENROUTER_API_KEY directly. Aliasing it to
# the openai provider caused confusing "OPENAI_API_KEY not set" errors.
PROVIDER_ALIASES: dict[str, dict[str, str]] = {
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
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GOOGLE_API_KEY",
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

    eval_runs: int = 1
    """Number of times to run each eval (baseline and final verification).
    When > 1, scores are averaged across runs and a standard deviation is
    reported alongside the mean. Use 3–5 for noisy tasks or small datasets
    where a single eval pass may not be representative.

    Note: eval_runs multiplies the cost of baseline and final evals. With
    eval_runs=3 you run 3× as many completions for those two checkpoints.
    Optimization iterations are unaffected (always 1 run each)."""

    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    """Strategy-specific parameters. For PDO: total_rounds, duels_per_round,
    samples_per_duel, initial_pool_size, thompson_alpha, mutation_frequency,
    num_top_to_mutate, max_pool_size."""

    train_ratio: float = 0.8
    """Fraction of the dataset used during optimization. The remaining examples
    are held out as a test set and used exclusively for baseline and final eval.
    Set to 1.0 to use the full dataset for everything (no split).
    Default: 0.8 (80% train / 20% test)."""

    val_ratio: float = 0.1
    """Fraction of the total dataset reserved as a validation set, carved out of
    the training portion. Used to detect overfitting mid-run: if the val score
    plateaus while train scores keep climbing, the prompt is fitting the training
    examples specifically. Set to 0.0 to disable validation splitting.

    With train_ratio=0.8 and val_ratio=0.1 (the defaults), the actual split is:
      70% train  /  10% val  /  20% test

    Requires train_ratio - val_ratio >= 0.1 (at least 10% left for training)."""

    early_stopping_patience: int = 3
    """Stop optimization early if the validation score has not improved for this
    many consecutive iterations. Only takes effect when val_ratio > 0.
    Set to 0 to disable. When early stopping triggers, the prompt with
    the best validation score is returned (not the most recent prompt).

    Default: 3. Recommended values: 2–4 iterations. Smaller values stop sooner
    (risk of stopping too early on noise); larger values are more conservative."""

    batch_size: int = 0
    """Mini-batch size per optimization iteration. 0 (default) = full training
    set. When > 0, each iteration samples this many examples at random from the
    training data before running the eval. Speeds up per-iteration cost on large
    datasets; the stochasticity can also help escape local optima. Each iteration
    draws a fresh sample so the optimizer sees variety across the run.

    Note: baseline and final verification evals always use the full test set —
    batch_size only affects the per-iteration training evals used by the
    optimization strategy."""

    batch_seed: int = 42
    """Base seed for reproducible mini-batch sampling. Iteration i uses
    ``batch_seed + i``, so every iteration's batch is different but the
    full run is deterministic and repeatable."""

    full_eval_steps: int = 0
    """When using mini-batch mode (``batch_size > 0``), run a full training-set
    eval every this many iterations to get an accurate checkpoint score.
    0 (default) = never — use mini-batch scores throughout.

    Example: with ``batch_size=32`` and ``full_eval_steps=5``, iterations
    1–4 score on a 32-example batch; iteration 5 scores on the full training
    set; iterations 6–9 use batches again; iteration 10 is a full eval; and
    so on. Full-eval iterations are marked in the trajectory and dashboard.

    Has no effect when ``batch_size=0`` (already using the full set every
    iteration)."""

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

    @staticmethod
    def _split_dataset(dataset: Any, train_ratio: float, seed: int = 42) -> tuple[Any, Any]:
        """Split a verdict Dataset into train and test subsets.

        Uses a deterministic shuffle so the same dataset always produces the
        same split. The test set is never seen by the optimization strategy —
        it is used only for baseline and final verification scores.

        Args:
            dataset: A verdict Dataset instance with a .conversations list.
            train_ratio: Fraction of examples to use for training (0 < ratio < 1).
            seed: Random seed for reproducibility (default 42).

        Returns:
            (train_dataset, test_dataset) tuple.
        """
        import random
        from aevyra_verdict.dataset import Dataset

        convos = list(dataset.conversations)
        n = len(convos)
        n_train = max(1, round(n * train_ratio))
        n_test = max(1, n - n_train)

        # Cap n_train so there's always at least 1 test example
        if n_train + n_test > n:
            n_train = n - n_test

        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)

        train_indices = set(indices[:n_train])
        train_convos = [convos[i] for i in range(n) if i in train_indices]
        test_convos = [convos[i] for i in range(n) if i not in train_indices]

        return Dataset(conversations=train_convos), Dataset(conversations=test_convos)

    @staticmethod
    def _split_dataset_3way(
        dataset: Any,
        train_ratio: float,
        val_ratio: float,
        seed: int = 42,
    ) -> tuple[Any, Any | None, Any]:
        """Split a verdict Dataset into (train, val, test) subsets.

        The validation set is carved out of what would otherwise be the training
        portion, so test size is unaffected by adding a val split.

        Args:
            dataset: A verdict Dataset instance with a .conversations list.
            train_ratio: Fraction NOT reserved for test (train + val combined).
                         Must be in (0, 1).
            val_ratio: Fraction of total dataset reserved for validation.
                       Must be < train_ratio. Set to 0.0 for a 2-way split.
            seed: Random seed for reproducibility (default 42).

        Returns:
            (train_dataset, val_dataset, test_dataset) where val_dataset is
            None when val_ratio == 0.0.
        """
        import random
        from aevyra_verdict.dataset import Dataset

        convos = list(dataset.conversations)
        n = len(convos)

        # Need at least 3 examples for a 3-way split (1 per partition).
        # Fall back to 2-way (no val) if the dataset is too small.
        if n < 3:
            n_test = max(1, n - round(n * train_ratio)) if n > 1 else 0
            n_train = n - n_test
            rng = random.Random(seed)
            indices = list(range(n))
            rng.shuffle(indices)
            train_convos = [convos[indices[i]] for i in range(n_train)]
            test_convos = [convos[indices[i]] for i in range(n_train, n)]
            return Dataset(conversations=train_convos), None, Dataset(conversations=test_convos)

        n_test = max(1, n - round(n * train_ratio))
        n_val = max(1, round(n * val_ratio)) if val_ratio > 0.0 else 0
        n_train = max(1, n - n_test - n_val)

        # Guard against rounding pushing total over n
        while n_train + n_val + n_test > n and n_train > 1:
            n_train -= 1

        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)

        train_convos = [convos[indices[i]] for i in range(max(0, n_train))]
        val_convos = [convos[indices[i]] for i in range(max(0, n_train), max(0, n_train + n_val))]
        test_convos = [convos[indices[i]] for i in range(max(0, n_train + n_val), n)]

        val_ds = Dataset(conversations=val_convos) if val_convos else None
        return Dataset(conversations=train_convos), val_ds, Dataset(conversations=test_convos)

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
            max_workers=self.config.max_workers,
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
        prior_duration_seconds: float = 0.0,
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
            prior_duration_seconds: Elapsed time already spent on this run before
                                    this session started (e.g. from the parent run
                                    up to the branching point). Added to every
                                    elapsed/duration measurement so branch runs
                                    show cumulative wall time including parent work.

        Returns:
            OptimizationResult with baseline/final scores and full history.
        """
        if self._dataset is None:
            raise ValueError("No dataset set. Call set_dataset() first.")
        if not self._providers:
            raise ValueError("No providers added. Call add_provider() first.")
        if not self._metrics:
            raise ValueError("No metrics added. Call add_metric() first.")

        # Fail fast for label-free datasets paired with reference-based metrics.
        if not self._dataset.has_ideals():
            needs_ideal = [
                m.name for m in self._metrics if getattr(m, "requires_ideal", False)
            ]
            if needs_ideal:
                raise ValueError(
                    f"Dataset has no ideal answers (label-free). "
                    f"The following metrics require reference answers: {needs_ideal}. "
                    f"Use LLMJudge for label-free evaluation: "
                    f"optimizer.add_metric(LLMJudge(judge_provider=...))."
                )

        # Check for Ollama and warn about parallel settings
        self._check_parallel_config()

        # --- Dataset split ---
        train_ratio = self.config.train_ratio
        val_ratio = self.config.val_ratio

        if val_ratio > 0.0 and 0.0 < train_ratio < 1.0:
            if val_ratio >= train_ratio:
                raise ValueError(
                    f"val_ratio ({val_ratio}) must be less than train_ratio ({train_ratio}). "
                    f"At least 10%% of data must remain for training."
                )
            train_dataset, val_dataset, test_dataset = self._split_dataset_3way(
                self._dataset, train_ratio=train_ratio, val_ratio=val_ratio
            )
            n_train = len(train_dataset.conversations)
            n_val = len(val_dataset.conversations) if val_dataset else 0
            n_test = len(test_dataset.conversations)
            logger.info(
                f"Dataset split (3-way): {n_train} train / {n_val} val / {n_test} test "
                f"— val set used to detect overfitting; test set for baseline and final eval"
            )
        elif 0.0 < train_ratio < 1.0:
            train_dataset, test_dataset = self._split_dataset(self._dataset, train_ratio)
            val_dataset = None
            n_train = len(train_dataset.conversations)
            n_val = 0
            n_test = len(test_dataset.conversations)
            logger.info(
                f"Dataset split: {n_train} train / {n_test} test "
                f"({train_ratio:.0%} / {1 - train_ratio:.0%}) — "
                f"baseline and final scores are on the held-out test set"
            )
        else:
            train_dataset = self._dataset
            test_dataset = self._dataset
            val_dataset = None
            n_train = len(self._dataset.conversations)
            n_val = 0
            n_test = n_train

        if self.config.batch_size > 0:
            effective_batch = min(self.config.batch_size, n_train)
            logger.info(
                f"Mini-batch mode: {effective_batch} examples/iter "
                f"(from {n_train} training examples) — "
                f"baseline and final evals use full test set"
            )

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

        import time as _time
        from datetime import datetime, timezone
        _run_start_monotonic = _time.monotonic()
        _run_started_at = datetime.now(timezone.utc).isoformat()

        run: Run | None = None
        checkpoint: CheckpointState | None = None
        # Cumulative wall time from prior sessions / parent branch work.
        # For branch runs this is seeded from the parent iteration's elapsed_seconds.
        # For resumed runs the checkpoint value takes precedence (it's always >= this).
        _prior_duration_seconds: float = prior_duration_seconds

        if resume_run is not None:
            run = resume_run
            checkpoint = run.load_checkpoint()
            if checkpoint:
                _prior_duration_seconds = checkpoint.accumulated_duration_seconds
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
                f"{p['provider_name']}/{p.get('model', p.get('label', ''))}"
                for p in self._providers
            ]
            run = run_store.create_run(
                config=config_dict,
                dataset_path=getattr(self._dataset, '_source_path', 'unknown'),
                prompt_path="",
                initial_prompt=initial_prompt,
                branched_from=branched_from,
            )

        # Mark run as actively running; cleared by mark_done() on normal exit,
        # or left in place on crash (dashboard will then correctly show 'interrupted')
        if run:
            run.mark_running()

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
                    "total_tokens": baseline.total_tokens,
                })
        elif checkpoint and checkpoint.baseline:
            logger.info(f"{tag} Resuming — using saved baseline.")
            baseline = EvalSnapshot(
                mean_score=checkpoint.baseline["mean_score"],
                scores_by_metric=checkpoint.baseline.get("scores_by_metric", {}),
                total_tokens=checkpoint.baseline.get("total_tokens", 0),
            )
        else:
            eval_runs = self.config.eval_runs
            eval_label = f" ({eval_runs} runs)" if eval_runs > 1 else ""
            n_test_samples = len(test_dataset.conversations) if test_dataset else "?"
            logger.info(f"{tag} Running baseline eval on TEST SET{eval_label} ({n_test_samples} held-out samples)...")
            baseline = self._run_eval(initial_prompt, dataset=test_dataset)
            std_label = f" ± {baseline.std_score:.4f}" if baseline.std_score > 0 else ""
            logger.info(f"{tag} Baseline TEST SET score: {baseline.mean_score:.4f}{std_label}")
            if run:
                run.save_baseline({
                    "mean_score": baseline.mean_score,
                    "scores_by_metric": baseline.scores_by_metric,
                    "total_tokens": baseline.total_tokens,
                })

        # Write an initial checkpoint immediately after the baseline so that
        # a crash during the very first iteration (e.g. mid-variant-eval) still
        # produces a resumable run.  Without this, checkpoint.json is never
        # written until the first iteration completes and --resume can't find it.
        if run and not checkpoint:
            run.save_checkpoint(CheckpointState(
                run_id=run.run_id,
                initial_prompt=initial_prompt,
                current_prompt=initial_prompt,
                completed_iterations=0,
                best_prompt=initial_prompt,
                best_score=0.0,
                score_trajectory=[],
                previous_reasoning="",
                strategy_state={},
                baseline={
                    "mean_score": baseline.mean_score,
                    "scores_by_metric": baseline.scores_by_metric,
                    "total_tokens": baseline.total_tokens,
                },
                accumulated_duration_seconds=_prior_duration_seconds,
            ))

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

        # Restore per-iteration state from saved files when resuming.
        # This reconstructs val_history (for early stopping), token totals (for accurate
        # reporting), and best-val tracking — all of which start from zero otherwise.
        _resumed_val_history: list[float] = []
        _resumed_eval_tokens: int = 0
        _resumed_reasoning_tokens: int = 0
        # Full val trajectory across ALL phases — never cleared unlike val_history
        # which resets at each phase boundary for per-phase early stopping.
        _val_trajectory_all: list[float] = []
        if checkpoint and run:
            for it in run.load_iterations():
                if it.val_score is not None:
                    _resumed_val_history.append(it.val_score)
                    _val_trajectory_all.append(it.val_score)
                _resumed_eval_tokens += it.eval_tokens
                _resumed_reasoning_tokens += it.reasoning_tokens
            _baseline_tok = checkpoint.baseline.get("total_tokens", 0) if checkpoint and checkpoint.baseline else 0
            if _baseline_tok or _resumed_eval_tokens or _resumed_reasoning_tokens:
                def _fmtk(n: int) -> str:
                    return f"{n / 1000:.1f}K" if n >= 1000 else str(n)
                logger.info(
                    f"Tokens so far: eval={_fmtk(_baseline_tok + _resumed_eval_tokens)}  "
                    f"reasoning={_fmtk(_resumed_reasoning_tokens)}"
                )

        # Restore best_val state from checkpoint (saved each iteration)
        _resumed_best_val_prompt: str = checkpoint.best_val_prompt if (checkpoint and checkpoint.best_val_prompt) else (checkpoint.current_prompt if checkpoint else initial_prompt)
        _resumed_best_val_score: float = checkpoint.best_val_score if checkpoint else -1.0
        _resumed_best_val_iter: int = checkpoint.best_val_iter if checkpoint else 0
        # Reconstruct the train score at the best-val iteration from saved iteration files
        _resumed_best_val_train_score: float = -1.0
        if checkpoint and run and _resumed_best_val_iter > 0:
            for it in run.load_iterations():
                if it.iteration == _resumed_best_val_iter:
                    _resumed_best_val_train_score = it.score
                    break

        # Shared mutable state for val tracking and early stopping
        _es: dict[str, Any] = {
            "val_history": _resumed_val_history,  # val score per iteration (restored on resume)
            "iterations": [],            # IterationRecord objects collected
            "best_train_prompt": checkpoint.current_prompt if checkpoint else initial_prompt,
            "best_train_score": checkpoint.best_score if checkpoint else -1.0,
            "best_val_prompt": _resumed_best_val_prompt,
            "best_val_score": _resumed_best_val_score,
            "best_val_train_score": _resumed_best_val_train_score,  # train score at best-val iter
            "best_val_iter": _resumed_best_val_iter,  # global iteration number of best val
            "trajectory": list(checkpoint.score_trajectory) if checkpoint else [],
            "strategy_state": dict(checkpoint.strategy_state) if checkpoint and checkpoint.strategy_state else {},
            "total_reasoning_tokens": _resumed_reasoning_tokens,  # seeded from saved iterations on resume
            "total_eval_tokens": baseline.total_tokens + _resumed_eval_tokens,  # baseline + prior iterations
        }

        def _fmt_k(n: int) -> str:
            """Format token count as e.g. 1.2K."""
            if n < 1000:
                return str(n)
            return f"{n / 1000:.1f}K"

        # Wrap the on_iteration callback to save checkpoints and run val eval
        def _checkpointing_callback(record):
            _es["iterations"].append(record)

            # Track best train prompt
            if record.score >= _es["best_train_score"]:
                _es["best_train_score"] = record.score
                _es["best_train_prompt"] = record.system_prompt

            if on_iteration:
                on_iteration(record)
            for cb in _callbacks:
                if hasattr(cb, "on_iteration"):
                    try:
                        cb.on_iteration(record)
                    except Exception:
                        logger.exception(f"Callback {cb!r} raised in on_iteration")
            if run:
                _es["trajectory"].append(record.score)
                # Update checkpoint (iteration saved after val eval below)
                run.save_checkpoint(CheckpointState(
                    run_id=run.run_id,
                    initial_prompt=initial_prompt,
                    current_prompt=record.system_prompt,
                    completed_iterations=record.iteration,
                    best_prompt=_es["best_train_prompt"],
                    best_score=_es["best_train_score"],
                    score_trajectory=list(_es["trajectory"]),
                    previous_reasoning=record.reasoning,
                    strategy_state=dict(_es["strategy_state"]),
                    baseline={
                        "mean_score": baseline.mean_score,
                        "scores_by_metric": baseline.scores_by_metric,
                        "total_tokens": baseline.total_tokens,
                    },
                    best_val_prompt=_es["best_val_prompt"] if val_dataset else None,
                    best_val_score=_es["best_val_score"],
                    best_val_iter=_es["best_val_iter"],
                    accumulated_duration_seconds=_prior_duration_seconds + _time.monotonic() - _run_start_monotonic,
                ))

            # --- Validation eval + early stopping ---
            if val_dataset is not None:
                val_snap = self._run_eval_single(record.system_prompt, dataset=val_dataset)
                val_score = val_snap.mean_score
                _es["val_history"].append(val_score)
                _val_trajectory_all.append(val_score)  # never cleared — for final summary
                record.val_score = val_score

                # Track best val prompt (the one least prone to overfitting)
                if val_score >= _es["best_val_score"]:
                    _es["best_val_score"] = val_score
                    _es["best_val_train_score"] = record.score
                    _es["best_val_prompt"] = record.system_prompt
                    _es["best_val_iter"] = record.iteration

            # Save iteration after val eval so val_score is included
            if run:
                _iter_elapsed = _prior_duration_seconds + _time.monotonic() - _run_start_monotonic
                run.save_iteration(IterationState(
                    iteration=record.iteration,
                    system_prompt=record.system_prompt,
                    score=record.score,
                    scores_by_metric=record.scores_by_metric,
                    reasoning=record.reasoning,
                    eval_tokens=getattr(record, "eval_tokens", 0),
                    reasoning_tokens=getattr(record, "reasoning_tokens", 0),
                    change_summary=getattr(record, "change_summary", ""),
                    val_score=getattr(record, "val_score", None),
                    is_full_eval=getattr(record, "is_full_eval", False),
                    elapsed_seconds=round(_iter_elapsed, 1),
                ))

                r_tok = getattr(record, "reasoning_tokens", 0)
                e_tok = getattr(record, "eval_tokens", 0)
                _es["total_reasoning_tokens"] += r_tok
                _es["total_eval_tokens"] += e_tok
                tok_parts = []
                if e_tok:
                    tok_parts.append(f"eval={_fmt_k(e_tok)} (total {_fmt_k(_es['total_eval_tokens'])})")
                if r_tok:
                    tok_parts.append(f"reason={_fmt_k(r_tok)} (total {_fmt_k(_es['total_reasoning_tokens'])})")
                tok_str = ("  " + "  ".join(tok_parts)) if tok_parts else ""
                _el_m, _el_s = divmod(int(_iter_elapsed), 60)
                _el_h, _el_m = divmod(_el_m, 60)
                _el_label = (
                    f"{_el_h}h {_el_m}m {_el_s}s" if _el_h
                    else f"{_el_m}m {_el_s}s" if _el_m
                    else f"{_el_s}s"
                )
                if val_dataset is not None:
                    logger.info(
                        f"{tag} Iteration {record.iteration}: "
                        f"train={record.score:.4f}  val={val_score:.4f}{tok_str}  [{_el_label}]"
                    )
                else:
                    logger.info(
                        f"{tag} Iteration {record.iteration}: "
                        f"train={record.score:.4f}{tok_str}  [{_el_label}]"
                    )

                # Check early stopping condition
                patience = self.config.early_stopping_patience
                if patience > 0 and len(_es["val_history"]) >= patience:
                    best_val_overall = _es["best_val_score"]
                    best_val_iter = _es["best_val_iter"]
                    # Count how many consecutive iterations have not beaten the best val.
                    # reversed_index=0 means the current iteration IS the best → 0 iters since best.
                    iters_since_best = next(
                        i for i, v in enumerate(reversed(_es["val_history"]))
                        if v == best_val_overall
                    )
                    if iters_since_best >= patience:
                        logger.info(
                            f"{tag} Early stopping triggered: val score has not improved "
                            f"for {patience} consecutive iteration(s) "
                            f"(best val {best_val_overall:.4f} at iteration {best_val_iter}). "
                            f"Using prompt from iteration {best_val_iter}."
                        )
                        raise _EarlyStop()

        def _update_strategy_state(state: dict) -> None:
            """Called by strategies to persist phase/iteration state into checkpoints.

            Flushes immediately to disk so that bootstrap progress and other
            between-iteration state (phase advances, sub-strategy state) survive
            a crash before the next full iteration checkpoint is written.

            Special key: ``_reset_val_history`` — if True, clears the val history
            and best-val tracking so early stopping starts fresh for the new phase.
            """
            if state.pop("_reset_val_history", False):
                _es["val_history"].clear()
                _es["best_val_score"] = -1.0
                _es["best_val_iter"] = 0
                _es["best_val_prompt"] = _es["best_train_prompt"]
                _es["best_val_train_score"] = -1.0
                logger.debug("[early stopping] val history reset for new phase")
            _es["strategy_state"].update(state)
            if run:
                # Use the completed_iterations from the last real iteration so
                # the resume banner stays accurate.
                _completed = (
                    _es["iterations"][-1].iteration if _es["iterations"]
                    else (checkpoint.completed_iterations if checkpoint else 0)
                )
                run.save_checkpoint(CheckpointState(
                    run_id=run.run_id,
                    initial_prompt=initial_prompt,
                    current_prompt=_es["best_train_prompt"],
                    completed_iterations=_completed,
                    best_prompt=_es["best_train_prompt"],
                    best_score=_es["best_train_score"],
                    score_trajectory=list(_es["trajectory"]),
                    previous_reasoning=(_es["iterations"][-1].reasoning if _es["iterations"] else ""),
                    strategy_state=dict(_es["strategy_state"]),
                    baseline={
                        "mean_score": baseline.mean_score,
                        "scores_by_metric": baseline.scores_by_metric,
                        "total_tokens": baseline.total_tokens,
                    },
                    best_val_prompt=_es["best_val_prompt"] if val_dataset else None,
                    best_val_score=_es["best_val_score"],
                    best_val_iter=_es["best_val_iter"],
                    accumulated_duration_seconds=_prior_duration_seconds + _time.monotonic() - _run_start_monotonic,
                ))

        _early_stopped = False
        try:
            result = strategy.run(
                initial_prompt=checkpoint.current_prompt if checkpoint else initial_prompt,
                dataset=train_dataset,
                providers=self._providers,
                metrics=self._metrics,
                agent=llm,
                config=self.config,
                on_iteration=_checkpointing_callback,
                update_strategy_state=_update_strategy_state,
                **({"resume_state": checkpoint.strategy_state} if checkpoint and checkpoint.strategy_state else {}),
            )
        except _EarlyStop:
            _early_stopped = True
            # Build a partial result from what we've collected so far
            best_prompt = _es["best_val_prompt"] if val_dataset else _es["best_train_prompt"]
            best_score = _es["best_val_score"] if val_dataset else _es["best_train_score"]
            from aevyra_reflex.result import OptimizationResult as _OR
            result = _OR(
                best_prompt=best_prompt,
                best_score=best_score,
                iterations=list(_es["iterations"]),
                converged=False,
            )

        # --- Step 3: Final verification eval (on held-out test set) ---
        eval_runs = self.config.eval_runs
        eval_label = f" ({eval_runs} runs)" if eval_runs > 1 else ""
        n_test_samples = len(test_dataset.conversations) if test_dataset else "?"
        # Select the prompt to test: when a val set was used, prefer the
        # prompt with the best val score (least likely to be overfit to train).
        # Fall back to the best-train prompt if val tracking was never updated.
        if val_dataset is not None and _es["best_val_score"] > -1.0:
            prompt_for_test = _es["best_val_prompt"]
            _bv_iter = _es["best_val_iter"]
            _bv_train = _es["best_val_train_score"]
            _bv_val = _es["best_val_score"]
            _bv_label = (
                f"iter {_bv_iter}: train={_bv_train:.4f}, val={_bv_val:.4f}"
                if _bv_train > -1.0 else f"iter {_bv_iter}: val={_bv_val:.4f}"
            )
            logger.info(
                f"{tag} Running TEST SET eval{eval_label} on best-val prompt "
                f"({_bv_label}) — {n_test_samples} held-out samples..."
            )
        else:
            prompt_for_test = result.best_prompt
            logger.info(f"{tag} Running TEST SET eval{eval_label} ({n_test_samples} held-out samples)...")
        final = self._run_eval(prompt_for_test, dataset=test_dataset)
        logger.info(
            f"{tag} TEST SET score: {final.mean_score:.4f}  "
            f"(baseline: {baseline.mean_score:.4f}  delta: {final.mean_score - baseline.mean_score:+.4f})"
        )

        # Attach baseline, final, and strategy info to result
        result.baseline = baseline
        result.baseline.system_prompt = initial_prompt
        result.final = final
        result.final.system_prompt = result.best_prompt
        result.best_score = final.mean_score
        # Convergence is based on the held-out test score, not training score.
        # A prompt that hit the threshold on train but not test has not converged.
        result.converged = final.mean_score >= self.config.score_threshold
        if not result.strategy_name:
            result.strategy_name = self.config.strategy

        # Record dataset split sizes (0 means no split was used)
        if 0.0 < self.config.train_ratio < 1.0:
            result.train_size = n_train
            result.test_size = n_test

        # Record validation split info
        if n_val > 0:
            result.val_size = n_val
            result.val_trajectory = _val_trajectory_all
        result.early_stopped = _early_stopped

        # Record mini-batch size when active
        if self.config.batch_size > 0:
            result.batch_size = self.config.batch_size

        # Statistical significance: paired test on per-sample scores
        p_val, is_sig = self._compute_significance(baseline, final)
        result.p_value = p_val
        result.is_significant = is_sig
        if p_val is not None:
            sig_label = "✓ significant" if is_sig else "✗ not significant"
            logger.info(f"{tag} Significance: p={p_val:.4f} ({sig_label} at α=0.05)")

        # Accumulate token counts across all phases.
        # _resumed_eval_tokens / _resumed_reasoning_tokens carry pre-interruption totals
        # for resumed runs (0 for fresh runs), ensuring accuracy across multiple resumes.
        result.total_eval_tokens = (
            baseline.total_tokens
            + _resumed_eval_tokens
            + sum(getattr(r, "eval_tokens", 0) for r in result.iterations)
            + final.total_tokens
        )
        result.total_reasoning_tokens = _resumed_reasoning_tokens + llm.tokens_used

        # Save final result
        _total_seconds = _prior_duration_seconds + _time.monotonic() - _run_start_monotonic
        result.duration_seconds = round(_total_seconds, 2)
        if run:
            result_dict = result.to_dict()
            result_dict["duration_seconds"] = round(_total_seconds, 2)
            result_dict["started_at"] = _run_started_at
            run.save_result(result_dict)
            _dur_m, _dur_s = divmod(int(_total_seconds), 60)
            _dur_h, _dur_m = divmod(_dur_m, 60)
            _dur_label = (
                f"{_dur_h}h {_dur_m}m {_dur_s}s" if _dur_h
                else f"{_dur_m}m {_dur_s}s" if _dur_m
                else f"{_dur_s}s"
            )
            logger.info(f"Run {run.run_id} saved to {run.run_dir}  (total duration: {_dur_label})")

        # Fire on_run_end
        for cb in _callbacks:
            if hasattr(cb, "on_run_end"):
                try:
                    cb.on_run_end(result)
                except Exception:
                    logger.exception(f"Callback {cb!r} raised in on_run_end")

        # Clear running sentinel so dashboard shows 'completed' not 'running'
        if run:
            run.mark_done()

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
                "To enable parallel inference, set the env var for both "
                "Ollama and reflex:\n"
                "  OLLAMA_NUM_PARALLEL=4 ollama serve &\n"
                "  OLLAMA_NUM_PARALLEL=4 aevyra-reflex optimize ... --max-workers 4",
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

    def _run_eval(self, prompt: str, dataset: Any = None, n_runs: int | None = None) -> EvalSnapshot:
        """Run verdict eval with the given system prompt.

        Args:
            prompt: The system prompt to evaluate.
            dataset: Dataset to evaluate on. Defaults to self._dataset.
                     Pass the test split here to get honest held-out scores.
            n_runs: Number of eval passes to average. Defaults to config.eval_runs.
                    When > 1, returns mean ± std across runs.
        """
        effective_runs = n_runs if n_runs is not None else self.config.eval_runs
        if effective_runs > 1:
            return self._run_eval_multi(prompt, dataset=dataset, n_runs=effective_runs)
        return self._run_eval_single(prompt, dataset=dataset)

    def _run_eval_single(self, prompt: str, dataset: Any = None) -> EvalSnapshot:
        """Run a single verdict eval pass and return an EvalSnapshot."""
        from aevyra_verdict import EvalRunner
        from aevyra_verdict.dataset import Conversation, Dataset, Message
        from aevyra_verdict.runner import RunConfig

        eval_dataset = dataset if dataset is not None else self._dataset

        # Inject system prompt
        injected = []
        for convo in eval_dataset.conversations:
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
            max_workers=self.config.max_workers,
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

    def _run_eval_multi(self, prompt: str, dataset: Any = None, n_runs: int = 3) -> EvalSnapshot:
        """Run eval n_runs times and return an averaged EvalSnapshot with std_score."""
        import statistics

        logger.info(f"Running eval {n_runs}× and averaging...")
        snapshots = [self._run_eval_single(prompt, dataset=dataset) for _ in range(n_runs)]

        run_means = [s.mean_score for s in snapshots]
        mean = statistics.mean(run_means)
        std = statistics.stdev(run_means) if len(run_means) > 1 else 0.0

        # Average per-sample scores across runs (samples are in the same order)
        n_samples = min(len(s.samples) for s in snapshots) if snapshots else 0
        avg_samples: list[SampleSnapshot] = []
        for i in range(n_samples):
            avg_score = sum(s.samples[i].score for s in snapshots) / n_runs
            avg_samples.append(SampleSnapshot(
                input=snapshots[0].samples[i].input,
                response=snapshots[0].samples[i].response,
                ideal=snapshots[0].samples[i].ideal,
                score=avg_score,
            ))

        # Average per-metric scores
        all_metric_keys: set[str] = set()
        for s in snapshots:
            all_metric_keys.update(s.scores_by_metric)
        per_metric = {
            k: sum(s.scores_by_metric.get(k, 0.0) for s in snapshots) / n_runs
            for k in all_metric_keys
        }

        total_tokens = sum(s.total_tokens for s in snapshots)

        return EvalSnapshot(
            mean_score=mean,
            scores_by_metric=per_metric,
            samples=avg_samples,
            total_tokens=total_tokens,
            std_score=std,
            n_runs=n_runs,
        )

    @staticmethod
    def _compute_significance(
        baseline: "EvalSnapshot",
        final: "EvalSnapshot",
    ) -> tuple[float | None, bool | None]:
        """Paired significance test comparing baseline vs final per-sample scores.

        Uses the Wilcoxon signed-rank test (scipy) if available, falling back to
        a manual paired t-test otherwise. Returns (p_value, is_significant).

        Returns (None, None) if there are fewer than 2 paired samples or if the
        differences are all zero (the test is undefined).
        """
        import math

        if not baseline.samples or not final.samples:
            return None, None

        n = min(len(baseline.samples), len(final.samples))
        if n < 2:
            return None, None

        b_scores = [s.score for s in baseline.samples[:n]]
        f_scores = [s.score for s in final.samples[:n]]
        diffs = [f - b for f, b in zip(f_scores, b_scores)]

        # All identical — test undefined
        if all(d == 0.0 for d in diffs):
            return None, None

        # --- Try scipy Wilcoxon signed-rank test (non-parametric, no normality assumption) ---
        try:
            from scipy.stats import wilcoxon  # type: ignore[import]
            _, p_value = wilcoxon(b_scores, f_scores)
            return float(p_value), bool(p_value < 0.05)
        except ImportError:
            pass

        # --- Fallback: manual paired t-test ---
        mean_d = sum(diffs) / n
        var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        if var_d == 0.0:
            return None, None
        t_stat = mean_d / math.sqrt(var_d / n)

        # Approximate two-tailed p-value from t-distribution (Abramowitz & Stegun 26.7.8)
        # degrees of freedom = n - 1
        df = n - 1
        x = df / (df + t_stat ** 2)
        # Regularized incomplete beta function approximation for small df
        # For df >= 2 this is accurate to ~1%
        p_approx = _beta_inc_approx(x, df / 2, 0.5)
        return float(p_approx), bool(p_approx < 0.05)


def _beta_inc_approx(x: float, a: float, b: float) -> float:  # noqa: N802
    """Rough two-tailed p-value via continued-fraction incomplete-beta approximation.

    Only used when scipy is not installed. Accurate to ~±0.01 for df >= 2.
    """
    import math

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Log of the beta function B(a, b)
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    # Continued-fraction expansion (Lentz's method, 40 terms)
    def _cf(x: float, a: float, b: float) -> float:
        qab = a + b
        qap = a + 1.0
        qam = a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        h = d
        for m in range(1, 41):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            if abs(d) < 1e-30:
                d = 1e-30
            c = 1.0 + aa / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-7:
                break
        return h

    bt = math.exp(math.log(x) * a + math.log(1.0 - x) * b - log_beta)
    # Use symmetry relation for better convergence
    if x < (a + 1.0) / (a + b + 2.0):
        ibeta = bt * _cf(x, a, b) / a
    else:
        ibeta = 1.0 - bt * _cf(1.0 - x, b, a) / b

    # Two-tailed p-value
    return float(min(2.0 * ibeta, 1.0))
