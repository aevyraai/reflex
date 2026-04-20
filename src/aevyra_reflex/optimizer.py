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
        resolved_api_key = api_key or os.environ.get(alias["env_key"])
        if not resolved_api_key:
            raise ValueError(
                f"No API key found for {provider_name}. "
                f"Set the {alias['env_key']} environment variable "
                f"or pass api_key= directly."
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
    Default: 0.8 (80% train+val / 20% test).

    A larger test set is preferred over a larger training set because the test
    set determines statistical significance: ~30 held-out samples are needed for
    a paired test to reliably detect moderate improvements (p < 0.05). The
    optimizer needs far fewer training examples to identify failure patterns."""

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
        self._pipeline_fn: Any | None = None
        self._raw_inputs: list[Any] | None = None

    def set_dataset(self, dataset: Any) -> PromptOptimizer:
        """Set the evaluation dataset (a verdict Dataset instance)."""
        self._dataset = dataset
        return self

    def set_pipeline(self, fn: Any) -> PromptOptimizer:
        """Set the pipeline function for agentic prompt optimization.

        Use this instead of ``set_dataset()`` when the prompt controls a node
        inside a multi-step pipeline (classifier, retriever, responder, etc.)
        rather than a single LLM call.

        The function must accept ``(prompt: str, input: Any) -> AgentTrace`` and
        will be called once per input per optimization iteration with the current
        candidate prompt.  The optimizer scores each trace using the metrics you
        added with ``add_metric()``.

        Example::

            def run_pipeline(prompt: str, ticket: str) -> AgentTrace:
                ticket_type = classify(ticket)
                policy      = retrieve(ticket_type)
                response    = generate(ticket, policy, prompt)
                return AgentTrace(
                    nodes=[
                        TraceNode("classify", ticket, ticket_type),
                        TraceNode("retrieve", ticket_type, policy),
                        TraceNode("generate", ticket, response, optimize=True),
                    ],
                    ideal=expected_answer,
                )

            result = (
                PromptOptimizer()
                .set_pipeline(run_pipeline)
                .set_inputs(tickets)
                .add_metric(LLMJudge(...))
                .run(starting_prompt)
            )

        Args:
            fn: Callable ``(prompt: str, input: Any) -> AgentTrace``.

        Returns:
            self (for chaining).
        """
        self._pipeline_fn = fn
        return self

    def set_inputs(self, inputs: list[Any]) -> PromptOptimizer:
        """Set the raw inputs for pipeline mode.

        Each element is passed as the second argument to the pipeline function
        registered via ``set_pipeline()``.  This must be called when using
        ``set_pipeline()``.

        Args:
            inputs: List of raw input values (strings, dicts, or any type your
                    pipeline function accepts).

        Returns:
            self (for chaining).
        """
        self._raw_inputs = list(inputs)
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
        # --- Pipeline mode validation ---
        _pipeline_mode = self._pipeline_fn is not None
        if _pipeline_mode:
            if self._raw_inputs is None:
                raise ValueError(
                    "Pipeline mode requires inputs. Call set_inputs(inputs) before run()."
                )
            if self._dataset is not None:
                raise ValueError(
                    "set_dataset() and set_pipeline() are mutually exclusive. "
                    "Use set_pipeline() + set_inputs() for pipeline mode, or "
                    "set_dataset() for the traditional single-LLM-call flow."
                )
            if not self._raw_inputs:
                raise ValueError("set_inputs() received an empty list. Provide at least one input.")
            # Build a synthetic dataset so split/val logic works unchanged
            self._dataset = self._build_synthetic_dataset(self._raw_inputs)
            logger.info(
                f"[pipeline mode] {len(self._raw_inputs)} inputs — "
                f"pipeline will be re-run each iteration with the current prompt candidate"
            )

        if self._dataset is None:
            raise ValueError("No dataset set. Call set_dataset() first.")
        if not self._metrics:
            raise ValueError("No metrics added. Call add_metric() first.")

        # Providers are not required in pipeline mode — the pipeline fn handles model calls
        if not _pipeline_mode and not self._providers:
            raise ValueError("No providers added. Call add_provider() first.")

        # Fail fast for label-free datasets paired with reference-based metrics.
        # Skip this check in pipeline mode — ideals come from AgentTrace.ideal at eval time
        # and the judge (LLMJudge) doesn't require a pre-populated ideal field.
        if not _pipeline_mode and not self._dataset.has_ideals():
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

        # Warn when the split produces unusably small partitions
        _total_n = len(self._dataset.conversations)
        if n_train < 10 and train_ratio < 1.0:
            logger.warning(
                "Training set is very small (%d examples). With only %d examples the "
                "optimizer has little signal to work from. Consider using --train-split 1.0 "
                "to disable splitting, or collecting more data (aim for ≥30 train examples).",
                n_train, n_train,
            )
        if n_test < 8 and train_ratio < 1.0:
            logger.warning(
                "Test set is very small (%d examples). Scores will be noisy and the "
                "significance test won't be reliable. Aim for ≥20 test examples, or use "
                "--train-split 1.0 to skip the split on small datasets.",
                n_test,
            )

        if self.config.batch_size > 0:
            effective_batch = min(self.config.batch_size, n_train)
            logger.info(
                f"Mini-batch mode: {effective_batch} examples/iter "
                f"(from {n_train} training examples) — "
                f"baseline and final evals use full test set"
            )

        # Detect dataset language so the reasoning model responds in kind.
        # We sample a few user messages from the training split (or the full
        # dataset when no split is used) and identify the dominant script.
        from aevyra_reflex.lang import detect_language
        _lang_samples = [
            c.last_user_message
            for c in train_dataset.conversations[:20]
            if c.last_user_message
        ]
        _detected_language = detect_language(_lang_samples)
        logger.info(f"Detected dataset language: {_detected_language}")

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
            # In pipeline mode use the inputs file path as the stable identifier
            # so --resume can match this run by inputs_file path.
            _dataset_path = (
                self.config.extra_kwargs.get("_inputs_file", "unknown")
                if _pipeline_mode
                else getattr(self._dataset, '_source_path', 'unknown')
            )
            run = run_store.create_run(
                config=config_dict,
                dataset_path=_dataset_path,
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
                    "samples": [
                        {"input": s.input, "response": s.response, "ideal": s.ideal, "score": s.score}
                        for s in baseline.samples
                    ],
                })
        elif checkpoint and checkpoint.baseline:
            logger.info(f"{tag} Resuming — using saved baseline.")
            from aevyra_reflex.result import SampleSnapshot
            raw_samples = checkpoint.baseline.get("samples", [])
            baseline = EvalSnapshot(
                mean_score=checkpoint.baseline["mean_score"],
                scores_by_metric=checkpoint.baseline.get("scores_by_metric", {}),
                total_tokens=checkpoint.baseline.get("total_tokens", 0),
                samples=[
                    SampleSnapshot(
                        input=s.get("input", ""),
                        response=s.get("response", ""),
                        ideal=s.get("ideal", ""),
                        score=s.get("score", 0.0),
                    )
                    for s in raw_samples
                ],
            )
            # Older checkpoints (written before _baseline_ckpt added "samples")
            # won't have per-sample scores. Try to recover them from baseline.json.
            if not baseline.samples and run:
                try:
                    _saved_bl = run.load_baseline()
                    if _saved_bl:
                        baseline.samples = [
                            SampleSnapshot(
                                input=s.get("input", ""),
                                response=s.get("response", ""),
                                ideal=s.get("ideal", ""),
                                score=s.get("score", 0.0),
                            )
                            for s in _saved_bl.get("samples", [])
                        ]
                except Exception:
                    pass  # best-effort; significance test will just report n/a
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
                    "samples": [
                        {"input": s.input, "response": s.response, "ideal": s.ideal, "score": s.score}
                        for s in baseline.samples
                    ],
                })

        # Fire on_baseline so callbacks can log the test-set baseline score
        for cb in _callbacks:
            if hasattr(cb, "on_baseline"):
                try:
                    cb.on_baseline(baseline)
                except Exception:
                    logger.exception(f"Callback {cb!r} raised in on_baseline")

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
                baseline=_baseline_ckpt(baseline),
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
            response_language=_detected_language,
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
        # Restore global best-val (falls back to per-phase best-val for old checkpoints)
        _resumed_global_best_val_prompt: str = (
            checkpoint.global_best_val_prompt if (checkpoint and checkpoint.global_best_val_prompt)
            else _resumed_best_val_prompt
        )
        _resumed_global_best_val_score: float = (
            checkpoint.global_best_val_score if checkpoint else -1.0
        )
        _resumed_global_best_val_iter: int = (
            checkpoint.global_best_val_iter if checkpoint else 0
        )
        # Reconstruct the train score at the best-val iteration from saved iteration files
        _resumed_best_val_train_score: float = -1.0
        _resumed_global_best_val_train_score: float = -1.0
        if checkpoint and run and (_resumed_best_val_iter > 0 or _resumed_global_best_val_iter > 0):
            for it in run.load_iterations():
                if it.iteration == _resumed_best_val_iter:
                    _resumed_best_val_train_score = it.score
                if it.iteration == _resumed_global_best_val_iter:
                    _resumed_global_best_val_train_score = it.score

        # Shared mutable state for val tracking and early stopping
        _es: dict[str, Any] = {
            "val_history": _resumed_val_history,  # val score per iteration (restored on resume)
            "iterations": [],            # IterationRecord objects collected
            "best_train_prompt": checkpoint.current_prompt if checkpoint else initial_prompt,
            "best_train_score": checkpoint.best_score if checkpoint else -1.0,
            # per-phase best-val: reset at each phase transition for early stopping
            "best_val_prompt": _resumed_best_val_prompt,
            "best_val_score": _resumed_best_val_score,
            "best_val_train_score": _resumed_best_val_train_score,  # train score at best-val iter
            "best_val_iter": _resumed_best_val_iter,  # global iteration number of best val
            # global best-val: never reset; tracks best val across all phases and resumes
            "global_best_val_prompt": _resumed_global_best_val_prompt,
            "global_best_val_score": _resumed_global_best_val_score,
            "global_best_val_iter": _resumed_global_best_val_iter,
            "global_best_val_train_score": _resumed_global_best_val_train_score,
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
                    baseline=_baseline_ckpt(baseline),
                    best_val_prompt=_es["best_val_prompt"] if val_dataset else None,
                    best_val_score=_es["best_val_score"],
                    best_val_iter=_es["best_val_iter"],
                    global_best_val_prompt=_es["global_best_val_prompt"] if val_dataset else None,
                    global_best_val_score=_es["global_best_val_score"],
                    global_best_val_iter=_es["global_best_val_iter"],
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
                # Also update global best-val (never reset between phases)
                if val_score >= _es["global_best_val_score"]:
                    _es["global_best_val_score"] = val_score
                    _es["global_best_val_prompt"] = record.system_prompt
                    _es["global_best_val_iter"] = record.iteration
                    _es["global_best_val_train_score"] = record.score

            # Fire on_iteration callbacks after val eval so val_score is populated
            if on_iteration:
                on_iteration(record)
            for cb in _callbacks:
                if hasattr(cb, "on_iteration"):
                    try:
                        cb.on_iteration(record)
                    except Exception:
                        logger.exception(f"Callback {cb!r} raised in on_iteration")

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
                    baseline=_baseline_ckpt(baseline),
                    best_val_prompt=_es["best_val_prompt"] if val_dataset else None,
                    best_val_score=_es["best_val_score"],
                    best_val_iter=_es["best_val_iter"],
                    global_best_val_prompt=_es["global_best_val_prompt"] if val_dataset else None,
                    global_best_val_score=_es["global_best_val_score"],
                    global_best_val_iter=_es["global_best_val_iter"],
                    accumulated_duration_seconds=_prior_duration_seconds + _time.monotonic() - _run_start_monotonic,
                ))

        _early_stopped = False
        _eval_fn = self._make_pipeline_eval_fn() if _pipeline_mode else None
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
                eval_fn=_eval_fn,
                **({"resume_state": checkpoint.strategy_state} if checkpoint and checkpoint.strategy_state else {}),
            )
        except _EarlyStop:
            _early_stopped = True
            # Build a partial result from what we've collected so far
            best_prompt = _es["global_best_val_prompt"] if val_dataset else _es["best_train_prompt"]
            best_score = _es["best_val_score"] if val_dataset else _es["best_train_score"]
            from aevyra_reflex.result import OptimizationResult as _OR
            result = _OR(
                best_prompt=best_prompt,
                best_score=best_score,
                iterations=list(_es["iterations"]),
                converged=False,
            )
        except Exception as _exc:
            # Log the crash to the run directory so users can see why it failed,
            # then re-raise so the CLI can display the traceback as normal.
            if run:
                import traceback as _tb
                try:
                    (run.run_dir / "error.txt").write_text(
                        f"Run crashed at {__import__('datetime').datetime.now().isoformat()}\n\n"
                        + _tb.format_exc()
                    )
                    run.mark_done()  # remove running sentinel so dashboard shows interrupted
                except Exception:
                    pass
            raise

        # --- Restore full iteration history for resumed runs ---
        # When a run is resumed purely to execute the final test eval (e.g. after
        # a crash between the last optimization iteration and the test eval), the
        # strategy.run() call completes with zero new iterations this session.
        # result.iterations is therefore empty, which produces "Iterations: 0"
        # and a blank Train traj in the summary. Load the saved files to fix this.
        if run and not result.iterations:
            from aevyra_reflex.result import IterationRecord as _IR
            _saved_iters = list(run.load_iterations())
            if _saved_iters:
                result.iterations = [
                    _IR(
                        iteration=it.iteration,
                        system_prompt=it.system_prompt,
                        score=it.score,
                        scores_by_metric=getattr(it, "scores_by_metric", {}),
                        reasoning=getattr(it, "reasoning", ""),
                        eval_tokens=getattr(it, "eval_tokens", 0),
                        reasoning_tokens=getattr(it, "reasoning_tokens", 0),
                        change_summary=getattr(it, "change_summary", ""),
                        val_score=getattr(it, "val_score", None),
                        is_full_eval=getattr(it, "is_full_eval", False),
                    )
                    for it in _saved_iters
                ]

        # --- Step 3: Final verification eval (on held-out test set) ---
        eval_runs = self.config.eval_runs
        eval_label = f" ({eval_runs} runs)" if eval_runs > 1 else ""
        n_test_samples = len(test_dataset.conversations) if test_dataset else "?"
        # Select the prompt to test: when a val set was used, prefer the
        # prompt with the best val score (least likely to be overfit to train).
        # Fall back to the best-train prompt if val tracking was never updated.
        if val_dataset is not None and _es["global_best_val_score"] > -1.0:
            prompt_for_test = _es["global_best_val_prompt"]
            _bv_iter = _es["global_best_val_iter"]
            _bv_train = _es["global_best_val_train_score"]
            _bv_val = _es["global_best_val_score"]
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

        # LLM-generated post-run analysis
        try:
            logger.info(f"{tag} Generating run analysis...")
            what_happened_prompt = _build_what_happened_prompt(result)
            result.what_happened = llm.generate(what_happened_prompt, temperature=0.3)
        except Exception:
            logger.debug("Failed to generate what_happened analysis", exc_info=True)

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

        # Fire on_final so callbacks can log the final test-set score before on_run_end
        for cb in _callbacks:
            if hasattr(cb, "on_final"):
                try:
                    cb.on_final(final)
                except Exception:
                    logger.exception(f"Callback {cb!r} raised in on_final")

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

    def _build_synthetic_dataset(self, inputs: list[Any]) -> Any:
        """Build a synthetic Verdict Dataset from raw pipeline inputs.

        Each input is stored in conversation metadata under ``_pipeline_input``
        so the eval function can retrieve it.  The conversation itself is a
        placeholder that the pipeline eval ignores — it's here so existing split
        and validation logic works unchanged.

        Args:
            inputs: Raw pipeline inputs (one per example).

        Returns:
            A verdict Dataset with one conversation per input.
        """
        from aevyra_verdict.dataset import Conversation, Dataset, Message

        convos = []
        for raw_input in inputs:
            # Store the raw input in metadata; use a minimal placeholder message
            # so the dataset is valid for splitting/val logic.
            input_str = raw_input if isinstance(raw_input, str) else str(raw_input)
            convos.append(Conversation(
                messages=[Message(role="user", content=input_str)],
                ideal=None,  # ideals come from AgentTrace.ideal at eval time
                metadata={"_pipeline_input": raw_input},
            ))
        return Dataset(conversations=convos)

    def _run_pipeline_eval(
        self,
        prompt: str,
        dataset: Any,
        *,
        bottom_k: int = 10,
        show_progress: bool = False,
    ) -> tuple[float, list[dict[str, Any]], int]:
        """Run the pipeline function for each input and score the resulting traces.

        Replaces EvalRunner in pipeline mode. Calls ``self._pipeline_fn(prompt, input)``
        for every conversation in ``dataset`` (using the ``_pipeline_input`` metadata
        field), collects the resulting ``AgentTrace`` objects, and scores them directly
        using ``metric.score()`` on the trace text.

        Args:
            prompt:   The current candidate system prompt.
            dataset:  A synthetic Dataset built by ``_build_synthetic_dataset()``.
            bottom_k: Number of worst-scoring samples to return for diagnosis.

        Returns:
            ``(mean_score, failing_samples, total_tokens)`` — the same shape that
            the strategy-layer ``_run_eval`` function returns.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        assert self._pipeline_fn is not None, "No pipeline function set"

        def _is_permanent_api_error(exc: Exception) -> bool:
            """Return True for errors that will always fail on retry (e.g. context overflow).

            400 Bad Request is the canonical permanent failure — the same payload
            will produce the same error regardless of how many times we retry.
            Transient errors (429, 5xx, timeouts) are handled by the OpenAI SDK's
            built-in retry logic before the exception ever reaches us.
            """
            status = getattr(exc, "status_code", None)
            if status == 400:
                return True
            msg = str(exc).lower()
            return (
                "input length" in msg
                or "context" in msg
                or "too long" in msg
                or "maximum context" in msg
                or "badrequest" in type(exc).__name__.lower()
            )

        def _run_one(convo: Any) -> tuple[float, dict[str, Any], int]:
            raw_input = convo.metadata.get("_pipeline_input", convo.last_user_message)
            try:
                trace = self._pipeline_fn(prompt, raw_input)
            except Exception as _exc:
                _msg = str(_exc)
                if _is_permanent_api_error(_exc):
                    # Context-window overflow or other permanent 400 — will never
                    # succeed on retry. Score 0.0 and continue the eval.
                    logger.warning(
                        "Pipeline call failed for one sample (permanent API error, "
                        "likely context-window overflow): %s — scoring as 0.0.",
                        _msg[:200],
                    )
                else:
                    # Transient error not caught by the SDK's retry layer.
                    # Log at ERROR level so it's visible; still score 0.0 so the
                    # rest of the test set can complete.
                    logger.error(
                        "Pipeline call raised an unexpected error for one sample "
                        "(%s): %s — scoring as 0.0.",
                        type(_exc).__name__, _msg[:200],
                    )
                raw_str = str(raw_input)[:500]
                return 0.0, {"input": raw_str, "response": f"[error: {_msg[:300]}]", "ideal": "", "score": 0.0}, 0

            trace_text = trace.to_trace_text()
            ideal = trace.ideal

            # Pipeline tokens reported by the user's pipeline_fn via AgentTrace.tokens
            pipeline_toks = getattr(trace, "tokens", 0)

            # Score the trace with every metric
            sample_scores: list[float] = []
            for metric in self._metrics:
                try:
                    result = metric.score(
                        response=trace_text,
                        ideal=ideal or "",
                        messages=[],
                    )
                    val = result.score if hasattr(result, "score") else float(result)
                    sample_scores.append(val)
                except Exception as _score_exc:
                    _smsg = str(_score_exc)
                    if _is_permanent_api_error(_score_exc):
                        logger.warning(
                            "Metric scoring failed for one sample (permanent API error): %s — scoring as 0.0.",
                            _smsg[:200],
                        )
                    else:
                        logger.error(
                            "Metric scoring raised an unexpected error for one sample "
                            "(%s): %s — scoring as 0.0.",
                            type(_score_exc).__name__, _smsg[:200],
                        )
                    sample_scores.append(0.0)

            mean_sample = sum(sample_scores) / len(sample_scores) if sample_scores else 0.0
            failing_info = {
                "input": str(raw_input)[:500],
                "response": trace_text[:3000],  # full trace incl. answer; 1000 cuts off the answer node
                "ideal": str(ideal or "")[:500],
                "score": mean_sample,
            }
            return mean_sample, failing_info, pipeline_toks

        # Suppress per-question progress traces during optimizer runs — the
        # interleaved output from parallel workers is noise. Users can still
        # set PIPELINE_QUIET=0 explicitly to re-enable traces for debugging.
        if "PIPELINE_QUIET" not in os.environ:
            os.environ["PIPELINE_QUIET"] = "1"

        max_workers = self.config.max_workers or 4
        all_scores: list[float] = []
        all_infos: list[dict[str, Any]] = []
        total_tokens = 0

        # Snapshot judge tokens before running so we compute the delta correctly
        _judge_before = [getattr(m, "judge_tokens_used", 0) for m in self._metrics]

        # Per-sample wall-clock timeout (seconds). A hung API call (e.g. a TCP
        # connection that never closes) will block a worker thread forever without
        # this. 300s is generous — any real completion should finish well inside
        # that window even for slow local models. Override with PIPELINE_SAMPLE_TIMEOUT.
        _sample_timeout = int(os.environ.get("PIPELINE_SAMPLE_TIMEOUT", "300"))

        n_inputs = len(dataset.conversations)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_one, convo): convo for convo in dataset.conversations}
            if show_progress:
                from tqdm import tqdm
                _cm = tqdm(total=n_inputs, desc="  pipeline", unit="q",
                           leave=False, dynamic_ncols=True)
            else:
                from contextlib import nullcontext
                _cm = nullcontext()
            with _cm as pbar:
                for future in as_completed(futures, timeout=_sample_timeout * n_inputs):
                    try:
                        score, info, pipeline_toks = future.result(timeout=_sample_timeout)
                    except TimeoutError:
                        convo = futures[future]
                        raw_input = convo.metadata.get("_pipeline_input", convo.last_user_message)
                        logger.warning(
                            "Pipeline call timed out after %ds for one sample — scoring as 0.0.",
                            _sample_timeout,
                        )
                        score, info, pipeline_toks = 0.0, {
                            "input": str(raw_input)[:500],
                            "response": f"[timed out after {_sample_timeout}s]",
                            "ideal": "",
                            "score": 0.0,
                        }, 0
                    all_scores.append(score)
                    all_infos.append(info)
                    total_tokens += pipeline_toks  # tokens from inside pipeline_fn
                    if pbar is not None:
                        pbar.update(1)

        # Add judge token delta (judge metrics accumulate across calls)
        for m, before in zip(self._metrics, _judge_before):
            total_tokens += getattr(m, "judge_tokens_used", 0) - before

        mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        sorted_infos = sorted(all_infos, key=lambda d: d["score"])
        failing = sorted_infos[:bottom_k]

        return mean_score, failing, total_tokens

    def _make_pipeline_eval_fn(self) -> Any:
        """Return a callable that strategies use instead of building their own EvalRunner.

        The returned function matches the ``eval_fn`` signature expected by all
        strategies: ``(prompt, dataset, *, bottom_k) -> (score, failing, tokens)``.

        Internally it delegates to ``_run_pipeline_eval``, ignoring the synthetic
        dataset argument (it always evaluates against ``_raw_inputs``).
        """
        # Build a fresh synthetic dataset from the full raw inputs — strategies
        # may receive a training-split subset, but in pipeline mode we re-run the
        # pipeline on the actual inputs carried in each conversation's metadata, so
        # we just delegate through and let _run_pipeline_eval extract them.
        def eval_fn(prompt: str, dataset: Any, *, bottom_k: int = 10) -> tuple[float, list[dict], int]:
            return self._run_pipeline_eval(prompt, dataset, bottom_k=bottom_k)

        return eval_fn

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
        """Run a single verdict eval pass and return an EvalSnapshot.

        In pipeline mode (when ``_pipeline_fn`` is set), delegates to
        ``_run_pipeline_eval()`` and wraps the result as an EvalSnapshot so the
        caller doesn't need to know which mode it's in.
        """
        if self._pipeline_fn is not None:
            eval_dataset = dataset if dataset is not None else self._dataset
            score, failing, total_tokens = self._run_pipeline_eval(
                prompt, eval_dataset,
                bottom_k=len(eval_dataset.conversations),
                show_progress=True,
            )
            sample_snapshots = [
                SampleSnapshot(
                    input=s.get("input", ""),
                    response=s.get("response", ""),
                    ideal=s.get("ideal", ""),
                    score=s.get("score", 0.0),
                )
                for s in failing
            ]
            return EvalSnapshot(
                mean_score=score,
                scores_by_metric={},
                samples=sample_snapshots,
                total_tokens=total_tokens,
            )

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

        # Snapshot judge token counters before running so we can compute the delta
        _judge_tokens_before = [getattr(m, "judge_tokens_used", 0) for m in self._metrics]

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
        # Also count tokens used by any LLM judge metrics (scored separately,
        # not part of model_results). Use the delta since _judge_tokens_before
        # to avoid double-counting across multiple eval calls.
        for m, before in zip(self._metrics, _judge_tokens_before):
            total_tokens += getattr(m, "judge_tokens_used", 0) - before

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
            logger.debug(
                "Significance test skipped: baseline_samples=%d final_samples=%d",
                len(baseline.samples), len(final.samples),
            )
            return None, None

        n = min(len(baseline.samples), len(final.samples))
        if n < 2:
            logger.debug("Significance test skipped: only %d paired samples (need ≥2)", n)
            return None, None

        b_scores = [s.score for s in baseline.samples[:n]]
        f_scores = [s.score for s in final.samples[:n]]
        diffs = [f - b for f, b in zip(f_scores, b_scores)]

        # All identical — test undefined
        if all(d == 0.0 for d in diffs):
            logger.debug("Significance test skipped: all per-sample differences are zero")
            return None, None

        # --- Try scipy Wilcoxon signed-rank test (non-parametric, no normality assumption) ---
        try:
            from scipy.stats import wilcoxon  # type: ignore[import]
            _, p_value = wilcoxon(b_scores, f_scores)
            return float(p_value), bool(p_value < 0.05)
        except ImportError:
            pass
        except Exception as exc:
            # Wilcoxon can raise ValueError with small n or degenerate differences
            # (e.g. too many ties, only 1 non-zero diff). Fall through to t-test.
            logger.debug("Wilcoxon test failed (%s: %s), falling back to t-test", type(exc).__name__, exc)

        # --- Fallback: manual paired t-test ---
        mean_d = sum(diffs) / n
        var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        if var_d == 0.0:
            logger.debug("Significance test skipped: zero variance in differences")
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


def _baseline_ckpt(snap: "EvalSnapshot") -> dict[str, Any]:
    """Serialize an EvalSnapshot for storage in checkpoint.json.

    Per-sample scores are included so that a run resumed purely to execute
    the final test eval can reconstruct ``baseline.samples`` and compute
    significance tests correctly.
    """
    return {
        "mean_score": snap.mean_score,
        "scores_by_metric": snap.scores_by_metric,
        "total_tokens": snap.total_tokens,
        "samples": [
            {"input": s.input, "response": s.response, "ideal": s.ideal, "score": s.score}
            for s in snap.samples
        ],
    }


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


def _build_what_happened_prompt(result: "OptimizationResult") -> str:  # type: ignore[name-defined]
    """Build the prompt sent to the reasoning model to generate WHAT HAPPENED.

    The prompt is constructed from the actual run data so the analysis is
    specific to *this* run rather than a generic template.
    """
    r = result

    baseline_score = r.baseline.mean_score if r.baseline else 0.0
    final_score = r.final.mean_score if r.final else (r.best_score or 0.0)
    improvement = final_score - baseline_score
    improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0.0
    n_iters = len(r.iterations)
    strategy = r.strategy_name or "unknown"
    traj = [f"{s:.3f}" for s in r.score_trajectory]

    # Build score trajectory context
    traj_str = " → ".join(traj) if traj else "(none)"
    val_traj_str = ""
    if r.val_trajectory:
        val_traj_str = " → ".join(f"{s:.3f}" for s in r.val_trajectory)

    # Dataset split info
    split_info = ""
    if r.train_size and r.test_size:
        if r.val_size:
            split_info = f"{r.train_size} train / {r.val_size} val / {r.test_size} test samples"
        else:
            split_info = f"{r.train_size} train / {r.test_size} test samples"
    elif r.train_size:
        split_info = f"{r.train_size} samples"

    # Significance
    sig_str = ""
    if r.p_value is not None:
        sig_str = (
            f"p={r.p_value:.4f} ({'significant' if r.is_significant else 'not significant'} at α=0.05)"
        )

    # Phase history (auto mode)
    phase_lines = ""
    if r.phase_history:
        parts = []
        for ph in r.phase_history:
            axis = ph.get("axis", "?")
            before = ph.get("score_before", 0.0)
            after = ph.get("score_after", 0.0)
            delta = after - before
            sign = "+" if delta >= 0 else ""
            parts.append(f"  Phase {ph.get('phase', '?')} ({axis}): {before:.4f} → {after:.4f} ({sign}{delta:.4f})")
        phase_lines = "\n".join(parts)

    # Failing sample examples (up to 3 worst-performing from final eval)
    failing_examples = ""
    if r.final and r.final.samples:
        sorted_samples = sorted(r.final.samples, key=lambda s: s.score)
        worst = sorted_samples[:3]
        parts = []
        for i, s in enumerate(worst, 1):
            inp = s.input[:300].replace("\n", " ") if s.input else "(no input)"
            resp = s.response[:200].replace("\n", " ") if s.response else "(no response)"
            ideal = s.ideal[:200].replace("\n", " ") if s.ideal else "(no ideal)"
            parts.append(
                f"  [{i}] score={s.score:.3f}\n"
                f"      input  : {inp}\n"
                f"      actual : {resp}\n"
                f"      ideal  : {ideal}"
            )
        failing_examples = "\n".join(parts)

    # Change summaries — grouped by qualitative category, not iteration number.
    # Deduplicate similar summaries so the LLM sees patterns, not a numbered list.
    change_log = ""
    if r.iterations:
        changes = [rec.change_summary for rec in r.iterations if rec.change_summary]
        if changes:
            change_log = "\n".join(f"  - {c}" for c in changes)

    # Identify which changes correlated with score gains vs drops
    positive_changes: list[str] = []
    negative_changes: list[str] = []
    if r.iterations:
        for i in range(1, len(r.iterations)):
            prev = r.iterations[i - 1].score
            curr = r.iterations[i].score
            summary = r.iterations[i].change_summary
            if not summary:
                continue
            if curr > prev + 0.005:
                positive_changes.append(summary)
            elif curr < prev - 0.005:
                negative_changes.append(summary)

    effective_log = ""
    if positive_changes:
        effective_log += "Changes that raised the score:\n" + "\n".join(f"  - {c}" for c in positive_changes)
    if negative_changes:
        if effective_log:
            effective_log += "\n"
        effective_log += "Changes that hurt the score:\n" + "\n".join(f"  - {c}" for c in negative_changes)

    # Early stopping
    early_stop_note = ""
    if r.early_stopped:
        early_stop_note = "Yes — validation score plateaued before max iterations."

    # Convergence
    converged_note = "Yes — met the score threshold." if r.converged else "No."

    # Build the full context block
    context_parts = [
        f"Strategy     : {strategy}",
        f"Iterations   : {n_iters}",
        f"Dataset split: {split_info}" if split_info else "",
        f"Baseline     : {baseline_score:.4f}",
        f"Final        : {final_score:.4f}  ({'+' if improvement >= 0 else ''}{improvement:.4f}, {'+' if improvement_pct >= 0 else ''}{improvement_pct:.1f}%)",
        f"Converged    : {converged_note}",
        f"Significance : {sig_str}" if sig_str else "",
        f"Train traj   : {traj_str}",
        f"Val traj     : {val_traj_str}" if val_traj_str else "",
        f"Early stop   : {early_stop_note}" if early_stop_note else "",
    ]
    context = "\n".join(line for line in context_parts if line)

    prompt = f"""You are analyzing the results of a prompt optimization run. Write the "WHAT HAPPENED" section — a concise, specific post-run analysis for the user.

--- RUN SUMMARY ---
{context}

--- WHAT EACH CHANGE DID ---
{effective_log if effective_log else change_log if change_log else "(no per-iteration change summaries available)"}
{f"--- PHASE BREAKDOWN (auto strategy) ---{chr(10)}{phase_lines}" if phase_lines else ""}
--- WORST-PERFORMING SAMPLES FROM FINAL EVAL ---
{failing_examples if failing_examples else "(no sample-level data available)"}

--- INSTRUCTIONS ---
Write exactly 2–3 short paragraphs. Be direct and specific. Rules:

- NEVER mention iteration numbers (e.g. "iteration 4", "iter 6"). Talk about *types* of changes: "adding structure", "tightening constraints", "adding explicit examples", etc.
- Paragraph 1: What worked and why. What kind of prompt change moved the score, and what does that tell us about the model's relationship to this task?
- Paragraph 2: What failed and why. What kinds of changes hurt or plateaued? What do the worst samples tell us about where the prompt still breaks down?
- Paragraph 3: One honest diagnosis of the ceiling (prompt problem vs model capability problem) and 1–2 specific next steps grounded in what you saw.
- Do NOT use bullet points. Plain prose only.
- Do NOT be generic — every sentence must be grounded in the data above.
- Do NOT start with "The optimizer".
- Keep it under 200 words total.
"""
    return prompt
