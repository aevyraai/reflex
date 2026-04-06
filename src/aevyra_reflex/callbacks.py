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

"""Callbacks for integrating reflex runs with experiment tracking platforms.

Usage — MLflow::

    from aevyra_reflex import PromptOptimizer, MLflowCallback

    result = (
        PromptOptimizer()
        .set_dataset(dataset)
        .add_provider("openai", "gpt-4o-mini")
        .add_metric(RougeScore())
        .run("You are a helpful assistant.", callbacks=[MLflowCallback()])
    )

Usage — MLflow with explicit run name::

    cb = MLflowCallback(run_name="my-experiment", tracking_uri="http://localhost:5000")
    result = optimizer.run("You are a helpful assistant.", callbacks=[cb])

Usage — Weights & Biases::

    from aevyra_reflex import WandbCallback

    result = optimizer.run("You are a helpful assistant.", callbacks=[WandbCallback()])

Usage — multiple callbacks::

    result = optimizer.run(prompt, callbacks=[MLflowCallback(), WandbCallback()])
"""

from __future__ import annotations

import logging
import tempfile
import os
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback protocol — implement any subset of these methods
# ---------------------------------------------------------------------------

@runtime_checkable
class Callback(Protocol):
    """Protocol for reflex run callbacks.

    You only need to implement the methods you care about.
    All methods are optional — implement none, some, or all.
    """

    def on_run_start(self, config: Any, initial_prompt: str) -> None:
        """Called once before the baseline eval begins.

        Args:
            config: The OptimizerConfig for this run.
            initial_prompt: The starting system prompt.
        """
        ...

    def on_iteration(self, record: Any) -> None:
        """Called after each completed iteration.

        Args:
            record: IterationRecord with fields:
                - iteration (int): 1-based iteration number
                - score (float): mean score for this iteration
                - scores_by_metric (dict[str, float]): per-metric breakdown
                - system_prompt (str): the prompt used this iteration
                - reasoning (str): the reasoning model's explanation
        """
        ...

    def on_run_end(self, result: Any) -> None:
        """Called once after the final verification eval completes.

        Args:
            result: OptimizationResult with fields:
                - best_prompt (str): the best system prompt found
                - best_score (float): score of the best prompt
                - baseline.mean_score (float): score before optimization
                - improvement (float): best_score - baseline
                - improvement_pct (float): improvement as a percentage
                - score_trajectory (list[float]): score after each iteration
                - phase_history (list[dict]): auto strategy phase breakdown
        """
        ...


# ---------------------------------------------------------------------------
# MLflowCallback
# ---------------------------------------------------------------------------

class MLflowCallback:
    """Logs a reflex optimization run to MLflow.

    What gets logged:
        - Params: strategy, reasoning_model, max_iterations, score_threshold, model
        - Metrics: score and per-metric scores at each iteration (step = iteration number)
        - Metrics: baseline_score, best_score, improvement, improvement_pct (on run end)
        - Artifact: best_prompt.txt (the winning prompt)

    Args:
        run_name: MLflow run name. Defaults to "reflex-<strategy>".
        tracking_uri: MLflow tracking URI. If None, uses MLFLOW_TRACKING_URI
                      env var or MLflow's default (local ./mlruns directory).
        experiment_name: MLflow experiment name. Defaults to "aevyra-reflex".
        tags: Extra tags to attach to the MLflow run.
        log_prompt_each_iter: If True, logs the system prompt as an artifact
                              at every iteration (not just the best at the end).
                              Useful for auditing but creates more artifacts.
    """

    def __init__(
        self,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        experiment_name: str = "aevyra-reflex",
        tags: dict[str, str] | None = None,
        log_prompt_each_iter: bool = False,
    ):
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.tags = tags or {}
        self.log_prompt_each_iter = log_prompt_each_iter
        self._mlflow_run = None

    def on_run_start(self, config: Any, initial_prompt: str) -> None:
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "mlflow is required for MLflowCallback. "
                "Install it with: pip install mlflow"
            )

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)

        run_name = self.run_name or f"reflex-{getattr(config, 'strategy', 'optimize')}"

        self._mlflow_run = mlflow.start_run(run_name=run_name, tags=self.tags)

        # Log config as params
        params = {
            "strategy":        getattr(config, "strategy", ""),
            "reasoning_model": getattr(config, "reasoning_model", ""),
            "max_iterations":  getattr(config, "max_iterations", ""),
            "score_threshold": getattr(config, "score_threshold", ""),
            "temperature":     getattr(config, "temperature", ""),
            "max_workers":     getattr(config, "max_workers", ""),
        }
        if getattr(config, "target_model", None):
            params["target_model"] = config.target_model
        if getattr(config, "target_source", None):
            params["target_source"] = config.target_source

        mlflow.log_params(params)
        logger.info(f"MLflow run started: {self._mlflow_run.info.run_id}")

    def on_iteration(self, record: Any) -> None:
        try:
            import mlflow
        except ImportError:
            return

        step = record.iteration

        mlflow.log_metric("score", record.score, step=step)

        for metric_name, metric_score in (record.scores_by_metric or {}).items():
            mlflow.log_metric(f"score_{metric_name}", metric_score, step=step)

        if self.log_prompt_each_iter:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", prefix=f"prompt_iter_{step}_",
                delete=False
            ) as f:
                f.write(record.system_prompt)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path="prompts")
            os.unlink(tmp_path)

    def on_run_end(self, result: Any) -> None:
        try:
            import mlflow
        except ImportError:
            return

        # Summary metrics
        mlflow.log_metric("best_score", result.best_score)

        if result.baseline:
            mlflow.log_metric("baseline_score", result.baseline.mean_score)

        if result.improvement is not None:
            mlflow.log_metric("improvement", result.improvement)

        if result.improvement_pct is not None:
            mlflow.log_metric("improvement_pct", result.improvement_pct)

        mlflow.log_metric("converged", float(result.converged))
        mlflow.log_metric("total_iterations", len(result.iterations))

        # Best prompt as artifact
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="best_prompt_", delete=False
        ) as f:
            f.write(result.best_prompt)
            tmp_path = f.name

        mlflow.log_artifact(tmp_path, artifact_path="prompts")
        os.unlink(tmp_path)

        if self._mlflow_run:
            mlflow.end_run()
            logger.info(
                f"MLflow run complete: {self._mlflow_run.info.run_id} "
                f"(best score: {result.best_score:.4f})"
            )
            self._mlflow_run = None


# ---------------------------------------------------------------------------
# WandbCallback
# ---------------------------------------------------------------------------

class WandbCallback:
    """Logs a reflex optimization run to Weights & Biases.

    What gets logged:
        - Config: strategy, reasoning_model, max_iterations, score_threshold, etc.
        - Metrics: score and per-metric scores at each iteration
        - Summary: best_score, baseline_score, improvement, improvement_pct
        - Artifact: best_prompt.txt (the winning prompt)

    Args:
        project: W&B project name. Defaults to "aevyra-reflex".
        run_name: W&B run name. Defaults to "reflex-<strategy>".
        tags: List of string tags to attach to the W&B run.
        mode: W&B run mode — "online" (default), "offline" (no network,
              writes locally), or "disabled" (no-op, useful for testing).
        entity: W&B entity (username or team). If None, uses the default
                from your W&B login.
        log_prompt_each_iter: If True, logs the system prompt as a W&B
                              Table row at every iteration.
    """

    def __init__(
        self,
        project: str = "aevyra-reflex",
        run_name: str | None = None,
        tags: list[str] | None = None,
        mode: str = "online",
        entity: str | None = None,
        log_prompt_each_iter: bool = False,
    ):
        self.project = project
        self.run_name = run_name
        self.tags = tags or []
        self.mode = mode
        self.entity = entity
        self.log_prompt_each_iter = log_prompt_each_iter
        self._run = None
        self._prompt_table = None

    def on_run_start(self, config: Any, initial_prompt: str) -> None:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is required for WandbCallback. "
                "Install it with: pip install wandb"
            )

        run_name = self.run_name or f"reflex-{getattr(config, 'strategy', 'optimize')}"

        wandb_config = {
            "strategy":        getattr(config, "strategy", ""),
            "reasoning_model": getattr(config, "reasoning_model", ""),
            "max_iterations":  getattr(config, "max_iterations", ""),
            "score_threshold": getattr(config, "score_threshold", ""),
            "temperature":     getattr(config, "temperature", ""),
            "max_workers":     getattr(config, "max_workers", ""),
        }
        if getattr(config, "target_model", None):
            wandb_config["target_model"] = config.target_model
        if getattr(config, "target_source", None):
            wandb_config["target_source"] = config.target_source

        self._run = wandb.init(
            project=self.project,
            name=run_name,
            tags=self.tags,
            mode=self.mode,
            entity=self.entity,
            config=wandb_config,
        )

        if self.log_prompt_each_iter:
            self._prompt_table = wandb.Table(columns=["iteration", "prompt"])

        logger.info(f"W&B run started: {self._run.url if self._run else 'offline'}")

    def on_iteration(self, record: Any) -> None:
        try:
            import wandb
        except ImportError:
            return

        metrics: dict[str, Any] = {"score": record.score, "iteration": record.iteration}
        for metric_name, metric_score in (record.scores_by_metric or {}).items():
            metrics[f"score_{metric_name}"] = metric_score

        wandb.log(metrics, step=record.iteration)

        if self.log_prompt_each_iter and self._prompt_table is not None:
            self._prompt_table.add_data(record.iteration, record.system_prompt)

    def on_run_end(self, result: Any) -> None:
        try:
            import wandb
        except ImportError:
            return

        # Summary metrics (appear in the run overview, not the charts)
        summary: dict[str, Any] = {
            "best_score":       result.best_score,
            "total_iterations": len(result.iterations),
            "converged":        result.converged,
        }
        if result.baseline:
            summary["baseline_score"] = result.baseline.mean_score
        if result.improvement is not None:
            summary["improvement"] = result.improvement
        if result.improvement_pct is not None:
            summary["improvement_pct"] = result.improvement_pct

        if self._run:
            self._run.summary.update(summary)

        # Best prompt as artifact
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="best_prompt_", delete=False
        ) as f:
            f.write(result.best_prompt)
            tmp_path = f.name

        artifact = wandb.Artifact(name="best-prompt", type="prompt")
        artifact.add_file(tmp_path, name="best_prompt.txt")
        if self._run:
            self._run.log_artifact(artifact)
        os.unlink(tmp_path)

        # Log prompt table if we collected one
        if self._prompt_table is not None and self._run:
            wandb.log({"prompts": self._prompt_table})

        if self._run:
            self._run.finish()
            logger.info(f"W&B run complete (best score: {result.best_score:.4f})")
            self._run = None
            self._prompt_table = None
