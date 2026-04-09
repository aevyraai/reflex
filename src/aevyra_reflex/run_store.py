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

"""Run persistence — checkpointing, resume, and multi-run versioning.

Manages a `.reflex/` directory that stores every optimization run with its
config, iterations, and results. Each run gets a sequential ID and a
timestamped directory:

    .reflex/
      runs/
        001_2026-04-04T10-32-15/
          config.json
          baseline.json
          checkpoint.json
          iterations/
            001.json
            002.json
          best_prompt.md
          result.json            # only when run completes
        002_2026-04-05T14-10-00/
          ...

Checkpointing:
    After each iteration, the strategy calls `store.save_iteration(...)`.
    This writes the iteration file and updates `checkpoint.json` + `best_prompt.md`.
    On crash, `store.load_checkpoint()` restores the full state so the
    strategy can skip completed iterations and continue.

Versioning:
    Each run is immutable once finished. `store.list_runs()` returns all
    runs sorted by ID. `store.compare_runs(id1, id2)` diffs configs and scores.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default root directory
DEFAULT_ROOT = ".reflex"


@dataclass
class IterationState:
    """Serializable state for a single completed iteration."""

    iteration: int
    system_prompt: str
    score: float
    scores_by_metric: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    eval_tokens: int = 0       # tokens consumed by eval model(s) this iteration
    reasoning_tokens: int = 0  # tokens consumed by reasoning model this iteration
    change_summary: str = ""   # one-liner: what the reasoning model changed this iteration
    val_score: float | None = None  # validation-set score this iteration (None when val_ratio=0)
    is_full_eval: bool = False      # True when scored on full training set (mini-batch checkpoint)
    timestamp: str = ""


@dataclass
class CheckpointState:
    """Full state needed to resume an interrupted run."""

    run_id: str
    initial_prompt: str
    current_prompt: str
    completed_iterations: int
    best_prompt: str
    best_score: float
    score_trajectory: list[float] = field(default_factory=list)
    previous_reasoning: str = ""

    # Strategy-specific state (auto phases, PDO pool, etc.)
    strategy_state: dict[str, Any] = field(default_factory=dict)

    # Baseline eval (so we don't re-run it on resume)
    baseline: dict[str, Any] | None = None

    timestamp: str = ""


@dataclass
class RunSummary:
    """Lightweight summary of a run for listing/comparison."""

    run_id: str
    run_dir: str
    timestamp: str
    status: str  # "running", "completed", "interrupted"
    strategy: str
    model: str
    dataset: str
    baseline_score: float | None
    best_score: float | None
    final_score: float | None
    iterations_completed: int
    duration_seconds: float | None = None
    parent_run_id: str | None = None    # set when this run was branched
    parent_iteration: int | None = None  # the iteration it branched from
    config: dict[str, Any] = field(default_factory=dict)


class RunStore:
    """Manages the .reflex/ directory and all run persistence.

    Usage::

        store = RunStore()  # uses .reflex/ in cwd
        run = store.create_run(config_dict, dataset_path, prompt_path)

        # Save baseline
        run.save_baseline(baseline_snapshot)

        # After each iteration
        run.save_iteration(iteration_state)

        # On completion
        run.save_result(result_dict)

        # Resume
        checkpoint = run.load_checkpoint()
    """

    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"

    def _ensure_dirs(self) -> None:
        """Create the root and runs directories if they don't exist."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def _next_run_id(self) -> str:
        """Generate the next sequential run ID like '001', '002', etc."""
        existing = self._list_run_dirs()
        if not existing:
            return "001"
        # Extract numeric prefixes
        max_id = 0
        for d in existing:
            match = re.match(r"^(\d+)_", d.name)
            if match:
                max_id = max(max_id, int(match.group(1)))
        return f"{max_id + 1:03d}"

    def _list_run_dirs(self) -> list[Path]:
        """List all run directories sorted by name."""
        if not self.runs_dir.exists():
            return []
        dirs = [d for d in self.runs_dir.iterdir() if d.is_dir()]
        return sorted(dirs)

    def create_run(
        self,
        config: dict[str, Any],
        dataset_path: str,
        prompt_path: str,
        initial_prompt: str,
        branched_from: dict[str, Any] | None = None,
    ) -> Run:
        """Create a new run directory and return a Run handle.

        Args:
            branched_from: Optional dict with ``run_id`` and ``iteration`` keys
                indicating this run was branched from a parent run.
        """
        self._ensure_dirs()
        run_id = self._next_run_id()
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        dir_name = f"{run_id}_{ts}"
        run_dir = self.runs_dir / dir_name
        run_dir.mkdir()
        (run_dir / "iterations").mkdir()

        # Write config
        run_config = {
            "run_id": run_id,
            "timestamp": ts,
            "dataset_path": str(dataset_path),
            "prompt_path": str(prompt_path),
            "initial_prompt": initial_prompt,
            "optimizer_config": config,
        }
        if branched_from:
            run_config["branched_from"] = branched_from
        _write_json(run_dir / "config.json", run_config)

        logger.info(f"Created run {run_id} at {run_dir}")
        return Run(run_id=run_id, run_dir=run_dir)

    def get_run(self, run_id: str) -> Run | None:
        """Get a Run handle by ID (e.g. '001')."""
        for d in self._list_run_dirs():
            if d.name.startswith(f"{run_id}_"):
                return Run(run_id=run_id, run_dir=d)
        return None

    def get_latest_run(self) -> Run | None:
        """Get the most recent run."""
        dirs = self._list_run_dirs()
        if not dirs:
            return None
        d = dirs[-1]
        run_id = d.name.split("_")[0]
        return Run(run_id=run_id, run_dir=d)

    def find_incomplete_run(
        self,
        dataset_path: str | None = None,
        model: str | None = None,
    ) -> Run | None:
        """Find the latest run that was interrupted (has checkpoint but no result).

        Optionally filter by dataset path and model to match a specific
        invocation.
        """
        for d in reversed(self._list_run_dirs()):
            checkpoint = d / "checkpoint.json"
            result = d / "result.json"
            if checkpoint.exists() and not result.exists():
                # Check config match if filters are given
                if dataset_path or model:
                    config_file = d / "config.json"
                    if config_file.exists():
                        config = _read_json(config_file)
                        if dataset_path and config.get("dataset_path") != str(dataset_path):
                            continue
                        if model:
                            opt_config = config.get("optimizer_config", {})
                            # Check if any of the CLI models match
                            models = opt_config.get("_cli_models", [])
                            if models and model not in models:
                                continue

                run_id = d.name.split("_")[0]
                return Run(run_id=run_id, run_dir=d)
        return None

    def list_runs(self) -> list[RunSummary]:
        """List all runs with summary info."""
        summaries = []
        for d in self._list_run_dirs():
            run_id = d.name.split("_")[0]
            config_file = d / "config.json"
            checkpoint_file = d / "checkpoint.json"
            result_file = d / "result.json"
            baseline_file = d / "baseline.json"

            config = _read_json(config_file) if config_file.exists() else {}
            opt_config = config.get("optimizer_config", {})

            # Determine status
            if result_file.exists():
                status = "completed"
            elif checkpoint_file.exists():
                status = "interrupted"
            else:
                status = "running"

            # Scores
            baseline_score = None
            if baseline_file.exists():
                baseline = _read_json(baseline_file)
                baseline_score = baseline.get("mean_score")

            best_score = None
            final_score = None
            iterations_completed = 0

            if checkpoint_file.exists():
                cp = _read_json(checkpoint_file)
                best_score = cp.get("best_score")
                iterations_completed = cp.get("completed_iterations", 0)

            duration_seconds = None
            if result_file.exists():
                result = _read_json(result_file)
                best_score = result.get("best_score")
                final_score = result.get("final", {}).get("mean_score")
                iterations_completed = len(result.get("iterations", []))
                duration_seconds = result.get("duration_seconds")

            raw_dataset = config.get("dataset_path", "")
            dataset_abs = str((self.root.parent / raw_dataset).resolve()) if raw_dataset else ""
            branched_from = config.get("branched_from")
            summaries.append(RunSummary(
                run_id=run_id,
                run_dir=str(d),
                timestamp=config.get("timestamp", ""),
                status=status,
                strategy=opt_config.get("strategy", ""),
                model=", ".join(opt_config.get("_cli_models", [])),
                dataset=dataset_abs,
                baseline_score=baseline_score,
                best_score=best_score,
                final_score=final_score,
                iterations_completed=iterations_completed,
                duration_seconds=duration_seconds,
                parent_run_id=branched_from.get("run_id") if branched_from else None,
                parent_iteration=branched_from.get("iteration") if branched_from else None,
                config=opt_config,
            ))
        return summaries


class Run:
    """Handle for a single run's directory. Provides read/write for all run artifacts."""

    def __init__(self, run_id: str, run_dir: Path):
        self.run_id = run_id
        self.run_dir = run_dir
        self.iterations_dir = run_dir / "iterations"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.json"

    @property
    def baseline_path(self) -> Path:
        return self.run_dir / "baseline.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoint.json"

    @property
    def result_path(self) -> Path:
        return self.run_dir / "result.json"

    @property
    def best_prompt_path(self) -> Path:
        return self.run_dir / "best_prompt.md"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "run.log.jsonl"

    @property
    def is_complete(self) -> bool:
        return self.result_path.exists()

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint_path.exists()

    def load_config(self) -> dict[str, Any]:
        """Load the run config."""
        return _read_json(self.config_path)

    def save_baseline(self, baseline: dict[str, Any]) -> None:
        """Save the baseline eval snapshot."""
        baseline["timestamp"] = _now_iso()
        _write_json(self.baseline_path, baseline)

    def load_baseline(self) -> dict[str, Any] | None:
        """Load the baseline eval snapshot, or None if not yet saved."""
        if self.baseline_path.exists():
            return _read_json(self.baseline_path)
        return None

    def save_iteration(self, state: IterationState) -> None:
        """Save one completed iteration and update the checkpoint.

        This is the core checkpointing operation — called after each
        iteration completes. Writes are ordered so that a crash at any
        point leaves the store in a consistent state:
          1. Write iteration file (atomic per-iteration)
          2. Update checkpoint.json (resume marker)
          3. Update best_prompt.md (never lose work)
        """
        if not state.timestamp:
            state.timestamp = _now_iso()

        # 1. Write iteration file
        iter_file = self.iterations_dir / f"{state.iteration:03d}.json"
        _write_json(iter_file, _iteration_to_dict(state))

    def save_checkpoint(self, checkpoint: CheckpointState) -> None:
        """Save the current checkpoint state (called after each iteration)."""
        if not checkpoint.timestamp:
            checkpoint.timestamp = _now_iso()
        _write_json(self.checkpoint_path, _checkpoint_to_dict(checkpoint))

        # Always keep best_prompt.md up to date
        self.best_prompt_path.write_text(checkpoint.best_prompt)

    def load_checkpoint(self) -> CheckpointState | None:
        """Load the checkpoint state for resuming, or None if no checkpoint."""
        if not self.checkpoint_path.exists():
            return None
        data = _read_json(self.checkpoint_path)
        return CheckpointState(
            run_id=data["run_id"],
            initial_prompt=data["initial_prompt"],
            current_prompt=data["current_prompt"],
            completed_iterations=data["completed_iterations"],
            best_prompt=data["best_prompt"],
            best_score=data["best_score"],
            score_trajectory=data.get("score_trajectory", []),
            previous_reasoning=data.get("previous_reasoning", ""),
            strategy_state=data.get("strategy_state", {}),
            baseline=data.get("baseline"),
            timestamp=data.get("timestamp", ""),
        )

    def load_iterations(self) -> list[IterationState]:
        """Load all completed iteration states in order."""
        iterations = []
        if not self.iterations_dir.exists():
            return iterations
        for f in sorted(self.iterations_dir.glob("*.json")):
            data = _read_json(f)
            iterations.append(IterationState(
                iteration=data["iteration"],
                system_prompt=data["system_prompt"],
                score=data["score"],
                scores_by_metric=data.get("scores_by_metric", {}),
                reasoning=data.get("reasoning", ""),
                eval_tokens=data.get("eval_tokens", 0),
                reasoning_tokens=data.get("reasoning_tokens", 0),
                change_summary=data.get("change_summary", ""),
                val_score=data.get("val_score"),
                is_full_eval=data.get("is_full_eval", False),
                timestamp=data.get("timestamp", ""),
            ))
        return iterations

    def save_log(self, entries: list[dict[str, Any]]) -> None:
        """Write the full run event log as newline-delimited JSON (run.log.jsonl).

        Each entry is one SSE event dict with an added ``ts`` field.
        The file is written atomically via a temp file so a mid-write crash
        never leaves a partial log.
        """
        lines = "\n".join(json.dumps(e, default=str) for e in entries)
        tmp = self.log_path.with_suffix(".tmp")
        tmp.write_text(lines + ("\n" if lines else ""), encoding="utf-8")
        tmp.rename(self.log_path)

    def load_log(self) -> list[dict[str, Any]]:
        """Load the run event log, or an empty list if not yet written."""
        if not self.log_path.exists():
            return []
        entries = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries

    def save_result(self, result: dict[str, Any]) -> None:
        """Save the final result. Marks the run as complete."""
        result["timestamp"] = _now_iso()
        result["run_id"] = self.run_id
        _write_json(self.result_path, result)

    def load_result(self) -> dict[str, Any] | None:
        """Load the final result, or None if not yet complete."""
        if self.result_path.exists():
            return _read_json(self.result_path)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic JSON write — write to temp file then rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.rename(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iteration_to_dict(state: IterationState) -> dict[str, Any]:
    d: dict[str, Any] = {
        "iteration": state.iteration,
        "system_prompt": state.system_prompt,
        "score": state.score,
        "scores_by_metric": state.scores_by_metric,
        "reasoning": state.reasoning,
        "eval_tokens": state.eval_tokens,
        "reasoning_tokens": state.reasoning_tokens,
        "timestamp": state.timestamp,
    }
    if state.val_score is not None:
        d["val_score"] = state.val_score
    if state.is_full_eval:
        d["is_full_eval"] = True
    return d


def _checkpoint_to_dict(state: CheckpointState) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "initial_prompt": state.initial_prompt,
        "current_prompt": state.current_prompt,
        "completed_iterations": state.completed_iterations,
        "best_prompt": state.best_prompt,
        "best_score": state.best_score,
        "score_trajectory": state.score_trajectory,
        "previous_reasoning": state.previous_reasoning,
        "strategy_state": state.strategy_state,
        "baseline": state.baseline,
        "timestamp": state.timestamp,
    }
