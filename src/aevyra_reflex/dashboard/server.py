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

"""Dashboard HTTP server — zero external dependencies.

Serves the React SPA and a JSON API backed by RunStore.
Also exposes /api/jobs endpoints so the UI can launch and monitor
optimization runs in background threads.
"""

from __future__ import annotations

import json
import logging
import queue as _queue_mod
import re
import threading
import time
import uuid
from dataclasses import asdict
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from aevyra_reflex.run_store import RunStore

logger = logging.getLogger(__name__)

# Path to the bundled SPA
_STATIC_DIR = Path(__file__).parent


def _now_utc_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ── Job registry ─────────────────────────────────────────────────────────────
# Each entry: { queue, cancelled (Event), done (bool), thread }
_active_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _run_job(job_id: str, config_dict: dict[str, Any], store: RunStore, job: dict[str, Any]) -> None:
    """Background thread: run the optimizer and push SSE events into job['queue']."""
    q: _queue_mod.Queue = job["queue"]
    cancelled: threading.Event = job["cancelled"]

    # Structured log: every event is buffered here so we can persist it after
    # the run completes.  Entries are plain dicts with an added ``ts`` field.
    _log_entries: list[dict[str, Any]] = []
    _started_at = time.monotonic()
    _started_iso = _now_utc_iso()

    def push(event_type: str, **kwargs: Any) -> None:
        entry = {"type": event_type, "ts": _now_utc_iso(), **kwargs}
        _log_entries.append(entry)
        q.put(entry)

    try:
        from aevyra_reflex.optimizer import OptimizerConfig, PromptOptimizer
        from aevyra_verdict import Dataset, BleuScore, ExactMatch, RougeScore

        metric_map = {"rouge": RougeScore, "bleu": BleuScore, "exact": ExactMatch}

        # ── Build config ──────────────────────────────────────────────────
        strategy = config_dict.get("strategy", "auto")
        max_iterations = int(config_dict.get("max_iterations", 10))
        score_threshold = float(config_dict.get("score_threshold", 0.85))
        target_model_label = None
        target_source = None

        # If a verdict results path was given, override the threshold
        verdict_path = config_dict.get("verdict_results_path", "").strip()
        target_models = [m.strip() for m in config_dict.get("target_models", []) if m.strip()]

        if verdict_path:
            try:
                from aevyra_reflex.optimizer import parse_verdict_results
                parsed = parse_verdict_results(verdict_path)
                score_threshold = parsed["best_score"]
                target_model_label = parsed["target_model"]
                target_source = "verdict_json"
                push("queued", message=f"Target set from verdict: {score_threshold:.4f} ({target_model_label})")
            except Exception as exc:
                push("error", message=f"Failed to read verdict results: {exc}")
                return

        # Reasoning model (optional — defaults to Claude inside OptimizerConfig)
        reasoning_raw = config_dict.get("reasoning_model") or ""
        reasoning_model = "claude-sonnet-4-20250514"
        reasoning_provider = None
        if reasoning_raw:
            parts = reasoning_raw.split("/", 1)
            if len(parts) == 2:
                reasoning_provider, reasoning_model = parts
            else:
                reasoning_model = reasoning_raw

        config = OptimizerConfig(
            strategy=strategy,
            max_iterations=max_iterations,
            score_threshold=score_threshold,
            reasoning_model=reasoning_model,
            reasoning_provider=reasoning_provider,
            target_model=target_model_label,
            target_source=target_source or ("manual" if not verdict_path and not target_models else None),
        )

        # ── Load dataset ──────────────────────────────────────────────────
        ds_path = config_dict["dataset_path"]
        try:
            ds = Dataset.from_jsonl(ds_path)
        except FileNotFoundError:
            push("error", message=f"Dataset not found: {ds_path}")
            return

        # ── Build optimizer ───────────────────────────────────────────────
        optimizer = PromptOptimizer(config=config)
        optimizer.set_dataset(ds)

        model_str = config_dict["model"]  # expected "provider/model"
        parts = model_str.split("/", 1)
        if len(parts) != 2:
            push("error", message=f"Model must be in 'provider/model' format, got: {model_str!r}")
            return
        optimizer.add_provider(parts[0], parts[1])

        judge_str = config_dict.get("judge", "").strip()
        if judge_str:
            from aevyra_verdict import LLMJudge
            from aevyra_verdict.providers import get_provider
            from aevyra_reflex.optimizer import _resolve_provider
            j_parts = judge_str.split("/", 1)
            if len(j_parts) != 2:
                push("error", message=f"Judge must be in provider/model format, got: {judge_str!r}")
                return
            resolved_j = _resolve_provider(j_parts[0], j_parts[1])
            j_kwargs = {}
            if resolved_j.get("api_key"):
                j_kwargs["api_key"] = resolved_j["api_key"]
            if resolved_j.get("base_url"):
                j_kwargs["base_url"] = resolved_j["base_url"]
            judge_provider = get_provider(resolved_j["provider_name"], resolved_j["model"], **j_kwargs)
            judge_criteria = config_dict.get("judge_criteria", "").strip() or None
            judge_dimensions = [d.strip() for d in config_dict.get("judge_dimensions", []) if d.strip()] or None
            optimizer.add_metric(LLMJudge(
                judge_provider=judge_provider,
                criteria=judge_criteria,
                dimensions=judge_dimensions,
            ))
        else:
            for m_name in config_dict.get("metrics", ["rouge"]):
                cls = metric_map.get(m_name.lower())
                if cls:
                    optimizer.add_metric(cls())

        # Auto threshold — benchmark target models live
        if target_models:
            from aevyra_reflex.optimizer import _resolve_provider
            push("queued", message=f"Benchmarking {len(target_models)} target model(s) to set threshold…")
            target_providers = []
            for tm in target_models:
                t_parts = tm.split("/", 1)
                if len(t_parts) != 2:
                    push("error", message=f"Target model must be in provider/model format, got: {tm!r}")
                    return
                resolved_t = _resolve_provider(t_parts[0], t_parts[1])
                resolved_t["label"] = tm
                target_providers.append(resolved_t)

            all_providers = list(optimizer._providers) + target_providers
            try:
                benchmark = optimizer.benchmark_and_set_target(
                    config_dict["prompt"], all_providers,
                )
                score_threshold = config.score_threshold  # updated by benchmark_and_set_target
                target_model_label = config.target_model
                target_source = "verdict_run"
                scores_str = ", ".join(
                    f"{lbl}: {sc:.4f}" for lbl, sc in
                    sorted(benchmark["model_scores"].items(), key=lambda x: -x[1])
                )
                push("queued", message=f"Benchmark complete — {scores_str}. Target: {score_threshold:.4f} ({target_model_label})")
            except Exception as exc:
                push("error", message=f"Benchmark failed: {exc}")
                return

        # ── Branch metadata (set when this job was started via /api/jobs/branch) ──
        branch_meta = config_dict.get("_branch")
        baseline_override = None
        branched_from = None
        if branch_meta:
            from aevyra_reflex.result import EvalSnapshot
            parent_bl = branch_meta["baseline"]
            baseline_override = EvalSnapshot(
                mean_score=parent_bl["mean_score"],
                scores_by_metric=parent_bl.get("scores_by_metric", {}),
            )
            branched_from = {
                "run_id": branch_meta["parent_run_id"],
                "iteration": branch_meta["parent_iteration"],
            }
            push("queued", message=(
                f"Branch from run {branch_meta['parent_run_id']} "
                f"iter {branch_meta['parent_iteration']} — "
                f"reusing baseline {parent_bl['mean_score']:.4f}"
            ))
        else:
            push("queued", message="Starting baseline evaluation…")

        # ── Iteration callback (also checks cancellation) ─────────────────
        def on_iteration(record: Any) -> None:
            if cancelled.is_set():
                raise RuntimeError("__cancelled__")
            push(
                "iteration",
                iteration=record.iteration,
                score=record.score,
                scores_by_metric=getattr(record, "scores_by_metric", {}),
                system_prompt=getattr(record, "system_prompt", ""),
                reasoning=getattr(record, "reasoning", ""),
                eval_tokens=getattr(record, "eval_tokens", 0),
                reasoning_tokens=getattr(record, "reasoning_tokens", 0),
                change_summary=getattr(record, "change_summary", ""),
            )

        result = optimizer.run(
            config_dict["prompt"],
            on_iteration=on_iteration,
            run_store=store,
            baseline_override=baseline_override,
            branched_from=branched_from,
        )

        # Grab the run_id from the latest entry in the store
        latest = store.get_latest_run()
        actual_run_id = latest.run_id if latest else None

        push(
            "complete",
            run_id=actual_run_id,
            best_score=result.best_score,
            baseline=result.baseline,
            improvement_pct=result.improvement_pct,
            best_prompt=result.best_prompt,
        )

    except RuntimeError as exc:
        if "__cancelled__" in str(exc):
            push("cancelled", message="Run was cancelled.")
        else:
            push("error", message=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job %s failed", job_id)
        push("error", message=str(exc))
    finally:
        duration = round(time.monotonic() - _started_at, 2)
        job["done"] = True
        job["duration_seconds"] = duration
        job["started_at"] = _started_iso

        # Persist structured log + duration to the run directory
        try:
            run = store.get_latest_run()
            if run is not None:
                run.save_log(_log_entries)
                # Patch duration into result.json if the run completed
                if run.result_path.exists():
                    result_data = json.loads(run.result_path.read_text())
                    result_data["duration_seconds"] = duration
                    result_data["started_at"] = _started_iso
                    run.result_path.write_text(
                        json.dumps(result_data, indent=2, default=str)
                    )
        except Exception:  # noqa: BLE001
            logger.exception("Job %s: failed to persist run log", job_id)


# ── Request handler ───────────────────────────────────────────────────────────

class _DashboardHandler(BaseHTTPRequestHandler):
    """Lightweight request handler with JSON API routes."""

    store: RunStore  # injected via subclass

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]

        # ── Read-only run API ──────────────────────────────────────────
        if path == "/api/runs":
            return self._api_list_runs()
        m = re.match(r"^/api/runs/(\w+)$", path)
        if m:
            return self._api_get_run(m.group(1))
        m = re.match(r"^/api/runs/(\w+)/iterations$", path)
        if m:
            return self._api_get_iterations(m.group(1))
        m = re.match(r"^/api/runs/(\w+)/iterations/(\w+)$", path)
        if m:
            return self._api_get_iteration(m.group(1), m.group(2))

        # ── Job SSE stream ─────────────────────────────────────────────
        m = re.match(r"^/api/jobs/(\w+)/stream$", path)
        if m:
            return self._api_stream_job(m.group(1))

        # ── SPA — serve app.html for everything else ───────────────────
        return self._serve_spa()

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]

        if path == "/api/jobs/start":
            return self._api_start_job()

        if path == "/api/jobs/branch":
            return self._api_branch_job()

        m = re.match(r"^/api/jobs/(\w+)/cancel$", path)
        if m:
            return self._api_cancel_job(m.group(1))

        self._json_error(404, "Not found")

    # ------------------------------------------------------------------ #
    # Read-only run API
    # ------------------------------------------------------------------ #

    def _api_list_runs(self) -> None:
        summaries = self.store.list_runs()
        data = [asdict(s) for s in summaries]
        self._json_response(data)

    def _api_get_run(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        if run is None:
            return self._json_error(404, f"Run {run_id} not found")

        config = run.load_config() if run.config_path.exists() else {}
        baseline = run.load_baseline()
        checkpoint = run.load_checkpoint()
        result = run.load_result()
        iterations = run.load_iterations()
        log_entries = run.load_log()

        data: dict[str, Any] = {
            "run_id": run.run_id,
            "run_dir": str(run.run_dir),
            "is_complete": run.is_complete,
            "config": config,
            "baseline": baseline,
            "iterations": [
                {
                    "iteration": it.iteration,
                    "score": it.score,
                    "scores_by_metric": it.scores_by_metric,
                    "reasoning": it.reasoning,
                    "system_prompt": it.system_prompt,
                    "eval_tokens": it.eval_tokens,
                    "reasoning_tokens": it.reasoning_tokens,
                    "change_summary": getattr(it, "change_summary", ""),
                    "timestamp": it.timestamp,
                }
                for it in iterations
            ],
            "result": result,
            "log": log_entries,
            "duration_seconds": (result or {}).get("duration_seconds"),
            "started_at": (result or {}).get("started_at") or config.get("timestamp"),
        }
        if checkpoint:
            data["checkpoint"] = {
                "completed_iterations": checkpoint.completed_iterations,
                "best_score": checkpoint.best_score,
                "score_trajectory": checkpoint.score_trajectory,
                "best_prompt": checkpoint.best_prompt,
            }

        self._json_response(data)

    def _api_get_iterations(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        if run is None:
            return self._json_error(404, f"Run {run_id} not found")
        iterations = run.load_iterations()
        data = [
            {
                "iteration": it.iteration,
                "score": it.score,
                "scores_by_metric": it.scores_by_metric,
                "reasoning": it.reasoning,
                "system_prompt": it.system_prompt,
                "eval_tokens": it.eval_tokens,
                "reasoning_tokens": it.reasoning_tokens,
                "timestamp": it.timestamp,
            }
            for it in iterations
        ]
        self._json_response(data)

    def _api_get_iteration(self, run_id: str, iter_id: str) -> None:
        run = self.store.get_run(run_id)
        if run is None:
            return self._json_error(404, f"Run {run_id} not found")
        iterations = run.load_iterations()
        try:
            idx = int(iter_id) - 1  # iterations are 1-indexed
        except ValueError:
            return self._json_error(400, f"Invalid iteration: {iter_id}")
        if idx < 0 or idx >= len(iterations):
            return self._json_error(404, f"Iteration {iter_id} not found")
        it = iterations[idx]
        self._json_response({
            "iteration": it.iteration,
            "score": it.score,
            "scores_by_metric": it.scores_by_metric,
            "reasoning": it.reasoning,
            "system_prompt": it.system_prompt,
            "eval_tokens": it.eval_tokens,
            "reasoning_tokens": it.reasoning_tokens,
            "timestamp": it.timestamp,
        })

    # ------------------------------------------------------------------ #
    # Job API (start / stream / cancel)
    # ------------------------------------------------------------------ #

    def _api_start_job(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            config_dict = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self._json_error(400, "Invalid JSON body")

        for field in ("dataset_path", "prompt", "model"):
            if not config_dict.get(field, "").strip():
                return self._json_error(400, f"Missing required field: {field}")

        job_id = uuid.uuid4().hex[:12]
        job: dict[str, Any] = {
            "queue": _queue_mod.Queue(),
            "cancelled": threading.Event(),
            "done": False,
        }
        with _jobs_lock:
            _active_jobs[job_id] = job

        t = threading.Thread(
            target=_run_job,
            args=(job_id, config_dict, self.store, job),
            daemon=True,
            name=f"reflex-job-{job_id}",
        )
        job["thread"] = t
        t.start()

        self._json_response({"job_id": job_id}, status=202)

    def _api_branch_job(self) -> None:
        """POST /api/jobs/branch — start a new run branched from an existing one.

        Expected body::

            {
                "parent_run_id": "003",
                "parent_iteration": 5,
                "strategy": "fewshot",
                "max_iterations": 10
            }

        The branch run starts from the prompt at the given iteration, reuses the
        parent's baseline score (no re-eval), and creates a new run in the store
        with ``branched_from`` set in its config.
        """
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return self._json_error(400, "Invalid JSON body")

        parent_run_id = req.get("parent_run_id", "").strip()
        parent_iteration = req.get("parent_iteration")
        strategy = req.get("strategy", "iterative")
        max_iterations = int(req.get("max_iterations", 10))

        if not parent_run_id:
            return self._json_error(400, "Missing required field: parent_run_id")
        if parent_iteration is None:
            return self._json_error(400, "Missing required field: parent_iteration")

        # Load parent run
        parent_run = self.store.get_run(parent_run_id)
        if parent_run is None:
            return self._json_error(404, f"Run {parent_run_id} not found")

        # Get the prompt at the requested iteration
        iterations = parent_run.load_iterations()
        target = next((it for it in iterations if it.iteration == int(parent_iteration)), None)
        if target is None:
            return self._json_error(404, f"Iteration {parent_iteration} not found in run {parent_run_id}")

        # Load parent baseline
        parent_baseline = parent_run.load_baseline()
        if parent_baseline is None:
            return self._json_error(400, f"Run {parent_run_id} has no baseline — cannot branch")

        # Load parent config to inherit model/dataset/metrics/reasoning settings
        parent_config = parent_run.load_config()
        opt_config = parent_config.get("optimizer_config", {})

        # Build a merged config_dict for the branch job — override strategy + iters
        branch_config = {
            "dataset_path": parent_config.get("dataset_path", ""),
            "prompt": target.system_prompt,
            "model": (opt_config.get("_cli_models") or [""])[0],
            "strategy": strategy,
            "max_iterations": max_iterations,
            "score_threshold": opt_config.get("score_threshold", 0.85),
            "reasoning_model": opt_config.get("reasoning_model", ""),
            "metrics": opt_config.get("metrics", ["rouge"]),
            "judge": opt_config.get("judge", ""),
            "judge_criteria": opt_config.get("judge_criteria", ""),
            "judge_dimensions": opt_config.get("judge_dimensions", []),
            # Branch metadata — passed through to optimizer.run()
            "_branch": {
                "parent_run_id": parent_run_id,
                "parent_iteration": int(parent_iteration),
                "baseline": parent_baseline,
            },
        }

        job_id = uuid.uuid4().hex[:12]
        job: dict[str, Any] = {
            "queue": _queue_mod.Queue(),
            "cancelled": threading.Event(),
            "done": False,
        }
        with _jobs_lock:
            _active_jobs[job_id] = job

        t = threading.Thread(
            target=_run_job,
            args=(job_id, branch_config, self.store, job),
            daemon=True,
            name=f"reflex-branch-{job_id}",
        )
        job["thread"] = t
        t.start()

        self._json_response({"job_id": job_id}, status=202)

    def _api_stream_job(self, job_id: str) -> None:
        job = _active_jobs.get(job_id)
        if job is None:
            return self._json_error(404, f"No active job: {job_id}")

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q: _queue_mod.Queue = job["queue"]
        while True:
            try:
                event = q.get(timeout=15)
                data = json.dumps(event, default=str)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
                if event.get("type") in ("complete", "error", "cancelled"):
                    break
            except _queue_mod.Empty:
                # Heartbeat — keeps the connection alive through proxies / browsers
                self.wfile.write(b": heartbeat\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def _api_cancel_job(self, job_id: str) -> None:
        job = _active_jobs.get(job_id)
        if job is None:
            return self._json_error(404, f"No active job: {job_id}")
        job["cancelled"].set()
        self._json_response({"status": "cancelling"})

    # ------------------------------------------------------------------ #
    # Static serving
    # ------------------------------------------------------------------ #

    def _serve_spa(self) -> None:
        spa_path = _STATIC_DIR / "app.html"
        if not spa_path.exists():
            self._json_error(500, "Dashboard frontend not found")
            return
        content = spa_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, status: int, message: str) -> None:
        self._json_response({"error": message}, status=status)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default stderr logging; use Python logger instead
        logger.debug(format, *args)


def serve(
    run_dir: str | Path = ".reflex",
    host: str = "127.0.0.1",
    port: int = 8337,
    open_browser: bool = True,
) -> None:
    """Start the dashboard server.

    Args:
        run_dir: Path to the .reflex/ directory.
        host: Bind address.
        port: Port to listen on.
        open_browser: Open the dashboard in the default browser.
    """
    store = RunStore(root=run_dir)

    # Inject store into handler class (ThreadingHTTPServer so SSE never blocks)
    handler_class = type(
        "_Handler",
        (_DashboardHandler,),
        {"store": store},
    )

    server = ThreadingHTTPServer((host, port), handler_class)
    url = f"http://{host}:{port}"

    print(f"Reflex dashboard running at {url}")
    print(f"Run directory: {run_dir}")
    print("Press Ctrl+C to stop.\n")

    if open_browser:
        import webbrowser
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard.")
        server.shutdown()
