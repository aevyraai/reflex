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

"""aevyra-reflex CLI — one command to optimize a system prompt."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated, Any, Optional

try:
    import typer
except ImportError:
    print("typer is required for the CLI. Install it with: pip install typer")
    sys.exit(1)

def _version_callback(value: bool) -> None:
    if value:
        typer.echo("aevyra-reflex 0.1.0")
        raise typer.Exit()

app = typer.Typer(
    name="aevyra-reflex",
    help="Agentic prompt optimization — diagnose eval failures, rewrite prompts, hit the target score.",
    no_args_is_help=True,
)

@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show version and exit.", callback=_version_callback, is_eager=True),
    ] = False,
) -> None:
    """aevyra-reflex — agentic prompt optimization."""


@app.command()
def optimize(
    dataset: Annotated[Path, typer.Argument(help="Path to a JSONL eval dataset.")],
    prompt: Annotated[Path, typer.Argument(help="Path to the initial system prompt (.md).")],
    model: Annotated[
        list[str],
        typer.Option("-m", "--model", help="Model to optimize, in 'provider/model' format."),
    ] = [],
    target: Annotated[
        list[str],
        typer.Option("--target", help="Target model(s) to benchmark against. The best score becomes the threshold."),
    ] = [],
    verdict_results: Annotated[
        Optional[Path],
        typer.Option("--verdict-results", help="Path to verdict results JSON. Sets the threshold from the best model's score."),
    ] = None,
    metric: Annotated[
        list[str],
        typer.Option("--metric", help="Built-in metrics: rouge, bleu, exact. Defaults to rouge unless --judge is set."),
    ] = [],
    judge: Annotated[
        Optional[str],
        typer.Option("--judge", help="LLM judge in 'provider/model' format. Can be used alone or with --metric."),
    ] = None,
    strategy: Annotated[
        str,
        typer.Option("-s", "--strategy", help="Strategy: 'auto' (default), 'iterative', 'pdo', 'fewshot', or 'structural'."),
    ] = "auto",
    reasoning_model: Annotated[
        Optional[str],
        typer.Option("--reasoning-model", help="LLM for reasoning in 'provider/model' format. Defaults to Claude Sonnet. Use 'ollama/llama3.1:70b' for local."),
    ] = None,
    reasoning_api_key: Annotated[
        Optional[str],
        typer.Option("--reasoning-api-key", help="API key for the reasoning model.", envvar="REFLEX_REASONING_API_KEY"),
    ] = None,
    reasoning_base_url: Annotated[
        Optional[str],
        typer.Option("--reasoning-base-url", help="Base URL for the reasoning model (for self-hosted endpoints)."),
    ] = None,
    max_iterations: Annotated[
        int,
        typer.Option("--max-iterations", help="Maximum optimization iterations/rounds."),
    ] = 10,
    threshold: Annotated[
        Optional[float],
        typer.Option("--threshold", help="Score threshold to stop early (0.0–1.0). Overrides --verdict-results and --target."),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("-o", "--output", help="Save the best prompt to this file."),
    ] = None,
    results_json: Annotated[
        Optional[Path],
        typer.Option("--results-json", help="Save full results (all iterations) to JSON."),
    ] = None,
    max_workers: Annotated[
        int,
        typer.Option("--max-workers", help="Max parallel threads for variant evals/duels. For Ollama, match to OLLAMA_NUM_PARALLEL."),
    ] = 4,
    run_dir: Annotated[
        Optional[Path],
        typer.Option("--run-dir", help="Directory for run history. Defaults to .reflex/ in cwd."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume the latest interrupted run for this dataset/model."),
    ] = False,
    resume_from: Annotated[
        Optional[str],
        typer.Option("--resume-from", help="Resume a specific run by ID (e.g. '001') or path."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Show debug output."),
    ] = False,
) -> None:
    """Optimize a system prompt: baseline eval → optimize → verify → show results.

    The score threshold is set from the best available reference, in priority order:
    --threshold (explicit) > --verdict-results (from file) > --target (live benchmark).
    If none are given, defaults to 0.85.

    \b
    Examples:
      # Optimize llama to match gpt-4o-mini's score (live benchmark)
      aevyra-reflex optimize data.jsonl prompt.md -m local/llama3.1 --target openai/gpt-4o-mini

      # Use existing verdict results as the target
      aevyra-reflex optimize data.jsonl prompt.md -m local/llama3.1 --verdict-results results.json

      # Multiple targets — best score wins
      aevyra-reflex optimize data.jsonl prompt.md -m local/llama3.1 \\
        --target openai/gpt-4o-mini --target openai/gpt-4o

      # Explicit threshold (ignores --target and --verdict-results)
      aevyra-reflex optimize data.jsonl prompt.md -m local/llama3.1 --threshold 0.90
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
    )
    # Silence noisy third-party HTTP/network loggers that flood verbose output
    for _noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3", "requests"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    from aevyra_reflex.optimizer import OptimizerConfig, PromptOptimizer, _resolve_provider

    # Validate inputs
    if not dataset.exists():
        typer.echo(f"Error: dataset file not found: {dataset}", err=True)
        raise typer.Exit(code=1)
    if not prompt.exists():
        typer.echo(f"Error: prompt file not found: {prompt}", err=True)
        raise typer.Exit(code=1)
    if not model:
        typer.echo("Error: at least one --model is required.", err=True)
        raise typer.Exit(code=1)
    if verdict_results and target:
        typer.echo("Error: use --verdict-results or --target, not both.", err=True)
        raise typer.Exit(code=1)

    # Load
    from aevyra_verdict import Dataset
    ds = Dataset.from_jsonl(str(dataset))
    initial_prompt = prompt.read_text().strip()

    # Resolve the score threshold.
    # Priority: --threshold > --verdict-results > --target > default 0.85
    effective_threshold = 0.85
    target_source = None
    target_model_label = None

    if threshold is not None:
        effective_threshold = threshold
        target_source = "manual"
    elif verdict_results:
        if not verdict_results.exists():
            typer.echo(f"Error: verdict results file not found: {verdict_results}", err=True)
            raise typer.Exit(code=1)
        from aevyra_reflex.optimizer import parse_verdict_results
        parsed = parse_verdict_results(verdict_results)
        effective_threshold = parsed["best_score"]
        target_model_label = parsed["target_model"]
        target_source = "verdict_json"
        typer.echo(f"Target from verdict: {effective_threshold:.4f} ({target_model_label})")
    # --target models: we'll benchmark after building metrics (need metrics first)

    # Resolve reasoning model provider/name from --reasoning-model
    resolved_reasoning_model = "claude-sonnet-4-20250514"
    resolved_reasoning_provider = None

    if reasoning_model:
        parts = reasoning_model.split("/", 1)
        if len(parts) == 2:
            resolved_reasoning_provider = parts[0]
            resolved_reasoning_model = parts[1]
        else:
            # No slash — treat as a bare model name (Ollama or heuristic)
            resolved_reasoning_model = reasoning_model

    # Build optimizer config
    config = OptimizerConfig(
        max_iterations=max_iterations,
        score_threshold=effective_threshold,
        strategy=strategy,
        reasoning_model=resolved_reasoning_model,
        reasoning_provider=resolved_reasoning_provider,
        reasoning_api_key=reasoning_api_key,
        reasoning_base_url=reasoning_base_url,
        max_workers=max_workers,
        target_model=target_model_label,
        target_source=target_source,
    )
    optimizer = PromptOptimizer(config=config)
    optimizer.set_dataset(ds)

    for m in model:
        parts = m.split("/", 1)
        if len(parts) != 2:
            typer.echo(f"Error: model must be in 'provider/model' format, got: {m}", err=True)
            raise typer.Exit(code=1)
        optimizer.add_provider(parts[0], parts[1])

    _add_metrics(optimizer, metric, judge)

    # If --target models were given, benchmark them now to set the threshold
    if target and threshold is None and not verdict_results:
        target_providers = []
        for t in target:
            parts = t.split("/", 1)
            if len(parts) != 2:
                typer.echo(f"Error: --target must be in 'provider/model' format, got: {t}", err=True)
                raise typer.Exit(code=1)
            resolved = _resolve_provider(parts[0], parts[1])
            resolved["label"] = t
            target_providers.append(resolved)

        # Also include the model-to-optimize so we can see the gap
        all_providers = list(optimizer._providers) + target_providers

        typer.echo("=" * 52)
        typer.echo("  Benchmarking models to set target score")
        typer.echo("=" * 52)
        typer.echo()

        benchmark = optimizer.benchmark_and_set_target(
            initial_prompt, all_providers,
        )

        typer.echo()
        typer.echo("  Model scores:")
        for label, score in sorted(
            benchmark["model_scores"].items(), key=lambda x: -x[1]
        ):
            marker = " ← target" if label == benchmark["best_model"] else ""
            typer.echo(f"    {label:40s} {score:.4f}{marker}")
        typer.echo()

        effective_threshold = config.score_threshold
        target_model_label = config.target_model

    # Banner
    metric_display = ', '.join(metric) if metric else 'rouge'
    if judge:
        metric_display += ' + LLM judge' if metric_display != 'rouge' else 'LLM judge'

    threshold_display = f"{effective_threshold:.4f}"
    if target_model_label:
        threshold_display += f" (from {target_model_label})"

    typer.echo("=" * 52)
    typer.echo("  aevyra-reflex")
    typer.echo("=" * 52)
    typer.echo(f"  Dataset    : {dataset.name} ({len(ds.conversations)} samples)")
    typer.echo(f"  Model(s)   : {', '.join(model)}")
    typer.echo(f"  Strategy   : {strategy}")
    typer.echo(f"  Metrics    : {metric_display}")
    reasoning_display = reasoning_model or "claude-sonnet-4-20250514"
    typer.echo(f"  Reasoning  : {reasoning_display}")
    typer.echo(f"  Target     : {threshold_display}")
    typer.echo(f"  Workers    : {max_workers}")
    typer.echo("=" * 52)
    typer.echo()

    # Set up run persistence
    from aevyra_reflex.run_store import RunStore

    store = RunStore(root=run_dir or ".reflex")
    resume_run = None

    if resume_from:
        # Try as run ID first, then as path
        resume_run = store.get_run(resume_from)
        if resume_run is None:
            # Try as a path
            resume_path = Path(resume_from)
            if resume_path.exists() and resume_path.is_dir():
                run_id = resume_path.name.split("_")[0]
                from aevyra_reflex.run_store import Run
                resume_run = Run(run_id=run_id, run_dir=resume_path)
        if resume_run is None:
            typer.echo(f"Error: run not found: {resume_from}", err=True)
            raise typer.Exit(code=1)
        if resume_run.is_complete:
            typer.echo(f"Run {resume_from} already completed. Start a new run instead.", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"Resuming run {resume_run.run_id}...")
    elif resume:
        resume_run = store.find_incomplete_run(
            dataset_path=str(dataset),
        )
        if resume_run is None:
            typer.echo("No interrupted run found. Starting a new run.")
        else:
            cp = resume_run.load_checkpoint()
            if cp:
                typer.echo(
                    f"Resuming run {resume_run.run_id} from iteration "
                    f"{cp.completed_iterations} (best: {cp.best_score:.4f})"
                )

    # Run the full pipeline: baseline → optimize → verify
    typer.echo("Step 1/3  Running baseline eval...")
    typer.echo()

    result = optimizer.run(
        initial_prompt,
        on_iteration=_make_progress_cb(typer),
        run_store=store,
        resume_run=resume_run,
    )

    # Show results
    typer.echo()
    typer.echo(result.summary())
    typer.echo()

    # Save outputs
    if output:
        result.save_best_prompt(output)
        typer.echo(f"Best prompt saved to: {output}")

    if results_json:
        result.to_json(results_json)
        typer.echo(f"Full results saved to: {results_json}")

    if not output:
        typer.echo("Optimized prompt:")
        typer.echo("-" * 52)
        # Truncate very long prompts in terminal output
        prompt_text = result.best_prompt
        if len(prompt_text) > 2000:
            prompt_text = prompt_text[:2000] + "\n... (truncated, use -o to save full prompt)"
        typer.echo(prompt_text)
        typer.echo("-" * 52)

    # Show run directory
    typer.echo()
    typer.echo(f"Run history: {store.root}/runs/")


@app.command()
def runs(
    run_dir: Annotated[
        Optional[Path],
        typer.Option("--run-dir", help="Directory for run history. Defaults to .reflex/ in cwd."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("-v", "--verbose", help="Show config details for each run."),
    ] = False,
) -> None:
    """List all optimization runs and their status.

    \b
    Example output:
       ID  Status        Strategy      Iters  Baseline      Best     Final  Duration  Dataset
    ──────────────────────────────────────────────────────────────────────────────────────────
      001  ✓ completed   auto              8    0.6102    0.7934    0.7891    4m 12s  eval.jsonl
      002  ⚡ interrupted  iterative        3    0.5890    0.6441        —        —    eval.jsonl
      003  ✓ completed   pdo              10    0.6240    0.8102    0.8055    6m 03s  eval.jsonl
    """
    from aevyra_reflex.run_store import RunStore

    store = RunStore(root=run_dir or ".reflex")
    all_runs = store.list_runs()

    if not all_runs:
        typer.echo("No runs found. Run 'aevyra-reflex optimize' to get started.")
        raise typer.Exit()

    typer.echo(f"{'ID':>4s}  {'Status':14s}  {'Strategy':12s}  {'Iters':>5s}  {'Baseline':>8s}  {'Best':>8s}  {'Final':>8s}  {'Duration':>8s}  {'Dataset'}")
    typer.echo("─" * 100)

    for r in all_runs:
        baseline_str = f"{r.baseline_score:.4f}" if r.baseline_score is not None else "—"
        best_str = f"{r.best_score:.4f}" if r.best_score is not None else "—"
        final_str = f"{r.final_score:.4f}" if r.final_score is not None else "—"

        status_icon = {"completed": "✓", "interrupted": "⚡", "running": "…"}.get(r.status, "?")
        status_display = f"{status_icon} {r.status}"

        if r.duration_seconds is not None:
            m, s = divmod(r.duration_seconds, 60)
            dur_str = f"{int(m)}m {int(s):02d}s" if m else f"{r.duration_seconds:.1f}s"
        else:
            dur_str = "—"

        dataset_name = Path(r.dataset).name if r.dataset else "—"

        typer.echo(
            f"{r.run_id:>4s}  {status_display:14s}  {r.strategy:12s}  "
            f"{r.iterations_completed:>5d}  {baseline_str:>8s}  {best_str:>8s}  "
            f"{final_str:>8s}  {dur_str:>8s}  {dataset_name}"
        )

        if verbose and r.config:
            reasoning = r.config.get("reasoning_model", "")
            models = r.config.get("_cli_models", [])
            if reasoning:
                typer.echo(f"       reasoning: {reasoning}")
            if models:
                typer.echo(f"       models: {', '.join(models)}")

    typer.echo()
    typer.echo(f"{len(all_runs)} run(s) in {store.root}/runs/")

    # Hint about resume if there's an interrupted run
    interrupted = [r for r in all_runs if r.status == "interrupted"]
    if interrupted:
        latest = interrupted[-1]
        typer.echo(f"\nTo resume run {latest.run_id}: aevyra-reflex optimize ... --resume")


@app.command()
def logs(
    run_id: Annotated[
        Optional[str],
        typer.Argument(help="Run ID to inspect (e.g. '001'). Defaults to the latest run."),
    ] = None,
    run_dir: Annotated[
        Optional[Path],
        typer.Option("--run-dir", help="Directory for run history. Defaults to .reflex/ in cwd."),
    ] = None,
    raw: Annotated[
        bool,
        typer.Option("--raw", help="Print raw JSONL (one event per line, suitable for piping/grep)."),
    ] = False,
) -> None:
    """Print the structured event log for a run.

    \b
    Examples:
      aevyra-reflex logs              # latest run (e.g. 003)
      aevyra-reflex logs 001          # specific run
      aevyra-reflex logs 002          # interrupted run — shows events up to crash
      aevyra-reflex logs --raw        # raw JSONL, one event per line
      aevyra-reflex logs --raw | grep '"type": "iteration"'

    \b
    Example output (aevyra-reflex logs 003):
      Run 003  ✓ completed  duration: 6m 3.0s
      Started: 2026-04-06T10:44:01+00:00
      ────────────────────────────────────────────────────────────
        10:44:01  [queued]     Benchmarking 2 target model(s) to set threshold…
        10:44:14  [queued]     Benchmark complete — openai/gpt-4o-mini: 0.7821,
                               anthropic/claude-haiku-4-5-20251001: 0.7340.
                               Target: 0.7821 (openai/gpt-4o-mini)
        10:44:14  [queued]     Starting baseline evaluation…
        10:44:22  [iteration]  iteration  1  score: 0.6240
        10:44:35  [iteration]  iteration  2  score: 0.6891
        10:44:49  [iteration]  iteration  3  score: 0.7102
        10:45:03  [iteration]  iteration  4  score: 0.7340
        10:45:18  [iteration]  iteration  5  score: 0.7558
        10:45:32  [iteration]  iteration  6  score: 0.7712
        10:45:47  [iteration]  iteration  7  score: 0.7890
        10:46:01  [iteration]  iteration  8  score: 0.7934
        10:46:15  [iteration]  iteration  9  score: 0.7988
        10:47:04  [iteration]  iteration 10  score: 0.8102
        10:50:04  [complete]   best score: 0.8055
      ────────────────────────────────────────────────────────────
        12 events  •  .reflex/runs/003_2026-04-06T10-44-01
    """
    from aevyra_reflex.run_store import RunStore
    import json as _json

    store = RunStore(root=run_dir or ".reflex")

    if run_id:
        run = store.get_run(run_id)
        if run is None:
            typer.echo(f"Error: run '{run_id}' not found.", err=True)
            raise typer.Exit(code=1)
    else:
        run = store.get_latest_run()
        if run is None:
            typer.echo("No runs found. Run 'aevyra-reflex optimize' to get started.")
            raise typer.Exit()

    entries = run.load_log()
    if not entries:
        typer.echo(f"Run {run.run_id} has no event log — logs are only available for runs")
        typer.echo("started after structured logging was added. Start a new run to see logs.")
        raise typer.Exit()

    if raw:
        for entry in entries:
            typer.echo(_json.dumps(entry))
        raise typer.Exit()

    # Pretty-print header
    result = run.load_result()
    duration = result.get("duration_seconds") if result else None
    started_at = (result or {}).get("started_at") or ""
    dur_str = ""
    if duration is not None:
        m, s = divmod(duration, 60)
        dur_str = f"  duration: {int(m)}m {s:.1f}s" if m else f"  duration: {s:.1f}s"

    typer.echo(f"\nRun {run.run_id}  {'✓ completed' if run.is_complete else '⚡ incomplete'}{dur_str}")
    if started_at:
        typer.echo(f"Started: {started_at}")
    typer.echo("─" * 60)

    EVENT_COLORS = {
        "queued":    typer.colors.CYAN,
        "iteration": typer.colors.GREEN,
        "complete":  typer.colors.BRIGHT_GREEN,
        "error":     typer.colors.RED,
        "cancelled": typer.colors.YELLOW,
    }

    for entry in entries:
        etype = entry.get("type", "?")
        ts_raw = entry.get("ts", "")
        # Format timestamp as HH:MM:SS if parseable
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(ts_raw)
            ts = dt.astimezone().strftime("%H:%M:%S")
        except Exception:
            ts = ts_raw[:19] if ts_raw else ""

        color = EVENT_COLORS.get(etype, typer.colors.WHITE)
        tag = typer.style(f"[{etype}]", fg=color, bold=True)

        if etype == "iteration":
            score = entry.get("score")
            iteration = entry.get("iteration")
            msg = f"iteration {iteration:>2d}  score: {score:.4f}" if score is not None else f"iteration {iteration}"
            # Show token counts if present
            eval_tok = entry.get("eval_tokens", 0)
            reason_tok = entry.get("reasoning_tokens", 0)
            if eval_tok or reason_tok:
                def _fmt_tok(n): return f"{n/1e6:.2f}M" if n >= 1_000_000 else (f"{n/1000:.1f}K" if n >= 1000 else str(n))
                tok_str = f"  eval: {_fmt_tok(eval_tok)}  reason: {_fmt_tok(reason_tok)}"
                msg += typer.style(tok_str, fg=typer.colors.BRIGHT_BLACK)
            # Show per-dimension scores if present
            sbm = entry.get("scores_by_metric", {})
            if sbm and len(sbm) > 1:
                dims = "  " + "  ".join(f"{k}: {v:.3f}" for k, v in sbm.items())
                msg += typer.style(dims, fg=typer.colors.BRIGHT_BLACK)
        else:
            msg = entry.get("message", "")
            if not msg and etype == "complete":
                best = entry.get("best_score")
                msg = f"best score: {best:.4f}" if best is not None else "done"

        typer.echo(f"  {ts}  {tag}  {msg}")

    # Token totals footer
    total_eval = sum(e.get("eval_tokens", 0) for e in entries)
    total_reason = sum(e.get("reasoning_tokens", 0) for e in entries)

    typer.echo("─" * 60)
    if total_eval or total_reason:
        def _fmt_tok(n): return f"{n/1e6:.2f}M" if n >= 1_000_000 else (f"{n/1000:.1f}K" if n >= 1000 else str(n))
        typer.echo(f"  Tokens — eval: {_fmt_tok(total_eval)}  reasoning: {_fmt_tok(total_reason)}")

    typer.echo(f"  {len(entries)} events  •  {run.run_dir}\n")


@app.command()
def dashboard(
    run_dir: Annotated[
        Optional[Path],
        typer.Option("--run-dir", help="Directory for run history. Defaults to .reflex/ in cwd."),
    ] = None,
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to serve on."),
    ] = 8337,
    host: Annotated[
        str,
        typer.Option("--host", help="Bind address."),
    ] = "127.0.0.1",
    no_open: Annotated[
        bool,
        typer.Option("--no-open", help="Don't open the browser automatically."),
    ] = False,
) -> None:
    """Launch the reflex dashboard — a local web UI for exploring runs."""
    from aevyra_reflex.dashboard import serve

    serve(
        run_dir=run_dir or ".reflex",
        host=host,
        port=port,
        open_browser=not no_open,
    )


def _make_progress_cb(typer_mod):
    """Create a progress callback that prints iteration scores."""
    step = {"phase": "baseline"}

    def callback(record):
        if step["phase"] == "baseline":
            step["phase"] = "optimize"
            typer_mod.echo()
            typer_mod.echo("Step 2/3  Optimizing...")
        typer_mod.echo(f"  iteration {record.iteration:>2d}  score: {record.score:.4f}")

    return callback


def _add_metrics(optimizer, metric_names: list[str], judge: str | None) -> None:
    """Resolve metric names to verdict Metric instances."""
    from aevyra_verdict import BleuScore, ExactMatch, LLMJudge, RougeScore
    from aevyra_verdict.providers import get_provider

    metric_map = {
        "rouge": RougeScore,
        "bleu": BleuScore,
        "exact": ExactMatch,
    }

    # --metric and --judge are mutually exclusive
    if metric_names and judge:
        typer.echo("Error: use --metric or --judge, not both.", err=True)
        raise typer.Exit(code=1)

    # If neither specified, default to rouge
    if not metric_names and not judge:
        metric_names = ["rouge"]

    for name in metric_names:
        name_lower = name.lower()
        if name_lower not in metric_map:
            available = ", ".join(sorted(metric_map))
            typer.echo(f"Error: unknown metric {name!r}. Available: {available}", err=True)
            raise typer.Exit(code=1)
        optimizer.add_metric(metric_map[name_lower]())

    if judge:
        parts = judge.split("/", 1)
        if len(parts) != 2:
            typer.echo(f"Error: --judge must be in 'provider/model' format, got: {judge}", err=True)
            raise typer.Exit(code=1)

        # Resolve provider aliases for judge too (openrouter, together, etc.)
        from aevyra_reflex.optimizer import _resolve_provider
        resolved = _resolve_provider(parts[0], parts[1])

        # Build kwargs for get_provider — only pass non-None values
        provider_kwargs: dict[str, Any] = {}
        if resolved.get("api_key"):
            provider_kwargs["api_key"] = resolved["api_key"]
        if resolved.get("base_url"):
            provider_kwargs["base_url"] = resolved["base_url"]

        judge_provider = get_provider(
            resolved["provider_name"], resolved["model"],
            **provider_kwargs,
        )
        optimizer.add_metric(LLMJudge(judge_provider=judge_provider))

    if not metric_names and not judge:
        typer.echo("Error: need at least one --metric or --judge.", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
