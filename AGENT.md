# AGENT.md

## Project overview

aevyra-reflex is an agentic prompt optimizer that works in two modes:

- **Standard mode**: given a JSONL eval dataset and a starting system prompt, it diagnoses why the model is underperforming, reasons about failure patterns, and iteratively rewrites the prompt until scores hit the target — all in a single command.
- **Pipeline mode**: given a `pipeline_fn(prompt, input) -> AgentTrace` and a list of raw inputs, it re-runs the full agent pipeline on every candidate prompt, scores the resulting execution traces (tool calls, intermediate outputs, final answer) via the judge, and optimizes the prompt against the whole system's behaviour — not just the final output string.

Both modes draw from four optimization axes (iterative, PDO, fewshot, structural) and adaptively pick the right one at each step.

It depends on `aevyra-verdict` for evaluation (running completions, scoring with metrics like ROUGE/BLEU/ExactMatch/LLMJudge). For reasoning, it uses an LLM (Claude by default, but configurable to any model including local ones via `--reasoning-model`).

## Architecture

```
src/aevyra_reflex/
├── __init__.py          # Public API exports
├── cli.py               # Typer CLI — optimize, runs, logs, dashboard commands
├── optimizer.py         # PromptOptimizer — orchestrates baseline → strategy → verify
├── result.py            # OptimizationResult, EvalSnapshot, IterationRecord
├── agent.py             # LLM agent wrapper (diagnosis, revision, judging)
├── trace.py             # AgentTrace, TraceNode — pipeline-mode optimization types
├── prompts.py           # All prompt templates for all strategies
├── run_store.py         # Run persistence, checkpointing, iteration logging
├── callbacks.py         # Callback protocol (MLflow, custom hooks)
├── dashboard/           # Local web UI (FastAPI + static assets)
└── strategies/
    ├── __init__.py      # Strategy registry (register_strategy, list_strategies)
    ├── base.py          # Abstract Strategy base class
    ├── iterative.py     # Diagnose-revise loop with causal rewrite log
    ├── pdo.py           # Dueling bandits with Thompson sampling
    ├── fewshot.py       # Few-shot example selection and optimization
    ├── structural.py    # Structural variant generation with transform history
    └── auto.py          # Adaptive multi-phase (default — combines all axes)
```

## Key concepts

- **PromptOptimizer.run()** does 3 steps: (1) baseline eval on held-out test set → (2) strategy optimization on train set → (3) verification eval on held-out test set. Supports callbacks, run checkpointing, and resume.
- **Dataset formats**: Reflex accepts JSONL and CSV files. JSONL auto-detects OpenAI, ShareGPT, and Alpaca schemas. CSV uses `input` and `ideal` column names by default; override with `--input-field` / `--output-field`. Pass `output_field=None` (or omit `--output-field`) for label-free CSV datasets. The Python API exposes `Dataset.from_csv(path, input_field=..., output_field=...)`.
- **Train/test split**: By default the dataset is split 80/20 (`train_ratio=0.8`). Optimization only sees the train split; baseline and final scores are computed on the held-out test set so reported improvement is honest. Controlled by `OptimizerConfig.train_ratio` / `--train-split`.
- **Validation split + early stopping**: Optional 3-way train/val/test split (`OptimizerConfig.val_ratio` / `--val-split`). Val set is evaluated after every iteration — if train score climbs but val plateaus, the prompt is overfitting. `OptimizerConfig.early_stopping_patience` / `--early-stopping-patience N` stops optimization when val score hasn't improved for N consecutive iterations and returns the best-val prompt. Val scores tracked in `IterationRecord.val_score` and `OptimizationResult.val_trajectory`; `early_stopped` flag set on result. Default: `train_ratio=0.8`, `val_ratio=0.1` → 70% train / 10% val / 20% test. Example with custom values: `--train-split 0.7 --val-split 0.1` → 60% train / 10% val / 30% test.
- **Mini-batch mode**: `OptimizerConfig.batch_size` / `--batch-size` limits how many training examples each optimization iteration samples. Default 0 = full training set. When set, each iteration draws a fresh random sample (seed = `batch_seed + iteration`) so the optimizer sees variety without evaluating every example every round. Speeds up per-iteration cost on large datasets; the stochasticity can help escape local optima. Baseline and final evals always use the full test set unaffected. `full_eval_steps` / `--full-eval-steps`: when using mini-batch mode, run a full training-set eval every N iterations to get accurate checkpoint scores. Full-eval iterations have `IterationRecord.is_full_eval=True` and are marked with a ◈ badge in the dashboard.
- **Statistical significance**: After every run, a paired Wilcoxon signed-rank test (scipy) or paired t-test (fallback) compares per-sample baseline vs final scores. `OptimizationResult.p_value` and `is_significant` are always populated; shown in `summary()` as `p=0.0234  ✓ significant (α=0.05)`. Use `--eval-runs N` to average N eval passes and report `mean ± std`.
- **Auto strategy** (default): Claude analyzes weaknesses → picks the best optimization axis → applies it for a few iterations → re-evaluates → picks next axis → repeat. Adaptively combines structural, iterative, fewshot, and pdo.
- **Iterative strategy**: eval → find worst samples → Claude diagnoses → revised prompt → repeat. Maintains a **causal rewrite log** (iteration, score delta, change summary, ✓/✗ outcome) fed back into subsequent prompts so the agent avoids repeating dead ends. **Label-free aware**: `iterative.py` detects `label_free = not dataset.has_ideals()` at run start and passes it to both `agent.diagnose_and_revise()` and `agent.refine()`. When `True`, `LABEL_FREE_EVAL_CONTEXT` is injected into `DIAGNOSE_FAILURES_PROMPT` / `REFINE_PROMPT` via the `{eval_context}` placeholder, redirecting the reasoning model from reference-comparison diagnosis to quality/format/instruction-following diagnosis. When `False`, `{eval_context}` is empty — zero behaviour change for labeled tasks.
- **Structural strategy**: generates structural variants in parallel (section reorder, markdown, XML tags, flat text) → evaluates all → keeps best → iterates. Maintains a **transform history** across rounds (which transforms were tried, which won, score deltas) injected into `freeform_restructure` to guide exploration.
- **PDO strategy**: pool of candidates → Thompson sampling picks duel pairs → pairwise judging → adaptive ranking → mutation → convergence. Pairwise judge uses `PAIRWISE_CRITERIA_WITH_IDEAL` when ideal is present, `PAIRWISE_CRITERIA_LABEL_FREE` (quality/instruction-following/conciseness) when it is not — set in `agent.judge_pairwise` at call time. **Ranking**: four methods available — `copeland` (wins-losses), `borda` (mean win rate), `elo` (Elo ratings from win matrix, 3-pass batch), `avg_winrate` (total wins/games). `fused` uses equal-weight Dirichlet fusion of all four. `auto` (default) maintains a Beta posterior per method (`method_alphas: np.ndarray`), updates it each round based on whether that method's predicted champion won more than it lost, then samples Dirichlet weights for fusion. Dominant method is logged each round. `_rank(W, method, method_alphas)` dispatches all modes; `_scores_for_method(W, method)` returns raw scores for a single method.
- **Few-shot strategy**: bootstrap exemplars from high-scoring samples → Claude selects diverse subset → inject as few-shot examples → iterate. Raises `ValueError` immediately for label-free datasets (fewshot depends on ideal answers to build demonstrations).
- **Pipeline mode** (`set_pipeline` + `set_inputs`): For prompts that live inside multi-step pipelines rather than controlling a single LLM call. `set_pipeline(fn)` registers a `fn(prompt, input) -> AgentTrace` callable; `set_inputs(inputs)` provides the raw inputs. Each optimization iteration re-runs the full pipeline with the current candidate prompt — so the classifier, retriever, etc. all run fresh — and scores each resulting `AgentTrace` via `metric.score()` directly (no EvalRunner). `AgentTrace` carries ordered `TraceNode` objects (name, input, output, `optimize` flag), an optional `ideal`, and `to_trace_text()` which formats the full trace for the judge. `_run_pipeline_eval()` runs all inputs in parallel via `ThreadPoolExecutor`. Supports all four strategies including `pdo` (uses `_run_duel_pipeline` to evaluate both candidates with `eval_fn` and picks the higher scorer). Mutually exclusive with `set_dataset()`.
- **Label-free evaluation**: datasets with no ideal answers. Reference metrics (`ExactMatch`, `BleuScore`, `RougeScore`) have `requires_ideal=True` (from verdict) and raise `ValueError` when used with a label-free dataset. CLI exits with an error if no `--judge` is given for a label-free dataset (`cli.py: _add_metrics`). `optimizer.py` re-checks at runtime. `auto` strategy auto-excludes `fewshot` from its available axes when `dataset.has_ideals()` is False (logs it at INFO). All other strategies work label-free with `LLMJudge`.
- **Source model migration**: `--source-model` / `OptimizerConfig.source_model` tells the reasoning model which model family the prompt was written for (e.g. `claude-sonnet`, `gpt-4o`). Enables idiom adaptation — XML tags → Markdown, role framing adjustments, verbosity tuning — when migrating prompts across model families.
- **Dataset language detection**: At run start, reflex samples up to 20 user messages from the training set and detects the dominant script family using Unicode character-range heuristics (no external dependencies). The detected language is passed to the reasoning model via `LLM(response_language=...)`, which prepends `LANGUAGE_INSTRUCTION` to every `generate()` call. This ensures the reasoning model writes revised prompts and diagnostic explanations in the same language as the dataset — preventing multilingual models (e.g. Qwen3) from silently switching to a different language. A 15% threshold guards against mislabelling English text that contains a handful of quoted foreign terms. Implemented in `lang.py` (`detect_language()`), `agent.py` (`LLM.response_language`), `prompts.py` (`LANGUAGE_INSTRUCTION`), and wired in `optimizer.py`.
- **Provider aliases** (`optimizer.py: PROVIDER_ALIASES`): maps short provider names to base URL + env var for the OpenAI-compat backend. Covers `together`, `fireworks`, `deepinfra`, `groq`, `lepton`, `perplexity`, and `gemini` (Google's v1beta OpenAI-compat endpoint, `GOOGLE_API_KEY`). Used by both `_resolve_provider` (eval side) and `_resolve_agent_backend` (reasoning side). `gemini-*` model names also auto-route via heuristic in `_resolve_agent_backend` without needing an explicit provider prefix. **`openrouter` is intentionally absent from `PROVIDER_ALIASES`** — it is handled natively by verdict's `OpenRouterProvider`, which reads `OPENROUTER_API_KEY` directly. Adding it to the aliases caused confusing `OPENAI_API_KEY` errors.
- **Run persistence**: every run is checkpointed to `.reflex/runs/`. Each iteration saves the prompt, score, reasoning, token counts, and `change_summary`. Interrupted runs are resumable with `--resume` / `--resume-from`.
- All evaluation goes through `aevyra-verdict`'s EvalRunner

## Development

```bash
pip install -e ".[dev]"
```

## Conventions

- Apache 2.0 license header on all `.py` files
- Type hints everywhere, `from __future__ import annotations`
- Logging via `logging.getLogger(__name__)`
- CLI uses typer
