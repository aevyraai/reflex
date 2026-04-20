"""
dev_assistant/pipeline.py

A developer assistant for the pyflow data-pipeline library.  The agent has
three tools:

  search_docs(query)       — keyword search over the pyflow reference docs
  calculate(expression)    — evaluate an arithmetic expression safely
  get_date()               — return today's ISO date

The agent loop runs for up to MAX_TOOL_ROUNDS rounds before forcing a final
answer.  This lets the model do multi-step reasoning: look up a throughput
benchmark, then calculate how long a job will take, then compose a grounded
answer.

Provider selection
------------------
The pipeline works with any OpenAI-compatible endpoint.  Two ways to configure:

1. Named provider shortcut (smoke test: --provider; reflex: PIPELINE_PROVIDER):

   Provider       Env key needed          Default model
   openrouter     OPENROUTER_API_KEY      qwen/qwen3-8b
   ollama         (none)                  qwen3:8b
   together       TOGETHER_API_KEY        Qwen/Qwen3-8B-Instruct-Turbo

2. Fully custom endpoint (overrides provider defaults):

   PIPELINE_BASE_URL=https://my-endpoint/v1
   PIPELINE_MODEL=my-model
   PIPELINE_API_KEY=my-key          # optional if endpoint is public

For the smoke test, CLI flags mirror all env vars (see __main__ below).

Usage
-----
    # Smoke test — OpenRouter (default):
    export OPENROUTER_API_KEY=sk-or-...
    python examples/dev_assistant/pipeline.py \\
      "How many records can I process in 2 hours with workers=4?"

    # Smoke test — Ollama:
    python examples/dev_assistant/pipeline.py --provider ollama \\
      "How many records can I process in 2 hours with workers=4?"

    # Smoke test — any custom endpoint:
    python examples/dev_assistant/pipeline.py \\
      --base-url https://my-endpoint/v1 --model my-model --api-key my-key \\
      "How many records can I process in 2 hours with workers=4?"

    # Reflex CLI — OpenRouter:
    export OPENROUTER_API_KEY=sk-or-...
    aevyra-reflex optimize \\
      --pipeline-file examples/dev_assistant/pipeline.py \\
      --inputs-file   examples/dev_assistant/questions.json \\
      examples/dev_assistant/prompt.md \\
      --judge openrouter/qwen/qwen3-8b \\
      --judge-criteria examples/dev_assistant/judge.md \\
      -o examples/dev_assistant/best_prompt.md

    # Reflex CLI — Ollama:
    PIPELINE_PROVIDER=ollama aevyra-reflex optimize \\
      --pipeline-file examples/dev_assistant/pipeline.py \\
      --inputs-file   examples/dev_assistant/questions.json \\
      examples/dev_assistant/prompt.md \\
      --judge-criteria examples/dev_assistant/judge.md \\
      -o examples/dev_assistant/best_prompt.md

    # Reflex CLI — custom endpoint:
    PIPELINE_BASE_URL=https://my-endpoint/v1 \\
    PIPELINE_MODEL=my-model \\
    PIPELINE_API_KEY=my-key \\
    aevyra-reflex optimize ...
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
from typing import Any

from openai import OpenAI

from aevyra_reflex import AgentTrace, TraceNode

# ---------------------------------------------------------------------------
# Provider registry and client initialisation
# ---------------------------------------------------------------------------

# Named shortcuts: (base_url, api_key_env_var, default_model)
# api_key_env_var=None means no key is required (e.g. local Ollama).
_PROVIDERS: dict[str, tuple[str, str | None, str]] = {
    "openrouter": (
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "qwen/qwen3-8b",
    ),
    "ollama": (
        "http://localhost:11434/v1",
        None,           # no key required
        "qwen3:8b",
    ),
    "together": (
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
        "Qwen/Qwen3-8B-Instruct-Turbo",
    ),
}


def _build_client(
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[OpenAI, str]:
    """
    Return (OpenAI client, model_name) from explicit args or env vars.

    Resolution order (first non-None wins):
      1. Explicit args passed to this function
      2. PIPELINE_BASE_URL / PIPELINE_MODEL / PIPELINE_API_KEY env vars
      3. Named provider (--provider flag or PIPELINE_PROVIDER env var)
      4. Default: openrouter
    """
    provider = provider or os.environ.get("PIPELINE_PROVIDER", "openrouter")
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Known providers: {list(_PROVIDERS)}. "
            "Use --base-url / --model / --api-key for a custom endpoint."
        )

    p_base_url, p_key_env, p_default_model = _PROVIDERS[provider]

    resolved_base_url = base_url or os.environ.get("PIPELINE_BASE_URL") or p_base_url
    resolved_model    = model    or os.environ.get("PIPELINE_MODEL")    or p_default_model

    if api_key:
        resolved_key = api_key
    elif os.environ.get("PIPELINE_API_KEY"):
        resolved_key = os.environ["PIPELINE_API_KEY"]
    elif p_key_env and os.environ.get(p_key_env):
        resolved_key = os.environ[p_key_env]
    elif p_key_env is None:
        resolved_key = "none"   # local endpoints (Ollama) ignore the key
    else:
        raise EnvironmentError(
            f"Provider '{provider}' requires an API key. "
            f"Set {p_key_env} or pass --api-key."
        )

    return OpenAI(base_url=resolved_base_url, api_key=resolved_key), resolved_model


# Initialised from env vars at import time (used by reflex pipeline mode).
# The smoke-test __main__ block may overwrite these globals after parsing flags.
client, MODEL = _build_client()

MAX_TOOL_ROUNDS = 4  # maximum agentic rounds before forcing a final answer


# ---------------------------------------------------------------------------
# Knowledge base — pyflow reference documentation
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: dict[str, str] = {
    "retry": """
RetryPolicy controls how failed steps are retried.
  RetryPolicy(max_attempts=3, backoff_factor=2.0, retry_on=[ExceptionType])
Pass it to Pipeline: Pipeline(retry=RetryPolicy(max_attempts=3, backoff_factor=2.0))
Omit retry_on to retry on any exception.
""",
    "map_flatmap": """
map() applies a function to each element and returns one output per input.
flat_map() applies a function that returns a list and flattens the results into
a single stream.  Use flat_map() when your transform produces variable-length output.
""",
    "parquet": """
ParquetSink writes pipeline output to a partitioned Parquet table.
  pipeline.sink(ParquetSink(path='output/', partition_by=['date', 'region']))
Partitions are created automatically based on column values.
""",
    "workers": """
Set workers=N in Pipeline() to run steps in parallel using a process pool.
  Pipeline(workers=4)
For IO-bound steps use workers='auto' to default to 2× CPU count.
Each worker is an OS process; shared state must use multiprocessing primitives.
""",
    "schema": """
SchemaCoercionPolicy handles schema mismatches when reading mixed-format data.
  Pipeline(schema_policy=SchemaCoercionPolicy(on_mismatch='cast'))
on_mismatch options: 'cast' (coerce types), 'drop' (skip mismatched rows),
'raise' (default, fail fast).
""",
    "checkpoint": """
CheckpointSink writes periodic checkpoints so interrupted runs can resume.
  CheckpointSink(path='.pyflow_cache/', every_n=100)
On restart, pyflow detects the checkpoint and skips already-processed records.
every_n controls how often checkpoints are written (every N records).
""",
    "filter": """
.filter(predicate_fn) drops records that do not satisfy the predicate.
  pipeline.filter(lambda r: r['status'] == 'active')
Filtered records are dropped entirely, not buffered.  Chain after any step.
""",
    "join": """
pipeline.join(other_pipeline, on='key_field', how='inner')
Supported join types: inner, left, right, outer.
Both pipelines must be finalized (no pending steps) before joining.
""",
    "metrics": """
Implement MetricsCollector and pass via Pipeline(metrics=MyCollector()).
pyflow calls on_record(record, step_name) after each step and
on_pipeline_end(summary) at completion.
""",
    "batch_size": """
Batch size depends on available memory and record size.
The docs recommend starting with 1000 and profiling:
  Pipeline(batch_size=1000, profile=True)
There is no single recommended value; profile your specific workload.
""",
    "kafka": """
KafkaSource reads from a Kafka topic.
  KafkaSource(topic='my-topic', bootstrap_servers='localhost:9092',
              group_id='my-group')
Set auto_offset_reset='earliest' to reprocess from the start of the topic.
""",
    "deduplicate": """
pipeline.deduplicate(key='field_name') removes duplicate records.
By default, the first occurrence is kept; set keep='last' for the most recent.
For large streams, set window=10000 to limit deduplication to the last N records.
""",
    "step_harness": """
StepHarness tests a pipeline step in isolation.
  harness = StepHarness(my_step_fn)
  harness.run([record1, record2])
Inspect harness.output; StepHarness captures exceptions and timing per record.
""",
    "on_error": """
By default, exceptions propagate and halt the pipeline.
  on_error='skip'         — drop failing records with a warning
  on_error='dead_letter'  — route them to a DeadLetterSink for manual inspection
""",
    "watermark": """
pipeline.watermark(event_time_field='ts', max_lateness_seconds=30)
Records arriving more than 30 s after the watermark are dropped by default.
Set on_late='emit' to still process late records.
""",
    "async_step": """
Decorate async step functions with @async_step.
pyflow runs them in an asyncio event loop per worker.
Mix sync and async steps freely; pyflow handles the event loop lifecycle.
""",
    "log_step_counts": """
Pipeline(log_step_counts=True) enables step-level record counters.
Counts are printed at pipeline completion and accessible via result.step_counts.
For live monitoring, attach a MetricsCollector.
""",
    "serialization": """
Inter-step serialization formats:
  msgpack (default, fastest)
  json    (human-readable, good for debugging)
  pickle  (arbitrary Python objects)
Set via Pipeline(serialization='json').
msgpack is recommended for production.
""",
    "database_sink": """
DatabaseSink(url='postgresql://...', pool_size=5, max_overflow=10)
pool_size and max_overflow map directly to SQLAlchemy's create_engine() pool settings.
""",
    "preview": """
pipeline.preview(n=10) runs the pipeline in dry-run mode, stopping after n records.
Returns a list of records and step-level timing.  Does not write to any sinks.
""",
    "step_timeout": """
@step(timeout_seconds=30) enforces a per-record timeout on a step function.
Raises StepTimeoutError on timeout.
Combined with on_error='skip', timed-out records are dropped without halting
the pipeline.
""",
    "clone": """
pipeline.clone(config_overrides) returns a new Pipeline with overridden settings
and the same steps.
  pipeline.clone({'workers': 8, 'batch_size': 500})
""",
    "backpressure": """
pyflow uses bounded queues between steps.  When a downstream queue fills,
upstream steps pause automatically.
Set queue_size=N in Pipeline() to control buffer size.  Default is 1000 per step.
""",
    "debug_step": """
debug_step(my_fn, sample_rate=0.1) logs 10% of records (input, output, latency)
at DEBUG level.
  pipeline.map(debug_step(my_fn, sample_rate=0.1))
Set sample_rate=1.0 to log every record.
""",
    "profile": """
pipeline.profile() returns a ProfilingReport with per-step metrics:
  records_in, records_out, avg_latency_ms, p99_latency_ms, error_count
Also includes a bottleneck property pointing to the slowest step.
""",
    "tap": """
pipeline.tap(side_effect_fn) runs a function on each record without altering
the stream.  The return value is discarded; the original record passes through
unchanged.  Use for logging, alerting, or metrics emission.
""",
    "branch": """
pipeline.branch(condition_fn, true_pipeline, false_pipeline)
Records where condition_fn returns True → true_pipeline.
Others → false_pipeline.  Both branches can have different sinks.
""",
    "rate_limit": """
Pipeline(rate_limit=100) caps throughput at 100 records per second.
For burst control: Pipeline(rate_limit=(100, 500)) as (sustained_rps, burst_rps).
Rate limiting applies globally across all workers.
""",
    "memory_windows": """
pyflow does not impose hard memory limits on windows.
The docs recommend keeping window sizes below 100k records per key to avoid
GC pressure.  Use CheckpointSink to spill large state to disk for windows
exceeding this.
""",
    "custom_source": """
Subclass BaseSource and implement __iter__: yield one record dict per call.
Register with @register_source('my_source').
  MySource(api_key='...', endpoint='...')
""",
    "version": """
pyflow version history:
  0.1.0 — initial release  (2024-01-15): basic map/filter/sink
  0.2.0 — (2024-03-10): KafkaSource, DatabaseSink
  0.3.0 — (2024-06-01): CheckpointSink, RetryPolicy, watermark
  0.4.0 — (2024-09-20): branch(), tap(), clone(), debug_step()
  0.5.0 — (2025-01-08): async_step, StepHarness, pipeline.profile()
  0.5.2 — current stable (2025-03-01): bug fixes and performance improvements
""",
    "performance": """
Processing speed and throughput benchmarks — how many records per second
pyflow can process with different worker counts.
Benchmark conditions: 8-core machine, msgpack serialization, simple map step.
  workers=1:  ~45,000 records/sec   (45k rps)
  workers=2:  ~88,000 records/sec   (88k rps)
  workers=4:  ~170,000 records/sec  (170k rps)
  workers=8:  ~310,000 records/sec  (310k rps)
IO-bound steps (e.g. database writes) scale better; use workers='auto'.
CPU-bound steps see ~85% parallel efficiency at workers=8.
Capacity calculation: multiply records/sec by duration in seconds.
  Example: workers=4 for 90 minutes = 170,000 * 5,400 = 918,000,000 records.
""",
}


# ---------------------------------------------------------------------------
# Search helpers — IDF-weighted keyword retrieval
# ---------------------------------------------------------------------------

def _stem(word: str) -> str:
    """Strip trailing plural 's' so 'workers'/'worker' and 'records'/'record' align."""
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _tokenize(text: str) -> set[str]:
    # Replace non-alphanumeric chars with spaces so "workers=4:" → "workers 4"
    # rather than "workers4" (which would never match the query token "worker").
    return {_stem(w) for w in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split() if len(w) > 1}


def _build_idf(kb: dict[str, str]) -> dict[str, float]:
    """Compute inverse-document-frequency for every token in the knowledge base."""
    N = len(kb)
    df: dict[str, int] = {}
    for passage in kb.values():
        for tok in _tokenize(passage):
            df[tok] = df.get(tok, 0) + 1
    return {tok: math.log(N / count) for tok, count in df.items()}


_IDF = _build_idf(KNOWLEDGE_BASE)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def search_docs(query: str) -> str:
    """IDF-weighted search over KNOWLEDGE_BASE; return the top-3 matching passages."""
    query_tokens = _tokenize(query)
    scores: list[tuple[float, str]] = []
    for key, passage in KNOWLEDGE_BASE.items():
        doc_tokens = _tokenize(passage)
        # IDF-weighted overlap: rare terms count more than ubiquitous ones
        score = sum(_IDF.get(t, 0.0) for t in query_tokens & doc_tokens)
        scores.append((score, key))
    # Sort descending by score; break ties by insertion order (stable key ordering)
    scores.sort(key=lambda x: -x[0])

    results: list[str] = []
    for score, key in scores[:3]:
        if score > 0:
            results.append(f"[{key}]\n{KNOWLEDGE_BASE[key].strip()}")

    return "\n\n".join(results) if results else "No relevant documentation found."


def calculate(expression: str) -> str:
    """
    Safely evaluate a numeric or date expression.

    Arithmetic examples:
      "170_000 * 5400"              →  918000000
      "10_000_000 / 88_000"         →  113.636...

    Date-difference examples — use the pre-bound `today` variable; do NOT
    call datetime.date.today() or get_date() inside the expression:
      "(today - datetime.date(2025,1,8)).days"   →  days since 2025-01-08
      "(datetime.date(2026,6,1) - today).days"   →  days until 2026-06-01
    """
    safe_globals: dict[str, Any] = {
        "__builtins__": {},
        "math": math,
        "datetime": datetime,
        # Pre-computed today so expressions like (datetime.date(2025,3,1) - today).days
        # work without the model needing to call datetime.date.today() or get_date()
        # inside eval (both fail: today() needs builtins, get_date is not in scope).
        "today": datetime.date.today(),
        # Allow get_date() calls inside calculate for models that try it anyway
        "get_date": get_date,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sum": sum,
    }
    try:
        result = eval(expression, safe_globals)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


def get_date() -> str:
    """Return today's date in ISO 8601 format (YYYY-MM-DD)."""
    return datetime.date.today().isoformat()


TOOL_REGISTRY: dict[str, Any] = {
    "search_docs": search_docs,
    "calculate": calculate,
    "get_date": get_date,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the pyflow documentation. Use this for any question about "
                "pyflow APIs, configuration, behaviour, version history, or "
                "performance benchmarks. For throughput or capacity questions use "
                "terms like 'throughput benchmark workers records per second'. "
                "For version/release questions use 'version release date history'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Specific keywords to look up. Prefer pyflow-specific "
                            "terms: 'throughput benchmark', 'workers', 'RetryPolicy', "
                            "'CheckpointSink', 'KafkaSource', etc."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a Python arithmetic or date expression. Use for any "
                "numerical calculation (throughput, time-to-completion, batch counts, "
                "record capacities) and for date differences after calling get_date(). "
                "The datetime module is available for date arithmetic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A valid Python expression. Arithmetic: '170_000 * 5400'. "
                            "Date diff: use the pre-bound `today` variable — "
                            "'(today - datetime.date(2025,1,8)).days' or "
                            "'(datetime.date(2026,6,1) - today).days'. "
                            "Do NOT call datetime.date.today() or get_date() inside the expression."
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_date",
            "description": (
                "Return today's date in ISO format. Use when the question involves "
                "how long ago something happened or how many days until a deadline."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen3 prepends before its answer."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _trace(prefix: str, msg: str) -> None:
    """Print a progress line to stderr (visible in CLI; does not affect return values).

    Set PIPELINE_QUIET=1 to suppress all progress output.
    """
    if os.environ.get("PIPELINE_QUIET", "").lower() not in ("1", "true", "yes"):
        import sys
        print(f"{prefix} {msg}", file=sys.stderr, flush=True)


def _fmt_args(args: dict | list) -> str:
    """One-line summary of tool arguments for progress output."""
    if not isinstance(args, dict):
        return repr(args)[:120]
    parts = []
    for k, v in args.items():
        v_str = str(v)
        parts.append(f"{k}={v_str[:60]!r}" if len(v_str) > 60 else f"{k}={v_str!r}")
    return ", ".join(parts)


def _fmt_result(result: str) -> str:
    """First line of a tool result, truncated to 80 chars."""
    first_line = result.strip().splitlines()[0] if result.strip() else result
    return first_line[:80] + ("…" if len(first_line) > 80 else "")


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def pipeline_fn(prompt: str, question: str | dict) -> AgentTrace:
    """
    Run a full agentic turn and return a structured trace for Reflex to score.

    The agent loop calls tools for up to MAX_TOOL_ROUNDS rounds.  This allows
    multi-step reasoning: e.g. call search_docs to retrieve a throughput
    benchmark, then call calculate to compute job duration, then compose a
    grounded answer.

    ``question`` may be a plain string or a dict with ``"question"`` and
    optional ``"ideal"`` fields (the format used by questions.json).

    Nodes
    -----
    tools_called   (optimize=False) — ordered list of every tool call made
    tool_results   (optimize=False) — ordered list of every tool result received
    answer         (optimize=True)  — final answer synthesised from tool results

    Progress lines are printed to stderr as each tool fires so the CLI shows
    activity rather than a silent wait.  Set PIPELINE_QUIET=1 to suppress.

    IMPORTANT: all LLM calls use temperature=0.0.  Reflex compares prompt
    variants by running this function multiple times; non-zero temperature
    introduces sampling noise that makes variant comparisons unreliable.
    """
    # Unpack dict inputs (e.g. {"question": "...", "ideal": "..."})
    if isinstance(question, dict):
        ideal: str | None = question.get("ideal")
        question = question.get("question", str(question))
    else:
        ideal = None

    # Short question prefix used in every progress line so parallel workers
    # stay identifiable when their output interleaves.
    q_prefix = f"[{question[:38]}{'…' if len(question) > 38 else ''}]"

    messages: list[dict] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]

    all_calls: list[dict] = []
    all_results: list[dict] = []
    final_answer = ""
    total_tokens = 0

    for _round in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.0,
        )
        if response.usage:
            total_tokens += response.usage.total_tokens
        msg = response.choices[0].message

        if not msg.tool_calls:
            # Model finished — extract answer (strip any thinking tokens)
            final_answer = _strip_thinking(msg.content or "")
            _trace(q_prefix, f"✓  answer  ({len(all_calls)} tool call{'s' if len(all_calls) != 1 else ''})")
            break

        # Append assistant turn (may include tool_calls)
        messages.append(msg)

        # Execute every tool call in this round
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
                if not isinstance(fn_args, dict):
                    # Model returned a JSON array instead of an object —
                    # treat each element as a positional arg value
                    fn_args = {str(i): v for i, v in enumerate(fn_args)} if isinstance(fn_args, list) else {}
            except json.JSONDecodeError:
                fn_args = {}

            _trace(q_prefix, f"→  {fn_name}({_fmt_args(fn_args)})")

            all_calls.append({"name": fn_name, "args": fn_args})

            if fn_name in TOOL_REGISTRY:
                try:
                    result = TOOL_REGISTRY[fn_name](**fn_args)
                except TypeError:
                    # Model used wrong argument name — try passing the first value positionally
                    try:
                        result = TOOL_REGISTRY[fn_name](*fn_args.values())
                    except Exception as exc:
                        result = f"Tool call error: {exc}"
            else:
                result = f"Unknown tool: {fn_name}"

            _trace(q_prefix, f"←  {_fmt_result(result)}")

            all_results.append({"name": fn_name, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )
    else:
        # Exceeded MAX_TOOL_ROUNDS — force a final answer
        _trace(q_prefix, f"→  (max rounds reached, forcing answer)")
        closing = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
        )
        if closing.usage:
            total_tokens += closing.usage.total_tokens
        final_answer = _strip_thinking(closing.choices[0].message.content or "")
        _trace(q_prefix, f"✓  answer  ({len(all_calls)} tool calls)")

    return AgentTrace(
        nodes=[
            TraceNode(
                name="tools_called",
                input=question,
                output=all_calls if all_calls else "(no tools called)",
                optimize=False,
            ),
            TraceNode(
                name="tool_results",
                input=all_calls,
                output=all_results if all_results else "(no tool results)",
                optimize=False,
            ),
            TraceNode(
                name="answer",
                input={"question": question, "tool_results": all_results},
                output=final_answer,
                optimize=True,
            ),
        ],
        ideal=ideal,
        tokens=total_tokens,
    )


# ---------------------------------------------------------------------------
# Manual smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(
        description="Dev assistant pipeline smoke test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Provider examples:
  # OpenRouter (default — needs OPENROUTER_API_KEY):
  python pipeline.py "How long for 10M records at workers=4?"

  # Ollama (local, no key needed):
  python pipeline.py --provider ollama "How long for 10M records at workers=4?"

  # Together AI (needs TOGETHER_API_KEY):
  python pipeline.py --provider together "How long for 10M records at workers=4?"

  # Any custom OpenAI-compatible endpoint:
  python pipeline.py --base-url https://my-endpoint/v1 --model my-model \\
    --api-key my-key "How long for 10M records at workers=4?"
""",
    )
    _parser.add_argument(
        "question",
        nargs="?",
        default=(
            "At workers=4, how many records can I process in 90 minutes, "
            "and what throughput does the documentation quote for that setting?"
        ),
        help="Question to ask the assistant (default: workers=4 capacity question)",
    )
    _parser.add_argument(
        "--provider",
        default=None,
        choices=list(_PROVIDERS),
        metavar="NAME",
        help=f"Named provider shortcut: {list(_PROVIDERS)} (default: openrouter)",
    )
    _parser.add_argument(
        "--base-url",
        default=None,
        metavar="URL",
        help="OpenAI-compatible base URL (overrides --provider default)",
    )
    _parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="Model name (overrides --provider default)",
    )
    _parser.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="API key (overrides env var lookup; not needed for Ollama)",
    )
    _parser.add_argument(
        "--prompt",
        default=None,
        metavar="FILE",
        help="Path to a prompt file (default: examples/dev_assistant/prompt.md)",
    )
    _args = _parser.parse_args()

    # Rebuild client from flags — overwrites module-level globals so pipeline_fn
    # picks up the new values (client and MODEL are looked up at call time).
    client, MODEL = _build_client(
        provider=_args.provider,
        base_url=_args.base_url,
        model=_args.model,
        api_key=_args.api_key,
    )

    _prompt_path = _args.prompt or os.path.join(os.path.dirname(__file__), "prompt.md")
    if not os.path.exists(_prompt_path):
        _parser.error(
            f"prompt file not found: {_prompt_path}\n"
            + ("  (run the optimizer first to generate best_prompt.md)" if "best_prompt" in _prompt_path else "")
        )
    with open(_prompt_path) as _f:
        _prompt = _f.read().strip()

    print(f"Provider : {_args.provider or os.environ.get('PIPELINE_PROVIDER', 'openrouter')}")
    print(f"Model    : {MODEL}")
    print(f"Prompt   : {_prompt_path}")
    print(f"Question : {_args.question}\n")

    trace = pipeline_fn(_prompt, _args.question)

    print("=== tools_called ===")
    print(json.dumps(trace.nodes[0].output, indent=2))
    print("\n=== tool_results ===")
    for r in trace.nodes[1].output if isinstance(trace.nodes[1].output, list) else []:
        print(f"  [{r['name']}] {r['result'][:300]}")
    print("\n=== answer ===")
    print(trace.nodes[2].output)
    print(f"\n=== tokens ===")
    print(f"  {trace.tokens} total  (prompt + completion across all rounds)")

    # --- Mechanical pre-check (no inference needed) ---
    # Detects the most common grounding failures before the LLM judge runs.
    # Small models acting as their own judge often miss these; this catches them
    # mechanically and flags them regardless of the LLM score.
    print("\n=== pre-check ===")
    _tools_called = trace.nodes[0].output
    _answer_text  = trace.nodes[2].output or ""
    _called_names = (
        {c["name"] for c in _tools_called}
        if isinstance(_tools_called, list) else set()
    )
    _q_lower = _args.question.lower()

    _warnings: list[str] = []

    # 1. No tools called at all
    if not _called_names:
        _warnings.append("no tools were called — answer is entirely from model memory")

    # 2. Arithmetic in answer but calculate not called
    _has_numbers = bool(re.search(r'\d[\d,_.]*\s*[×x\*\/\+\-]\s*[\d,_.]+|\d+\s+records|\d+\s+second|\d+\s+minute|\d+\s+hour', _answer_text))
    if _has_numbers and "calculate" not in _called_names:
        _warnings.append("answer contains arithmetic but calculate was not called — numbers are unverified")

    # 3. Date claim in answer but get_date not called
    _has_date_claim = bool(re.search(r'\d+\s+(day|week|month|year)|ago|since\s+\d{4}', _answer_text, re.IGNORECASE))
    if _has_date_claim and "get_date" not in _called_names:
        _warnings.append("answer makes a date/age claim but get_date was not called")

    # 4. Question asks about workers/throughput but search_docs not called
    _needs_docs = any(kw in _q_lower for kw in ("workers", "how many records", "throughput", "records per", "how long", "how fast"))
    if _needs_docs and "search_docs" not in _called_names:
        _warnings.append("question requires documentation but search_docs was not called")

    if _warnings:
        for _w in _warnings:
            print(f"  ⚠  {_w}")
        print(f"  → predicted score: 1–2/5  (grounding failures detected)")
        print(f"  tip: run the full optimizer with a frontier judge to fix these —")
        print(f"       --judge openrouter/anthropic/claude-sonnet-4-5")
    else:
        print(f"  ✓  no obvious grounding failures detected")
