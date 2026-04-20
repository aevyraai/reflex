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

Usage
-----
    # Run a single question manually:
    python examples/dev_assistant/pipeline.py "How many records can I process
    in 2 hours with workers=4?"

    # Run via reflex CLI (pipeline mode):
    export OPENROUTER_API_KEY=sk-or-...
    aevyra-reflex optimize \
      --pipeline-file examples/dev_assistant/pipeline.py \
      --inputs-file  examples/dev_assistant/questions.json \
      examples/dev_assistant/prompt.md \
      --judge openrouter/qwen/qwen3-8b \
      --judge-criteria examples/dev_assistant/judge.md \
      -o examples/dev_assistant/best_prompt.md
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
# OpenRouter client — Qwen3 8B
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = "qwen/qwen3-8b"
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
    scores.sort(reverse=True)

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

    Date-difference examples (use after calling get_date()):
      "(datetime.date(2026,4,19) - datetime.date(2024,1,15)).days"  →  826
    """
    safe_globals: dict[str, Any] = {
        "__builtins__": {},
        "math": math,
        "datetime": datetime,
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
                            "Date diff: '(datetime.date(2026,4,19) - datetime.date(2024,1,15)).days'."
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


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------

def pipeline_fn(prompt: str, question: str) -> AgentTrace:
    """
    Run a full agentic turn and return a structured trace for Reflex to score.

    The agent loop calls tools for up to MAX_TOOL_ROUNDS rounds.  This allows
    multi-step reasoning: e.g. call search_docs to retrieve a throughput
    benchmark, then call calculate to compute job duration, then compose a
    grounded answer.

    Nodes
    -----
    tools_called   (optimize=False) — ordered list of every tool call made
    tool_results   (optimize=False) — ordered list of every tool result received
    answer         (optimize=True)  — final answer synthesised from tool results
    """
    messages: list[dict] = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]

    all_calls: list[dict] = []
    all_results: list[dict] = []
    final_answer = ""

    for _round in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # Model finished — extract answer (strip any thinking tokens)
            final_answer = _strip_thinking(msg.content or "")
            break

        # Append assistant turn (may include tool_calls)
        messages.append(msg)

        # Execute every tool call in this round
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            all_calls.append({"name": fn_name, "args": fn_args})

            if fn_name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[fn_name](**fn_args)
            else:
                result = f"Unknown tool: {fn_name}"

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
        closing = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        final_answer = _strip_thinking(closing.choices[0].message.content or "")

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
        ]
    )


# ---------------------------------------------------------------------------
# Manual smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (
            "At workers=4, how many records can I process in 90 minutes, "
            "and what throughput does the documentation quote for that setting?"
        )
    )

    _prompt_path = os.path.join(os.path.dirname(__file__), "prompt.md")
    with open(_prompt_path) as _f:
        _prompt = _f.read().strip()

    print(f"Question : {_question}\n")
    trace = pipeline_fn(_prompt, _question)

    print("=== tools_called ===")
    print(json.dumps(trace.nodes[0].output, indent=2))
    print("\n=== tool_results ===")
    for r in trace.nodes[1].output if isinstance(trace.nodes[1].output, list) else []:
        print(f"  [{r['name']}] {r['result'][:300]}")
    print("\n=== answer ===")
    print(trace.nodes[2].output)
