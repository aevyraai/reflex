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

"""RAG Q&A pipeline for pyflow documentation.

Two-node pipeline:
  1. retrieve_docs  — keyword lookup against a local documentation knowledge base
  2. generate_answer — GPT-4o-mini produces an answer grounded in the retrieved docs

Reflex optimizes the system prompt that controls node 2.  Because the
full trace (retrieved docs + generated answer) is passed to the judge,
it can detect when the model ignores what was retrieved — which a static
single-LLM-call dataset cannot.

Usage (run this file directly to test):
  python examples/rag_qa/pipeline.py

Optimize with reflex:
  aevyra-reflex optimize examples/rag_qa/prompt.md \\
    --pipeline-file examples/rag_qa/pipeline.py \\
    --inputs-file examples/rag_qa/questions.json \\
    --judge openrouter/qwen/qwen3-8b \\
    --judge-criteria examples/rag_qa/judge.md \\
    -s auto \\
    --reasoning-model openrouter/qwen/qwen3-8b \\
    -o examples/rag_qa/best_prompt.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aevyra_reflex import AgentTrace, TraceNode

# ---------------------------------------------------------------------------
# Knowledge base — simulates a vector DB with pyflow documentation snippets.
# In production, replace retrieve_docs() with your real retrieval call.
# ---------------------------------------------------------------------------

_KNOWLEDGE_BASE: dict[str, str] = {
    "retry": """\
## Retry configuration
Use `RetryPolicy` to handle transient errors in pipeline steps:

```python
from pyflow import Pipeline, RetryPolicy

pipeline = Pipeline(
    retry=RetryPolicy(
        max_attempts=3,
        backoff_factor=2.0,
        retry_on=[ConnectionError, TimeoutError],  # omit to retry on any exception
    )
)
```

Set `max_attempts=1` (default) to disable retries. `backoff_factor` controls
exponential wait time between attempts: wait = backoff_factor ** attempt_number seconds.""",

    "map flat_map": """\
## map() vs flat_map()
`map(fn)` applies a function to each record and returns one output per input:
```python
pipeline.map(lambda r: {**r, 'value': r['value'] * 2})
```

`flat_map(fn)` applies a function that returns a list and flattens into a single stream:
```python
pipeline.flat_map(lambda r: [{'tag': t} for t in r['tags']])
```

Use `flat_map()` when your transform produces variable-length output (e.g. splitting
a record into one-per-tag). Records returning an empty list are effectively filtered out.""",

    "parquet sink partition": """\
## Writing partitioned Parquet output
Use `ParquetSink` with `partition_by` to write a partitioned table:

```python
from pyflow.sinks import ParquetSink

pipeline.sink(ParquetSink(
    path='output/',
    partition_by=['date', 'region'],   # creates output/date=.../region=.../
    compression='snappy',              # optional, default 'snappy'
))
```

Partitions are created automatically from record field values. The output path
must not exist or must be an existing Parquet directory.""",

    "parallel workers cpu": """\
## Parallel execution
Set `workers=N` in `Pipeline()` to run steps in parallel using a process pool:

```python
pipeline = Pipeline(workers=4)
```

For IO-bound steps (network, disk), use `workers='auto'` to default to 2× CPU count.
For CPU-bound steps, set `workers` equal to the number of physical cores.

Note: parallel workers serialize inter-step records using msgpack by default.
Set `serialization='pickle'` if your records contain non-serializable Python objects.""",

    "schema mismatch coercion": """\
## Handling schema mismatches
Use `SchemaCoercionPolicy` to control what happens when incoming records don't
match the expected schema:

```python
from pyflow import Pipeline, SchemaCoercionPolicy

pipeline = Pipeline(
    schema_policy=SchemaCoercionPolicy(
        on_mismatch='cast',    # 'cast' | 'drop' | 'raise' (default)
        strict_nulls=True,     # treat None as a type violation
    )
)
```

`cast` — attempt type coercion (e.g. "42" → 42). Fails with `CastError` if impossible.
`drop` — silently skip mismatched records.
`raise` — raise `SchemaMismatchError` on first violation (fail fast, default).""",

    "checkpoint resume": """\
## Checkpointing and resuming interrupted runs
Use `CheckpointSink` to write periodic snapshots so a pipeline can be resumed:

```python
from pyflow.sinks import CheckpointSink

pipeline.sink(CheckpointSink(
    path='.pyflow_cache/',
    every_n=100,          # checkpoint every 100 records
))
```

On restart, pyflow detects the latest checkpoint and skips already-processed records.
Delete the cache directory to force a full re-run. Checkpoints are keyed by pipeline
definition hash, so changing a step function invalidates them automatically.""",

    "filter": """\
## Filtering records
Chain `.filter(predicate_fn)` after any step to drop unwanted records:

```python
pipeline.filter(lambda r: r['status'] == 'active')
pipeline.filter(lambda r: r['score'] >= 0.5 and r['label'] is not None)
```

The predicate receives one record dict and must return True to keep it.
Filtered records are dropped entirely — they do not reach downstream steps or sinks.
There is no "quarantine" mode; use `on_error='dead_letter'` if you need to capture drops.""",

    "join stream": """\
## Joining two data streams
Use `pipeline.join()` to merge two pipelines on a shared key:

```python
result = orders.join(
    customers,
    on='customer_id',
    how='inner',           # 'inner' | 'left' | 'right' | 'outer'
)
```

Both pipelines must have no pending uncommitted steps before joining.
For large datasets, set `broadcast=True` on the smaller pipeline to load it
into memory rather than sort-merging — faster when one side fits in RAM.""",

    "metrics collector": """\
## Custom metrics collection
Implement `MetricsCollector` to capture per-step statistics:

```python
from pyflow import MetricsCollector

class MyCollector(MetricsCollector):
    def on_record(self, record: dict, step_name: str) -> None:
        ...  # called after each step

    def on_pipeline_end(self, summary: dict) -> None:
        ...  # called at pipeline completion with a summary dict

pipeline = Pipeline(metrics=MyCollector())
```

The built-in `LoggingCollector` prints per-step counts to stdout. Use it as a
reference implementation.""",

    "batch size memory": """\
## Batch size and memory
`batch_size` controls how many records are buffered before flushing to a sink
or passing to aggregation steps.

```python
pipeline = Pipeline(batch_size=1000)
```

There is no single recommended value. Start with `batch_size=1000` and profile:

```python
pipeline = Pipeline(batch_size=1000, profile=True)
result = pipeline.run(source)
print(result.step_counts)
```

For in-memory aggregation over large windows, the docs recommend keeping window
sizes below 100k records per key to avoid GC pressure.""",

    "kafka source": """\
## Reading from Kafka
Use `KafkaSource` to consume a Kafka topic as a pipeline source:

```python
from pyflow.sources import KafkaSource

source = KafkaSource(
    topic='my-topic',
    bootstrap_servers='localhost:9092',
    group_id='my-consumer-group',
    auto_offset_reset='latest',    # 'latest' (default) or 'earliest'
)
pipeline = Pipeline().source(source)
```

Set `auto_offset_reset='earliest'` to reprocess the topic from the beginning.
Consumer group offsets are committed automatically after each batch.""",

    "deduplicate": """\
## Deduplicating records
Use `.deduplicate(key='field_name')` to remove duplicate records by a field value:

```python
pipeline.deduplicate(key='id')
pipeline.deduplicate(key='id', keep='last')   # keep most recent instead of first
```

For large streams, set `window=10000` to limit deduplication to the last N records
(sliding window deduplication). Without a window, the deduplication set grows
unboundedly — use only for bounded datasets or small-cardinality keys.""",

    "step harness test": """\
## Testing pipeline steps in isolation
Use `StepHarness` to unit-test a step function without running the full pipeline:

```python
from pyflow.testing import StepHarness

harness = StepHarness(my_step_fn)
harness.run([
    {'input': 'record one'},
    {'input': 'record two'},
])
print(harness.output)    # list of output records
print(harness.errors)    # list of (record, exception) for any failures
```

`StepHarness` captures exceptions and per-record timing without raising.""",

    "error handling dead letter": """\
## Error handling: skip and dead-letter routing
By default, exceptions in a pipeline step propagate and halt execution.

Change the behavior with `on_error`:

```python
pipeline = Pipeline(on_error='skip')       # drop failing records with a warning
pipeline = Pipeline(on_error='dead_letter') # route to a DeadLetterSink
```

Configure the dead-letter sink:
```python
from pyflow.sinks import DeadLetterSink
pipeline.sink(DeadLetterSink(path='failed_records.jsonl'))
```

Each dead-letter record includes the original record, the exception type,
message, and step name where the error occurred.""",

    "watermark late": """\
## Watermarks and late-arriving events (streaming)
Use `.watermark()` to handle out-of-order events in streaming mode:

```python
pipeline.watermark(
    event_time_field='ts',
    max_lateness_seconds=30,   # records > 30s late are dropped by default
    on_late='drop',            # 'drop' (default) | 'emit'
)
```

Set `on_late='emit'` to still process late records rather than discard them.
The watermark advances to `max(observed_event_time) - max_lateness_seconds`.""",

    "async step": """\
## Async step functions
Decorate async step functions with `@async_step` to use them in a pipeline:

```python
from pyflow import async_step

@async_step
async def fetch_enrichment(record: dict) -> dict:
    async with aiohttp.ClientSession() as s:
        data = await s.get(f"https://api.example.com/{record['id']}")
        return {**record, 'enrichment': await data.json()}

pipeline.map(fetch_enrichment)
```

pyflow runs async steps in an asyncio event loop per worker. Sync and async
steps can be mixed freely in the same pipeline.""",

    "log step counts": """\
## Logging record counts per step
Enable per-step counters with `log_step_counts`:

```python
pipeline = Pipeline(log_step_counts=True)
result = pipeline.run(source)
print(result.step_counts)
# {'source': 10000, 'filter_active': 8734, 'enrich': 8734, 'sink': 8734}
```

Counts are printed at pipeline completion. For live monitoring during long
runs, attach a `MetricsCollector` with an `on_record` hook instead.""",

    "serialization": """\
## Inter-step serialization formats
pyflow serializes records between steps when using multiple workers.
Configure the format via `Pipeline(serialization='...')`:

| Format    | Speed  | Readability | Notes                             |
|-----------|--------|-------------|-----------------------------------|
| `msgpack` | Fast   | Binary      | Default. Best for production.     |
| `json`    | Medium | Human-readable | Useful for debugging.          |
| `pickle`  | Slow   | Binary      | Required for non-serializable Python objects. |

Example:
```python
pipeline = Pipeline(workers=4, serialization='json')
```""",

    "database sink pool": """\
## Database sink connection pooling
Pass SQLAlchemy pool settings to `DatabaseSink`:

```python
from pyflow.sinks import DatabaseSink

pipeline.sink(DatabaseSink(
    url='postgresql://user:pass@localhost/mydb',
    table='events',
    pool_size=5,        # number of persistent connections
    max_overflow=10,    # extra connections allowed above pool_size
    pool_timeout=30,    # seconds to wait for a connection before raising
))
```

These map directly to SQLAlchemy's `create_engine()` pool settings.
For SQLite, set `pool_size=1` (SQLite does not support concurrent writers).""",

    "preview dry run": """\
## Previewing pipeline output
Use `pipeline.preview(n=10)` to inspect the first N records without writing to sinks:

```python
records = pipeline.preview(n=10)
for r in records:
    print(r)
```

`preview()` runs the pipeline in dry-run mode and stops after n records pass through
all steps. Returns a list of records. Does not write to any sinks, including
CheckpointSink. Useful for validating transforms before a full run.""",

    "timeout step": """\
## Per-step timeouts
Use the `@step` decorator with `timeout_seconds` to limit execution time:

```python
from pyflow import step

@step(timeout_seconds=30)
def call_external_api(record: dict) -> dict:
    ...
```

If execution exceeds the timeout, pyflow raises `StepTimeoutError`. Combine with
`on_error='skip'` to drop timed-out records without halting the pipeline:

```python
pipeline = Pipeline(on_error='skip').map(call_external_api)
```""",

    "clone config": """\
## Cloning a pipeline with config overrides
Use `Pipeline.clone()` to create a variant with different settings without
redefining steps:

```python
base = Pipeline(workers=1, batch_size=100)
# ... add steps to base ...

production = base.clone({'workers': 8, 'batch_size': 500})
debug = base.clone({'serialization': 'json', 'log_step_counts': True})
```

`clone()` returns a new `Pipeline` with the same step sequence but overridden
configuration. The original pipeline is unchanged.""",

    "backpressure queue": """\
## Backpressure and queue sizing
pyflow uses bounded queues between steps. When a downstream queue is full,
upstream steps pause automatically — no records are dropped.

Configure queue size:
```python
pipeline = Pipeline(queue_size=1000)   # default: 1000 per step
```

Increase `queue_size` when you have bursty upstream steps and a slower
(but higher-throughput) downstream step. Decrease it to reduce peak memory.
The total in-flight record count is approximately `queue_size × (num_steps - 1)`.""",

    "debug step": """\
## Debug logging for a specific step
Wrap a step with `debug_step()` to log sample records without modifying logic:

```python
from pyflow.testing import debug_step

pipeline.map(debug_step(my_fn, sample_rate=0.1))   # log 10% of records
pipeline.map(debug_step(my_fn, sample_rate=1.0))   # log every record
```

Logs at DEBUG level: step name, input record, output record, and latency_ms.
Remove `debug_step()` before production — it adds per-record logging overhead.""",

    "profile bottleneck": """\
## Profiling pipeline performance
Enable profiling to identify bottlenecks:

```python
result = pipeline.profile()
```

Returns a `ProfilingReport` with per-step metrics:
- `records_in`, `records_out` — throughput at each step
- `avg_latency_ms`, `p99_latency_ms` — step execution time
- `error_count` — exceptions raised (when `on_error != 'raise'`)
- `bottleneck` — the step with the highest p99 latency

Example:
```python
report = pipeline.profile()
print(f"Bottleneck: {report.bottleneck}")
print(report.step_metrics['enrich'])
```""",

    "tap side effect": """\
## Side effects with tap()
Use `.tap(fn)` to run a side effect without modifying the pipeline stream:

```python
pipeline.tap(lambda r: alerts.send(r) if r['error_rate'] > 0.1 else None)
pipeline.tap(lambda r: metrics.increment('records_seen'))
```

The function receives each record. Its return value is ignored; the original
record passes through unchanged to the next step. Exceptions in `tap()` do NOT
propagate unless you configure `on_error='raise'`.""",

    "branch route": """\
## Branching the pipeline
Use `pipeline.branch()` to route records to different pipelines:

```python
high_priority, low_priority = pipeline.branch(
    lambda r: r['priority'] == 'high'
)
high_priority.sink(FastSink(...))
low_priority.sink(BatchSink(...))
```

Records where the condition returns `True` go to the first branch; others go
to the second. Both branches run in the same worker pool. You can chain
further steps on each branch independently.""",

    "rate limit": """\
## Rate limiting pipeline throughput
Use `rate_limit` to cap how many records the pipeline consumes per second:

```python
pipeline = Pipeline(rate_limit=100)             # 100 records/sec sustained
pipeline = Pipeline(rate_limit=(100, 500))      # 100 rps sustained, 500 rps burst
```

Rate limiting applies globally across all workers. Use it to avoid overwhelming
downstream APIs or databases. For per-step rate limiting, wrap the step with a
`RateLimitedStep` decorator.""",

    "window memory limit": """\
## Memory limits for windowed aggregation
pyflow does not impose hard memory limits on aggregation windows.

The documentation recommends keeping window sizes below **100k records per key**
to avoid garbage collector pressure with large in-memory state.

For windows that exceed this:
- Use `CheckpointSink` to spill state to disk periodically
- Reduce window size by increasing the event time granularity
- Switch to an external state store (e.g. Redis) via a custom `StateBackend`

Monitor memory with `Pipeline(profile=True)` and watch resident set size (RSS)
growth as window size increases.""",

    "custom source": """\
## Implementing a custom source
Subclass `BaseSource` and implement `__iter__` to yield record dicts:

```python
from pyflow.sources import BaseSource, register_source

@register_source('my_api')
class MyApiSource(BaseSource):
    def __init__(self, api_key: str, endpoint: str):
        self.api_key = api_key
        self.endpoint = endpoint

    def __iter__(self):
        page = 0
        while True:
            batch = fetch_page(self.endpoint, self.api_key, page)
            if not batch:
                break
            yield from batch
            page += 1
```

Use it like any built-in source:
```python
pipeline = Pipeline().source(MyApiSource(api_key='...', endpoint='https://...'))
```""",
}


def retrieve_docs(question: str) -> str:
    """Simple keyword retrieval from the local knowledge base.

    In production, replace this with a vector similarity search against your
    actual documentation store (e.g. pgvector, Qdrant, Pinecone).
    """
    q = question.lower()
    scored: list[tuple[int, str]] = []
    for key, passage in _KNOWLEDGE_BASE.items():
        keywords = key.split()
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            scored.append((score, passage))

    if not scored:
        return "No relevant documentation found for this query."

    # Return the best-matching passage (or two if there's a tie at the top)
    scored.sort(key=lambda x: -x[0])
    results = [p for _, p in scored[:1]]
    return "\n\n---\n\n".join(results)


def generate_answer(question: str, docs: str, prompt: str) -> str:
    """Call gpt-4o-mini with the system prompt and retrieved docs."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package required: pip install openai"
        )

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Documentation retrieved for this question:\n\n{docs}"
                    f"\n\n---\n\nQuestion: {question}"
                ),
            },
        ],
        max_tokens=300,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


# Load ideal answers from questions.json for reference
_QUESTIONS_FILE = Path(__file__).parent / "questions.json"
_IDEALS: dict[str, str] = {}
if _QUESTIONS_FILE.exists():
    for item in json.loads(_QUESTIONS_FILE.read_text()):
        _IDEALS[item["question"]] = item["ideal"]


def pipeline_fn(prompt: str, question: str) -> AgentTrace:
    """Run the full RAG pipeline for one question.

    This is the function reflex calls each iteration with the current
    candidate system prompt.  Both nodes execute fresh on every call —
    so prompt changes that teach the model to cite the docs actually
    work against the real retrieved content, not a static snapshot.

    Args:
        prompt:   The current system prompt candidate being evaluated.
        question: A developer question about the pyflow library.

    Returns:
        AgentTrace with two nodes: retrieve_docs and generate_answer.
    """
    docs = retrieve_docs(question)
    answer = generate_answer(question, docs, prompt)

    return AgentTrace(
        nodes=[
            TraceNode("retrieve_docs", input=question, output=docs),
            TraceNode(
                "generate_answer",
                input=question,
                output=answer,
                optimize=True,  # this is the node the prompt controls
            ),
        ],
        ideal=_IDEALS.get(question),
    )


# ---------------------------------------------------------------------------
# Manual test — run this file directly to verify the pipeline works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    prompt = Path(__file__).parent / "prompt.md"
    test_question = "How do I configure retry logic for failed pipeline steps?"

    print(f"Testing pipeline with prompt: {prompt}\n")
    print(f"Question: {test_question}\n")

    trace = pipeline_fn(
        prompt=prompt.read_text().strip(),
        question=test_question,
    )

    print("=== PIPELINE TRACE ===")
    print(trace.to_trace_text())
    print(f"\nIdeal: {trace.ideal}")
