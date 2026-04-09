# aevyra-reflex

[![CI](https://github.com/aevyraai/reflex/actions/workflows/ci.yml/badge.svg)](https://github.com/aevyraai/reflex/actions/workflows/ci.yml)

Agentic prompt optimization. Reflex reads your eval results, diagnoses why
your model is underperforming, and rewrites the prompt until the scores
match your target — no manual prompt engineering required.

Point it at a dataset, a model, and a target score. Reflex figures out the rest.

```bash
aevyra-reflex optimize dataset.jsonl prompt.md -m local/llama3.1 -o best_prompt.md
```

Reflex is an agent, not a script. It observes eval results, reasons about
failure patterns, and adaptively picks the right optimization technique at
each step. By default it draws from four axes:

- **structural** — reorganize the prompt's layout, formatting, and information
  hierarchy. Inspired by Microsoft's SAMMO.
- **iterative** — diagnose specific failure patterns and surgically revise the
  wording.
- **fewshot** — curate the most informative few-shot examples from the dataset.
  Inspired by Stanford's DSPy.
- **pdo** — tournament-style search over prompt variants using dueling bandits
  with Thompson sampling. Based on Meta's PDO paper
  ([arXiv:2510.13907](https://arxiv.org/abs/2510.13907)).

Each axis can also be used standalone with `-s iterative`, `-s pdo`, etc.

All evaluation runs through [aevyra-verdict](https://github.com/aevyraai/verdict).
For reasoning, reflex uses Claude by default — but you can swap in any model,
including local ones like Qwen3-8B or DeepSeek-R1.

## Install

```bash
pip install aevyra-reflex
```

Requires `aevyra-verdict`. By default, uses Claude for reasoning
(`ANTHROPIC_API_KEY` env var), but you can use any model — see
[Choosing a reasoning model](#choosing-a-reasoning-model) below.

## Quick start

One command does the whole thing — runs a baseline eval, optimizes the prompt,
re-evaluates with the improved prompt, and shows a before/after comparison:

```bash
aevyra-reflex optimize dataset.jsonl prompt.md -m local/llama3.1 -o best_prompt.md
```

Output:

```
====================================================
  aevyra-reflex
====================================================
  Dataset    : dataset.jsonl (50 samples)
  Split      : 40 train / 10 test (80% / 20%)
  Model(s)   : local/llama3.1
  Strategy   : auto
  Metrics    : rouge
  Reasoning  : claude-sonnet-4-20250514
  Target     : 0.85
====================================================

Step 1/3  Running baseline eval...

Step 2/3  Optimizing...
  iteration  1  score: 0.6234
  iteration  2  score: 0.7105
  iteration  3  score: 0.8612

Step 3/3  Verifying...

====================================================
  OPTIMIZATION RESULTS
====================================================
  Train / test     : 40 / 10 samples
  Baseline score   : 0.5821  (on 10-sample test set)
  Final score      : 0.8612  (on 10-sample test set)
  Improvement      : +0.2791 (+47.9%)
  Iterations       : 3
  Converged        : True
----------------------------------------------------
  Per-metric breakdown:
    rouge                           0.5821 → 0.8612  (+0.2791)
----------------------------------------------------
  Trajectory : 0.623 → 0.711 → 0.861
====================================================

Best prompt saved to: best_prompt.md
```

Or use the Python API:

```python
from aevyra_verdict import Dataset, RougeScore
from aevyra_reflex import PromptOptimizer, OptimizerConfig

dataset = Dataset.from_jsonl("dataset.jsonl")

config = OptimizerConfig(
    strategy="auto",             # or "iterative", "pdo", "fewshot", "structural"
    max_iterations=10,
    score_threshold=0.85,
)

result = (
    PromptOptimizer(config)
    .set_dataset(dataset)
    .add_provider("openai", "gpt-5.4-nano")
    .add_metric(RougeScore())
    .run("You are a helpful assistant.")
)

print(result.summary())
print(result.best_prompt)
```

## How it works

### Auto strategy (default)

The auto strategy runs a multi-phase pipeline:

1. Run a baseline eval to measure the starting score
2. The reasoning model analyzes the prompt's weaknesses and recommends an optimization axis
3. Apply that axis for a few iterations (each axis has its own budget)
4. Re-evaluate — if the threshold is met, stop
5. Otherwise the reasoning model picks the next axis based on what changed
6. Repeat until the global iteration budget runs out

A typical auto run might look like: structural (fix formatting) → iterative
(fix wording) → fewshot (add examples) — each phase building on the previous
one's improvements.

### Iterative strategy

Each iteration:

1. Inject the current system prompt into every conversation in the dataset
2. Run completions against the target model via verdict
3. Score all responses with the configured metrics
4. Identify the worst-scoring samples
5. Send them to the reasoning model: "here's the prompt, here are the failures —
   what's wrong and how should we fix it?"
6. The reasoning model proposes a revised prompt grounded in the actual failure patterns
7. If the score meets the threshold, stop. Otherwise repeat with the new prompt.

The reasoning model maintains a **causal rewrite log** across iterations — a compact record of what changed each round and whether it helped. From iteration 2 onwards this history is fed back into the prompt, so the model knows which edits helped (✓), had no effect (✗ no effect), or hurt (✗ hurt) and can avoid repeating dead ends:

```
Iter 1 (score: 0.6234, Δ+0.0871 — ✓ helped): Added numbered reasoning steps
Iter 2 (score: 0.7105, Δ+0.0029 — ✗ no effect): Added "think carefully" instruction
```

### PDO strategy

Maintains a pool of candidate prompts and uses dueling bandits to find the best:

1. Generate an initial pool of diverse prompts from the base instruction
2. Each round, Thompson sampling selects two prompts to duel
3. Both prompts are evaluated on a sample of the dataset
4. An LLM judge picks the winner on each sample; majority wins the duel
5. Win matrix is updated; Copeland rankings are recalculated
6. Periodically, the top-ranked prompts are mutated to generate new candidates
7. Worst performers are pruned to keep the pool manageable

The PDO strategy is inspired by Meta's Prompt Duel Optimizer but rebuilt on
top of verdict's evaluation infrastructure.

### Few-shot strategy

Optimizes *which examples* to include in the prompt:

1. Bootstrap: run the bare instruction and collect the highest-scoring samples
   as candidate exemplars
2. Ask the reasoning model to select a diverse, informative subset of examples
3. Build a composite prompt: instruction + curated few-shot examples
4. Run eval, identify remaining failures
5. The reasoning model swaps examples to better cover the failure modes
6. Periodically re-bootstrap to discover new exemplar candidates

### Structural strategy

Optimizes the *organization and formatting* of the prompt:

1. Run eval with the current prompt structure
2. The reasoning model analyzes structural weaknesses (section ordering,
   formatting, hierarchy, constraint placement)
3. Generate variants using different transformations (markdown headers,
   XML tags, flat paragraphs, role/task/format split, etc.)
4. The reasoning model also generates a free-form structural improvement
5. Evaluate all variants; keep the best
6. Repeat, accumulating knowledge about which structures help or hurt

Inspired by Microsoft's SAMMO framework for structure-aware prompt optimization.

## Parallel execution

Strategies like `structural` and `pdo` evaluate multiple prompt variants per
iteration. These variants are evaluated **in parallel** using threads. For cloud
APIs (OpenAI, Anthropic) this works out of the box.

For **Ollama**, you need to tell it to handle parallel requests — by default it
processes one request at a time:

```bash
# 1. Start Ollama with parallel inference enabled
OLLAMA_NUM_PARALLEL=4 ollama serve

# 2. Tell reflex to use matching parallelism
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m local/llama3.2:8b \
  --max-workers 4 \
  -o best_prompt.md
```

Higher `OLLAMA_NUM_PARALLEL` uses more VRAM. For a 8B model on a GPU with 8GB+
VRAM, 4 is a good starting point. For 1B models, you can go higher.

If you don't set `OLLAMA_NUM_PARALLEL`, reflex auto-detects this and falls back
to sequential execution (1 worker), logging a warning with setup instructions.

## Run persistence and resume

Every run is checkpointed to a `.reflex/` directory. If a run crashes or
is interrupted, resume from where it left off — no lost work:

```bash
# Resume the latest interrupted run
aevyra-reflex optimize dataset.jsonl prompt.md -m local/llama3.1 --resume

# Resume a specific run by ID
aevyra-reflex optimize dataset.jsonl prompt.md -m local/llama3.1 --resume-from 003

# List all runs
aevyra-reflex runs
```

```
  ID  Status        Strategy      Iters  Baseline      Best     Final  Dataset
------------------------------------------------------------------------------------------
 001  ✓ completed   auto              5    0.5821    0.8612    0.8612  dataset.jsonl
 002  ⚡ interrupted  iterative        3    0.6100    0.7450         —  dataset.jsonl
```

Each run captures its config, dataset path, initial prompt, and per-iteration
state, so every experiment is reproducible and comparable.

## Honest eval scores with train/test split

By default, reflex holds out 20% of your dataset for evaluation. The optimization
loop only sees the training examples — the held-out test set is used exclusively
for the baseline and final scores you see in the results summary.

This prevents the reported improvement from being inflated by the same examples
that drove the rewrites. The split is deterministic (seed 42), so results are
reproducible.

```bash
# Default: 80/20 split
aevyra-reflex optimize dataset.jsonl prompt.md -m local/llama3.1

# Custom split
aevyra-reflex optimize dataset.jsonl prompt.md -m local/llama3.1 --train-split 0.9

# No split (useful for very small datasets < 20 examples)
aevyra-reflex optimize dataset.jsonl prompt.md -m local/llama3.1 --train-split 1.0
```

## Migrating a prompt to a new model

If you've written a prompt for Claude and want to optimize it for Llama or GPT-4o,
use `--source-model` to tell reflex which model family it's migrating *from*. The
reasoning model uses this to adapt model-family idioms automatically — XML tags to
Markdown headers, role framing structure, verbosity adjustments, and so on:

```bash
# Migrate a Claude prompt to Llama 3.1
aevyra-reflex optimize dataset.jsonl claude_prompt.md \
  -m local/llama3.1 \
  --source-model claude-sonnet \
  -o llama_prompt.md

# Migrate a GPT-4o prompt to Qwen3
aevyra-reflex optimize dataset.jsonl gpt4o_prompt.md \
  -m local/qwen3:8b \
  --source-model gpt-4o \
  -o qwen3_prompt.md
```

Or via the Python API:

```python
config = OptimizerConfig(
    strategy="iterative",
    source_model="claude-sonnet",   # prompt was written for this model
)
```

Without `--source-model`, reflex still optimizes the prompt — it just won't have
the explicit migration context to guide its rewrites.

## Validation split and early stopping

Add a validation set to detect overfitting mid-run. Reflex evaluates each
candidate prompt on the val examples after every iteration and logs both the
train and val scores. If train keeps climbing but val plateaus, the prompt is
fitting training examples specifically. Use `--early-stopping-patience` to
stop automatically when that happens:

```bash
# 3-way split: 70% train / 10% val / 20% test
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m local/llama3.1 \
  --train-split 0.8 \
  --val-split 0.1

# Same split, stop early when val stagnates for 3 consecutive iterations
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m local/llama3.1 \
  --train-split 0.8 \
  --val-split 0.1 \
  --early-stopping-patience 3
```

The summary shows the full picture:

```
  Train/val/test   : 70 / 10 / 20 samples
  Baseline score   : 0.5500  (on 20-sample test set)
  Final score      : 0.7100  (on 20-sample test set)
  Improvement      : +0.1600 (+29.1%)
  Iterations       : 6
  Early stopped    : Yes (val score plateaued)
  Train traj   : 0.600 → 0.650 → 0.710 → 0.720 → 0.725 → 0.724
  Val traj     : 0.580 → 0.640 → 0.690 → 0.688 → 0.685 → 0.682
```

When early stopping triggers, reflex returns the prompt with the best
validation score — not the latest one — so the final prompt generalizes
rather than fitting the training examples specifically.

## Statistical significance

After every run, reflex tests whether the improvement is real or noise — a
paired test on per-sample scores (Wilcoxon signed-rank via scipy, paired t-test
fallback). The result appears in the summary:

```
  Baseline score   : 0.5821
  Final score      : 0.8612
  Improvement      : +0.2791 (+47.9%)
  Significance     : p=0.0021  ✓ significant (α=0.05, paired test)
```

For noisy tasks where LLM responses vary run-to-run, average multiple eval
passes with `--eval-runs`:

```bash
# Average 3 eval passes for baseline and final, report mean ± std
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m local/llama3.1 \
  --eval-runs 3
```

```
  Baseline score   : 0.5821 ± 0.0180  (3 runs)
  Final score      : 0.8612 ± 0.0110  (3 runs)
```

Install scipy for the Wilcoxon test (recommended); the t-test fallback works
without it:

```bash
pip install "aevyra-reflex[stats]"
```

## Choosing a reasoning model

By default, reflex uses Claude Sonnet for reasoning — analyzing failures,
proposing prompt rewrites, and deciding which strategy to apply. You can swap
in any model with `--reasoning-model`:

```bash
# Use Qwen3-8B locally via Ollama (good balance of reasoning + speed)
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m local/llama3.2:1b \
  --reasoning-model ollama/qwen3:8b \
  -o best_prompt.md

# DeepSeek R1 distill for stronger math/logic reasoning
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m local/llama3.2:1b \
  --reasoning-model ollama/deepseek-r1:8b \
  -o best_prompt.md

# OpenAI
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m local/llama3.1 \
  --reasoning-model openai/gpt-4o \
  -o best_prompt.md

# Any OpenAI-compatible endpoint (vLLM, TGI, etc.)
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m local/llama3.1 \
  --reasoning-model openai/my-model \
  --reasoning-base-url http://localhost:8000/v1
```

Or via the Python API:

```python
config = OptimizerConfig(
    reasoning_model="qwen3:8b",
    reasoning_provider="ollama",
)
```

Qwen3-8B is a strong choice for local reasoning — it supports a thinking mode
for complex analysis, 131K context, and runs well on consumer GPUs. For harder
tasks, DeepSeek-R1-0528-Qwen3-8B edges it out on math/logic benchmarks.

## Dashboard

Explore your runs visually with the built-in web dashboard:

```bash
aevyra-reflex dashboard
```

Opens a local web UI at `http://localhost:8128` showing all your runs with
score trajectory charts, prompt diffs between iterations, reasoning analysis,
and config snapshots. Click into any run to drill down into individual
iterations and see exactly what the reasoning model changed and why.

Each iteration card shows **eval token** and **reasoning token** counts so you
can track model usage without leaving the UI.

**Branch runs** let you pick any iteration from a completed or interrupted run
and continue optimizing from that point with a different strategy — no
baseline re-evaluation required. Hover over any iteration card in the flow
graph and click the `⎇` button that appears. Branch runs appear indented
directly below their parent in the runs list.

```bash
# Custom port or run directory
aevyra-reflex dashboard --port 9000 --run-dir ./experiments/.reflex
```

## CLI

```bash
# Auto (default) — just run it, auto picks the strategies
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m local/llama3.1 \
  --max-iterations 20 \
  --max-workers 4 \
  -o best_prompt.md

# Iterative only
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m openai/gpt-5.4-nano \
  -s iterative \
  --metric rouge \
  --threshold 0.85 \
  -o best_prompt.md

# PDO with more rounds
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m local/llama3.1 \
  -s pdo \
  --max-iterations 50 \
  -o best_prompt.md \
  --results-json results.json

# Few-shot example optimization
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m local/llama3.1 \
  -s fewshot \
  -o best_prompt.md

# Structural optimization
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m local/llama3.1 \
  -s structural \
  -o best_prompt.md

# Add an LLM judge alongside ROUGE
aevyra-reflex optimize dataset.jsonl prompt.md \
  -m openai/gpt-5.4-nano \
  --metric rouge \
  --judge openai/gpt-5.4 \
  -o best_prompt.md
```

## Configuration

```python
from aevyra_reflex import OptimizerConfig

# Auto (default) — just set budget and threshold, auto handles the rest
config = OptimizerConfig(
    strategy="auto",
    max_iterations=20,          # total budget across all phases
    score_threshold=0.85,
    extra_kwargs={
        "max_phases": 4,        # max number of optimization phases
        "start_structural": True,  # always start with structural
        # Override per-phase budgets:
        # "phase_budgets": {"structural": 3, "iterative": 4, "fewshot": 3, "pdo": 15},
    },
)

# Iterative
config = OptimizerConfig(
    strategy="iterative",
    max_iterations=10,
    score_threshold=0.85,
    reasoning_model="claude-sonnet-4-20250514",
    eval_temperature=0.0,
)

# PDO — pass strategy-specific params via extra_kwargs
config = OptimizerConfig(
    strategy="pdo",
    max_iterations=50,        # total rounds
    score_threshold=0.90,
    extra_kwargs={
        "duels_per_round": 3,
        "samples_per_duel": 10,
        "initial_pool_size": 6,
        "thompson_alpha": 1.2,
        "mutation_frequency": 5,
        "num_top_to_mutate": 2,
        "max_pool_size": 20,
    },
)
# Few-shot
config = OptimizerConfig(
    strategy="fewshot",
    max_iterations=8,
    score_threshold=0.85,
    extra_kwargs={
        "max_examples": 5,           # examples to include in prompt
        "candidate_pool_size": 20,    # exemplars to bootstrap
        "bootstrap_rounds": 3,       # re-bootstrap every N iterations
        "selection_strategy": "diverse",
    },
)

# Structural
config = OptimizerConfig(
    strategy="structural",
    max_iterations=6,
    score_threshold=0.85,
    extra_kwargs={
        "variants_per_round": 4,     # structural variants to try per iteration
    },
)
```

## Status

> Core implementation is complete. All four strategies (iterative, PDO,
> fewshot, structural) are functional. The public API (`PromptOptimizer`,
> `OptimizerConfig`, `OptimizationResult`) is stable.

## Contributing

Open an issue before starting any significant work.

## License

Apache 2.0
