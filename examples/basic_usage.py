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

"""Basic usage example for aevyra-reflex.

Demonstrates both the iterative and PDO strategies using verdict for evaluation.

Prerequisites:
  - pip install aevyra-reflex
  - ANTHROPIC_API_KEY set (for the optimization agent)
  - API key for whichever provider you're targeting
"""

from aevyra_verdict import Dataset, RougeScore
from aevyra_reflex import OptimizerConfig, PromptOptimizer


def run_iterative():
    """Simple iterative optimization — diagnose and revise."""
    dataset = Dataset.from_jsonl("examples/sample_data.jsonl")

    config = OptimizerConfig(
        strategy="iterative",
        max_iterations=5,
        score_threshold=0.80,
    )

    result = (
        PromptOptimizer(config)
        .set_dataset(dataset)
        .add_provider("openai", "gpt-5.4-nano")
        .add_metric(RougeScore())
        .run("You are a helpful assistant. Answer accurately and concisely.")
    )

    print(result.summary())
    print()
    print("Best prompt:")
    print(result.best_prompt)


def run_pdo():
    """PDO strategy — dueling bandits with Thompson sampling."""
    dataset = Dataset.from_jsonl("examples/sample_data.jsonl")

    config = OptimizerConfig(
        strategy="pdo",
        max_iterations=30,
        score_threshold=0.85,
        extra_kwargs={
            "duels_per_round": 3,
            "samples_per_duel": 10,
            "initial_pool_size": 6,
        },
    )

    result = (
        PromptOptimizer(config)
        .set_dataset(dataset)
        .add_provider("openai", "gpt-5.4-nano")
        .add_metric(RougeScore())
        .run("You are a helpful assistant.")
    )

    print(result.summary())
    result.save_best_prompt("best_prompt.md")
    result.to_json("optimization_results.json")
    print("Results saved.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "pdo":
        run_pdo()
    else:
        run_iterative()
