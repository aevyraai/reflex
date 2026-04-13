"""
Download 100 examples from CNN/DailyMail and convert to reflex JSONL format.

Usage:
    pip install datasets
    python examples/create_cnn_dailymail.py
"""

import json
import random
import re

from datasets import load_dataset


def extract_ideal(highlights: str) -> str:
    """Use the highlights field as the ideal summary — clean up whitespace."""
    return " ".join(highlights.split())


def main():
    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")

    # Reproducible sample
    rng = random.Random(42)
    indices = rng.sample(range(len(dataset)), 100)
    examples = [dataset[i] for i in indices]

    output_path = "examples/cnn_dailymail.jsonl"
    with open(output_path, "w") as f:
        for ex in examples:
            article = ex["article"].strip()
            ideal = extract_ideal(ex["highlights"])

            # Truncate very long articles to ~800 words to keep inference fast
            words = article.split()
            if len(words) > 800:
                article = " ".join(words[:800]) + "..."

            record = {
                "messages": [
                    {
                        "role": "user",
                        "content": article,
                    }
                ],
                "ideal": ideal,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Written {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
