"""
Code Dataset Loader
The Stack (deduplicated) — 6TB of permissively licensed source code.
Raw code files across major languages, no instruction formatting.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


LANGUAGES = ["python", "javascript", "java", "typescript", "cpp", "go", "rust", "c", "shell"]


def load_stack_dedup(num_tokens=15_000_000_000, avg_tokens=600):
    """
    Load The Stack deduplicated (bigcode/the-stack-dedup)
    6TB of permissively licensed code across 300+ languages, globally deduplicated.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per file

    Returns:
        List of raw code strings
    """
    print("\n" + "="*60)
    print("Loading The Stack dedup (bigcode/the-stack-dedup)...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Languages: {LANGUAGES}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)
    examples_per_lang = num_examples // len(LANGUAGES)

    formatted = []

    for lang in LANGUAGES:
        print(f"\nLoading {lang}...")
        try:
            ds = load_dataset(
                "bigcode/the-stack-dedup",
                data_dir=f"data/{lang}",
                split="train",
                streaming=True
            )
            ds = ds.take(examples_per_lang)

            for example in tqdm(ds, desc=f"Processing {lang}", total=examples_per_lang):
                code = example.get("content", "").strip()
                if code:
                    formatted.append(code)

        except Exception as e:
            print(f"  Warning: could not load {lang} — {e}")
            continue

    print(f"\n✓ Loaded {len(formatted):,} code files from The Stack")
    return formatted


def load_all_code(stack_tokens=15_000_000_000):
    """
    Load all code data.

    Returns:
        List of code strings
    """
    return load_stack_dedup(stack_tokens)


if __name__ == "__main__":
    data = load_all_code(stack_tokens=100_000)
    print(f"\nSample:\n{data[0][:500]}...")
