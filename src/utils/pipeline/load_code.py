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


def load_stack_dedup_chunked(num_tokens=15_000_000_000, avg_tokens=600, chunk_size=10000):
    """
    Load The Stack deduplicated in chunks to avoid RAM overflow.
    Yields batches of code files instead of loading everything at once.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per file
        chunk_size: Number of files per chunk

    Yields:
        List of raw code strings (chunk_size files at a time)
    """
    print("\n" + "="*60)
    print("Loading The Stack dedup (bigcode/the-stack-dedup) in chunks...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Languages: {LANGUAGES}")
    print(f"Chunk size: {chunk_size:,} files")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)
    examples_per_lang = num_examples // len(LANGUAGES)

    chunk = []
    total_processed = 0

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
                    chunk.append(code)

                if len(chunk) >= chunk_size:
                    yield chunk
                    total_processed += len(chunk)
                    chunk = []

        except Exception as e:
            print(f"  Warning: could not load {lang} — {e}")
            continue

    # Yield remaining files
    if chunk:
        yield chunk
        total_processed += len(chunk)

    print(f"\n✓ Processed {total_processed:,} code files from The Stack")


def load_all_code(stack_tokens=15_000_000_000):
    """
    Load all code data.

    Returns:
        List of code strings
    """
    return load_stack_dedup(stack_tokens)


def load_all_code_chunked(stack_tokens=15_000_000_000, chunk_size=10000):
    """
    Load all code data in chunks.

    Yields:
        List of code strings (chunk_size files at a time)
    """
    return load_stack_dedup_chunked(stack_tokens, chunk_size=chunk_size)


if __name__ == "__main__":
    data = load_all_code(stack_tokens=100_000)
    print(f"\nSample:\n{data[0][:500]}...")
