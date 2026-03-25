"""
PG-19 Books Dataset Loader
Project Gutenberg books - 28GB of long-form narrative text.
Out-of-copyright books for learning long-range dependencies.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_pg19(num_tokens=15_000_000_000, avg_tokens=2000):
    """
    Load PG-19 (emozilla/pg19)
    Project Gutenberg books for long-form text understanding.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per book

    Returns:
        List of text strings
    """
    print("\n" + "="*60)
    print("Loading PG-19 Books (emozilla/pg19)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "emozilla/pg19",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing PG-19", total=num_examples):
            text = example.get("text", "").strip()
            if text:
                formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} books from PG-19")
        return formatted

    except Exception as e:
        print(f"✗ Error loading PG-19: {e}")
        return []


def load_pg19_chunked(num_tokens=15_000_000_000, avg_tokens=2000, chunk_size=1000):
    """
    Load PG-19 in chunks to avoid RAM overflow.
    Yields batches of books instead of loading everything at once.
    NO SHUFFLE - maintains deterministic order for checkpointing.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per book
        chunk_size: Number of books per chunk

    Yields:
        List of text strings (chunk_size books at a time)
    """
    print("\n" + "="*60)
    print("Loading PG-19 Books (emozilla/pg19) in chunks...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Chunk size: {chunk_size:,} books")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "emozilla/pg19",
            split="train",
            streaming=True
        )
        ds = ds.take(num_examples)

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing PG-19", total=num_examples):
            text = example.get("text", "").strip()
            if text:
                chunk.append(text)

            if len(chunk) >= chunk_size:
                yield chunk
                total_processed += len(chunk)
                chunk = []

        if chunk:
            yield chunk
            total_processed += len(chunk)

        print(f"✓ Processed {total_processed:,} books from PG-19")

    except Exception as e:
        print(f"✗ Error loading PG-19: {e}")
        yield []


if __name__ == "__main__":
    data = load_pg19(num_tokens=1_000_000)
    print(f"\nSample:\n{data[0][:500]}...")
