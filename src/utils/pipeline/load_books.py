"""
Books Dataset Loader
Project Gutenberg books (pre-1919, public domain).
Long-form text for coherence and narrative understanding.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_books(num_tokens=5_000_000_000, avg_tokens=150_000):
    """
    Load PG-19 books (deepmind/pg19)
    ~1.9B tokens of Project Gutenberg books published before 1919.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per book (books are very long)

    Returns:
        List of text strings
    """
    print("\n" + "="*60)
    print("Loading Books (deepmind/pg19)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "deepmind/pg19",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing Books", total=num_examples):
            text = example.get("text", "").strip()
            if text:
                formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} books from PG-19")
        return formatted

    except Exception as e:
        print(f"✗ Error loading PG-19: {e}")
        return []


def load_books_chunked(num_tokens=5_000_000_000, avg_tokens=150_000, chunk_size=100):
    """
    Load PG-19 books in chunks to avoid RAM overflow.
    Yields batches of books instead of loading everything at once.
    NO SHUFFLE - maintains deterministic order for checkpointing.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per book (books are very long)
        chunk_size: Number of books per chunk (smaller because books are huge)

    Yields:
        List of text strings (chunk_size books at a time)
    """
    print("\n" + "="*60)
    print("Loading Books (deepmind/pg19) in chunks...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Chunk size: {chunk_size:,} books")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "deepmind/pg19",
            split="train",
            streaming=True
        )
        ds = ds.take(num_examples)  # NO SHUFFLE - deterministic order

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing Books", total=num_examples):
            text = example.get("text", "").strip()
            if text:
                chunk.append(text)

            if len(chunk) >= chunk_size:
                yield chunk
                total_processed += len(chunk)
                chunk = []

        # Yield remaining books
        if chunk:
            yield chunk
            total_processed += len(chunk)

        print(f"✓ Processed {total_processed:,} books from PG-19")

    except Exception as e:
        print(f"✗ Error loading PG-19: {e}")
        yield []


if __name__ == "__main__":
    data = load_books(num_tokens=10_000_000)
    print(f"\nSample (first 500 chars):\n{data[0][:500]}...")
