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


if __name__ == "__main__":
    data = load_books(num_tokens=10_000_000)
    print(f"\nSample (first 500 chars):\n{data[0][:500]}...")
