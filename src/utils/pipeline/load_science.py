"""
Science Dataset Loader
peS2o: 38M permissively licensed scientific papers from Semantic Scholar.
Covers STEM, medicine, social sciences.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_science(num_tokens=5_000_000_000, avg_tokens=1200):
    """
    Load peS2o scientific papers (allenai/peS2o)
    ~70B tokens of STEM papers from Semantic Scholar corpus.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per paper

    Returns:
        List of text strings
    """
    print("\n" + "="*60)
    print("Loading Science papers (allenai/peS2o)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "allenai/peS2o",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing peS2o", total=num_examples):
            text = example.get("text", "").strip()
            if text:
                formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} papers from peS2o")
        return formatted

    except Exception as e:
        print(f"✗ Error loading peS2o: {e}")
        return []


def load_science_chunked(num_tokens=5_000_000_000, avg_tokens=1200, chunk_size=10000):
    """
    Load peS2o scientific papers in chunks to avoid RAM overflow.
    Yields batches of papers instead of loading everything at once.
    NO SHUFFLE - maintains deterministic order for checkpointing.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per paper
        chunk_size: Number of papers per chunk

    Yields:
        List of text strings (chunk_size papers at a time)
    """
    print("\n" + "="*60)
    print("Loading Science papers (allenai/peS2o) in chunks...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Chunk size: {chunk_size:,} papers")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "allenai/peS2o",
            split="train",
            streaming=True
        )
        ds = ds.take(num_examples)  # NO SHUFFLE - deterministic order

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing peS2o", total=num_examples):
            text = example.get("text", "").strip()
            if text:
                chunk.append(text)

            if len(chunk) >= chunk_size:
                yield chunk
                total_processed += len(chunk)
                chunk = []

        # Yield remaining papers
        if chunk:
            yield chunk
            total_processed += len(chunk)

        print(f"✓ Processed {total_processed:,} papers from peS2o")

    except Exception as e:
        print(f"✗ Error loading peS2o: {e}")
        yield []


if __name__ == "__main__":
    data = load_science(num_tokens=1_000_000)
    print(f"\nSample:\n{data[0][:500]}...")
