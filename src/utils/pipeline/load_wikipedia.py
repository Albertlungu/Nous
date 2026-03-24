"""
Wikipedia Dataset Loader
High-quality encyclopedic content. Standard anchor dataset for all major LLMs.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_wikipedia(num_tokens=5_000_000_000, avg_tokens=500):
    """
    Load English Wikipedia (wikimedia/wikipedia, 20231101.en)

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per article

    Returns:
        List of text strings
    """
    print("\n" + "="*60)
    print("Loading Wikipedia (wikimedia/wikipedia, 20231101.en)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing Wikipedia", total=num_examples):
            text = example.get("text", "").strip()
            if text:
                formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} articles from Wikipedia")
        return formatted

    except Exception as e:
        print(f"✗ Error loading Wikipedia: {e}")
        return []


def load_wikipedia_chunked(num_tokens=5_000_000_000, avg_tokens=500, chunk_size=10000):
    """
    Load Wikipedia in chunks to avoid RAM overflow.
    Yields batches of articles instead of loading everything at once.
    NO SHUFFLE - maintains deterministic order for checkpointing.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per article
        chunk_size: Number of articles per chunk

    Yields:
        List of text strings (chunk_size articles at a time)
    """
    print("\n" + "="*60)
    print("Loading Wikipedia (wikimedia/wikipedia, 20231101.en) in chunks...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Chunk size: {chunk_size:,} articles")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True
        )
        ds = ds.take(num_examples)  # NO SHUFFLE - deterministic order

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing Wikipedia", total=num_examples):
            text = example.get("text", "").strip()
            if text:
                chunk.append(text)

            if len(chunk) >= chunk_size:
                yield chunk
                total_processed += len(chunk)
                chunk = []

        # Yield remaining articles
        if chunk:
            yield chunk
            total_processed += len(chunk)

        print(f"✓ Processed {total_processed:,} articles from Wikipedia")

    except Exception as e:
        print(f"✗ Error loading Wikipedia: {e}")
        yield []


if __name__ == "__main__":
    data = load_wikipedia(num_tokens=1_000_000)
    print(f"\nSample:\n{data[0][:500]}...")
