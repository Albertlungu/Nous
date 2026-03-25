"""
FineWeb-Edu Dataset Loader
Educational web content filtered from Common Crawl.
Best-performing open web dataset for knowledge/reasoning benchmarks.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_fineweb(num_tokens=50_000_000_000, avg_tokens=650):
    """
    Load FineWeb-Edu (HuggingFaceFW/fineweb-edu)
    ~1.3T tokens of educational web text filtered from Common Crawl.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per document

    Returns:
        List of text strings
    """
    print("\n" + "="*60)
    print("Loading FineWeb-Edu (HuggingFaceFW/fineweb-edu)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-100BT",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing FineWeb-Edu", total=num_examples):
            text = example.get("text", "").strip()
            if text:
                formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} documents from FineWeb-Edu")
        return formatted

    except Exception as e:
        print(f"✗ Error loading FineWeb-Edu: {e}")
        return []


def load_fineweb_chunked(num_tokens=50_000_000_000, avg_tokens=650, chunk_size=10000):
    """
    Load FineWeb-Edu in chunks to avoid RAM overflow.
    Yields batches of documents instead of loading everything at once.
    NO SHUFFLE - maintains deterministic order for checkpointing.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per document
        chunk_size: Number of documents per chunk

    Yields:
        List of text strings (chunk_size documents at a time)
    """
    print("\n" + "="*60)
    print("Loading FineWeb-Edu (HuggingFaceFW/fineweb-edu) in chunks...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Chunk size: {chunk_size:,} documents")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-100BT",
            split="train",
            streaming=True
        )
        ds = ds.take(num_examples)  # NO SHUFFLE - deterministic order

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing FineWeb-Edu", total=num_examples):
            text = example.get("text", "").strip()
            if text:
                chunk.append(text)

            if len(chunk) >= chunk_size:
                yield chunk
                total_processed += len(chunk)
                chunk = []

        # Yield remaining documents
        if chunk:
            yield chunk
            total_processed += len(chunk)

        print(f"✓ Processed {total_processed:,} documents from FineWeb-Edu")

    except Exception as e:
        print(f"✗ Error loading FineWeb-Edu: {e}")
        yield []


if __name__ == "__main__":
    data = load_fineweb(num_tokens=1_000_000)
    print(f"\nSample:\n{data[0][:500]}...")
