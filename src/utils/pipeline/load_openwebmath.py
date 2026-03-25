"""
OpenWebMath Dataset Loader
~15B tokens of high-quality mathematical text from Common Crawl.
Filtered for mathematical reasoning, proofs, and problem solving.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_openwebmath(num_tokens=30_000_000_000, avg_tokens=800):
    """
    Load OpenWebMath (open-web-math/open-web-math)
    ~15B tokens of mathematical web text from Common Crawl.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per document

    Returns:
        List of text strings
    """
    print("\n" + "="*60)
    print("Loading OpenWebMath (open-web-math/open-web-math)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "open-web-math/open-web-math",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing OpenWebMath", total=num_examples):
            text = example.get("text", "").strip()
            if text:
                formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} documents from OpenWebMath")
        return formatted

    except Exception as e:
        print(f"✗ Error loading OpenWebMath: {e}")
        return []


def load_openwebmath_chunked(num_tokens=30_000_000_000, avg_tokens=800, chunk_size=10000):
    """
    Load OpenWebMath in chunks to avoid RAM overflow.
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
    print("Loading OpenWebMath (open-web-math/open-web-math) in chunks...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Chunk size: {chunk_size:,} documents")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "open-web-math/open-web-math",
            split="train",
            streaming=True
        )
        ds = ds.take(num_examples)

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing OpenWebMath", total=num_examples):
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

        print(f"✓ Processed {total_processed:,} documents from OpenWebMath")

    except Exception as e:
        print(f"✗ Error loading OpenWebMath: {e}")
        yield []


if __name__ == "__main__":
    data = load_openwebmath(num_tokens=1_000_000)
    print(f"\nSample:\n{data[0][:500]}...")
