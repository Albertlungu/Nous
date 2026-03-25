"""
ML ArXiv Papers Dataset Loader
Machine learning papers from arXiv.org.
Focused on ML/AI research papers.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_ml_arxiv(num_tokens=20_000_000_000, avg_tokens=1500):
    """
    Load ML ArXiv Papers (CShorten/ML-ArXiv-Papers)
    Machine learning papers from arXiv.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per paper

    Returns:
        List of text strings
    """
    print("\n" + "="*60)
    print("Loading ML ArXiv Papers (CShorten/ML-ArXiv-Papers)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "CShorten/ML-ArXiv-Papers",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing ML ArXiv Papers", total=num_examples):
            title = example.get("title", "").strip()
            abstract = example.get("abstract", "").strip()
            full_text = example.get("full_text", "").strip()

            if full_text:
                text = f"Title: {title}\n\nAbstract: {abstract}\n\n{full_text}"
            elif abstract:
                text = f"Title: {title}\n\nAbstract: {abstract}"
            else:
                continue

            formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} papers from ML ArXiv")
        return formatted

    except Exception as e:
        print(f"✗ Error loading ML ArXiv Papers: {e}")
        return []


def load_ml_arxiv_chunked(num_tokens=20_000_000_000, avg_tokens=1500, chunk_size=5000):
    """
    Load ML ArXiv Papers in chunks to avoid RAM overflow.
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
    print("Loading ML ArXiv Papers (CShorten/ML-ArXiv-Papers) in chunks...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Chunk size: {chunk_size:,} papers")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "CShorten/ML-ArXiv-Papers",
            split="train",
            streaming=True
        )
        ds = ds.take(num_examples)

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing ML ArXiv Papers", total=num_examples):
            title = example.get("title", "").strip()
            abstract = example.get("abstract", "").strip()
            full_text = example.get("full_text", "").strip()

            if full_text:
                text = f"Title: {title}\n\nAbstract: {abstract}\n\n{full_text}"
            elif abstract:
                text = f"Title: {title}\n\nAbstract: {abstract}"
            else:
                continue

            chunk.append(text)

            if len(chunk) >= chunk_size:
                yield chunk
                total_processed += len(chunk)
                chunk = []

        if chunk:
            yield chunk
            total_processed += len(chunk)

        print(f"✓ Processed {total_processed:,} papers from ML ArXiv")

    except Exception as e:
        print(f"✗ Error loading ML ArXiv Papers: {e}")
        yield []


if __name__ == "__main__":
    data = load_ml_arxiv(num_tokens=1_000_000)
    print(f"\nSample:\n{data[0][:500]}...")
