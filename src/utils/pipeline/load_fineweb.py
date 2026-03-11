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


if __name__ == "__main__":
    data = load_fineweb(num_tokens=1_000_000)
    print(f"\nSample:\n{data[0][:500]}...")
