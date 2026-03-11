"""
DCLM-Baseline Dataset Loader
4T token filtered web dataset from DataComp-LM.
Strong general web content with model-based quality filtering.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_dclm(num_tokens=20_000_000_000, avg_tokens=700):
    """
    Load DCLM-Baseline (mlfoundations/dclm-baseline-1.0)
    4T tokens of web text filtered via fastText classifier.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per document

    Returns:
        List of text strings
    """
    print("\n" + "="*60)
    print("Loading DCLM-Baseline (mlfoundations/dclm-baseline-1.0)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "mlfoundations/dclm-baseline-1.0",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing DCLM", total=num_examples):
            text = example.get("text", "").strip()
            if text:
                formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} documents from DCLM-Baseline")
        return formatted

    except Exception as e:
        print(f"✗ Error loading DCLM-Baseline: {e}")
        return []


if __name__ == "__main__":
    data = load_dclm(num_tokens=1_000_000)
    print(f"\nSample:\n{data[0][:500]}...")
