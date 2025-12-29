"""
FLAN v2 Dataset Loader
Target: 8B tokens
Diverse tasks with Chain-of-Thought reasoning
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_flan(num_tokens=8_000_000_000, avg_tokens=800):
    """
    Load FLAN v2 dataset

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per example

    Returns:
        List of formatted text strings
    """
    print("\n" + "="*60)
    print("Loading FLAN v2 dataset...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset("conceptofmind/flan2021_submix_original", split="train", streaming=True)
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing FLAN v2", total=num_examples):
            # FLAN format: {inputs, targets}
            text = f"Instruction: {example['inputs']}\nOutput: {example['targets']}"
            formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} examples from FLAN v2")
        return formatted

    except Exception as e:
        print(f"✗ Error loading FLAN v2: {e}")
        return []


if __name__ == "__main__":
    # Test the loader
    data = load_flan(num_tokens=1_000_000)  # Test with 1M tokens
    print(f"\nSample example:\n{data[0][:500]}...")
