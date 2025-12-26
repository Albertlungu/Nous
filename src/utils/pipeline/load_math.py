"""
Math Dataset Loaders
Target: 5B tokens total (3B from Orca-Math + 2B from MetaMathQA)
Mathematical reasoning and problem solving
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_orca_math(num_tokens=3_000_000_000, avg_tokens=700):
    """
    Load Orca-Math dataset (math word problems with reasoning)

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per math problem

    Returns:
        List of formatted math problem strings
    """
    print("\n" + "="*60)
    print("Loading Orca-Math...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
        ds = ds.shuffle(seed=42).select(range(min(num_examples, len(ds))))

        formatted = []
        for example in tqdm(ds, desc="Processing Orca-Math"):
            text = f"Instruction: {example['question']}\nOutput: {example['answer']}"
            formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} math problems from Orca-Math")
        return formatted

    except Exception as e:
        print(f"✗ Error loading Orca-Math: {e}")
        return []


def load_metamath(num_tokens=2_000_000_000, avg_tokens=600):
    """
    Load MetaMathQA dataset (math reasoning)

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per example

    Returns:
        List of formatted math reasoning strings
    """
    print("\n" + "="*60)
    print("Loading MetaMathQA...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
        ds = ds.take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing MetaMathQA", total=num_examples):
            text = f"Instruction: {example['query']}\nOutput: {example['response']}"
            formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} examples from MetaMathQA")
        return formatted

    except Exception as e:
        print(f"✗ Error loading MetaMathQA: {e}")
        return []


def load_all_math(orca_tokens=3_000_000_000, meta_tokens=2_000_000_000):
    """
    Load all math datasets

    Returns:
        Combined list of math examples
    """
    all_math = []

    # Load Orca-Math
    orca_data = load_orca_math(orca_tokens)
    all_math.extend(orca_data)

    # Load MetaMathQA
    meta_data = load_metamath(meta_tokens)
    all_math.extend(meta_data)

    print(f"\n{'='*60}")
    print(f"Total math examples: {len(all_math):,}")
    print(f"{'='*60}")

    return all_math


if __name__ == "__main__":
    # Test the loader
    data = load_orca_math(num_tokens=100_000)  # Test with 100k tokens
    print(f"\nSample math problem:\n{data[0][:500]}...")
