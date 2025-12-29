"""
OpenOrca Dataset Loader
Target: 10B tokens
GPT-4/3.5 generated instructions for general knowledge
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_openorca(num_tokens=10_000_000_000, avg_tokens=800):
    """
    Load OpenOrca dataset

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per example

    Returns:
        List of formatted text strings
    """
    print("\n" + "="*60)
    print("Loading OpenOrca dataset...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        ds = ds.take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing OpenOrca", total=num_examples):
            # Combine system prompt with question
            instruction = f"{example['system_prompt']}\n\n{example['question']}"
            text = f"Instruction: {instruction}\nOutput: {example['response']}"
            formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} examples from OpenOrca")
        return formatted

    except Exception as e:
        print(f"✗ Error loading OpenOrca: {e}")
        return []


if __name__ == "__main__":
    # Test the loader
    data = load_openorca(num_tokens=1_000_000)  # Test with 1M tokens
    print(f"\nSample example:\n{data[0][:500]}...")
