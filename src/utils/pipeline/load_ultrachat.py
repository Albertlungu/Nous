"""
UltraChat Dataset Loader
Target: 10B tokens
Multi-turn conversations for context understanding
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_ultrachat(num_tokens=10_000_000_000, avg_tokens=1200):
    """
    Load UltraChat dataset (multi-turn conversations)

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per conversation (longer for multi-turn)

    Returns:
        List of formatted conversation strings
    """
    print("\n" + "="*60)
    print("Loading UltraChat (multi-turn conversations)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset("stingning/ultrachat", split="train", streaming=True)
        ds = ds.take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing UltraChat", total=num_examples):
            # UltraChat has conversations as list of messages
            if 'data' in example and isinstance(example['data'], list):
                conversation = example['data']

                # Build multi-turn conversation format
                formatted_conv = ""
                for i, message in enumerate(conversation):
                    if i % 2 == 0:  # User message
                        formatted_conv += f"User: {message}\n"
                    else:  # Assistant message
                        formatted_conv += f"Assistant: {message}\n"

                formatted.append(formatted_conv.strip())

        print(f"✓ Loaded {len(formatted):,} conversations from UltraChat")
        return formatted

    except Exception as e:
        print(f"✗ Error loading UltraChat: {e}")
        return []


if __name__ == "__main__":
    # Test the loader
    data = load_ultrachat(num_tokens=1_000_000)  # Test with 1M tokens
    print(f"\nSample conversation:\n{data[0][:500]}...")
