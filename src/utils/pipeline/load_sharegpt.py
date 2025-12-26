"""
ShareGPT Dataset Loader
Target: 4B tokens
Real ChatGPT conversations (high quality multi-turn)
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_sharegpt(num_tokens=4_000_000_000, avg_tokens=1500):
    """
    Load ShareGPT dataset (real ChatGPT conversations)

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per conversation

    Returns:
        List of formatted conversation strings
    """
    print("\n" + "="*60)
    print("Loading ShareGPT (real ChatGPT conversations)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train")
        ds = ds.shuffle(seed=42).select(range(min(num_examples, len(ds))))

        formatted = []
        for example in tqdm(ds, desc="Processing ShareGPT"):
            # ShareGPT format: {conversations: [{from: "human/gpt", value: "..."}]}
            if 'conversations' in example:
                conversation = example['conversations']
                formatted_conv = ""

                for message in conversation:
                    role = "User" if message['from'] == 'human' else "Assistant"
                    formatted_conv += f"{role}: {message['value']}\n"

                formatted.append(formatted_conv.strip())

        print(f"✓ Loaded {len(formatted):,} conversations from ShareGPT")
        return formatted

    except Exception as e:
        print(f"✗ Error loading ShareGPT: {e}")
        return []


if __name__ == "__main__":
    # Test the loader
    data = load_sharegpt(num_tokens=100_000)  # Test with 100k tokens
    print(f"\nSample conversation:\n{data[0][:500]}...")
