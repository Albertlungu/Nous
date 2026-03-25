"""
GSM8K Dataset Loader
Grade School Math 8K - math word problems with step-by-step solutions.
Includes both original GSM8K and enhanced version with reasoning.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_gsm8k(num_tokens=5_000_000_000, avg_tokens=400):
    """
    Load GSM8K (openai/gsm8k)
    8.5K grade school math word problems with solutions.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per problem

    Returns:
        List of text strings (problem + solution)
    """
    print("\n" + "="*60)
    print("Loading GSM8K (openai/gsm8k)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "openai/gsm8k",
            "main",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing GSM8K", total=num_examples):
            question = example.get("question", "").strip()
            answer = example.get("answer", "").strip()

            if question and answer:
                text = f"Question: {question}\n\nAnswer: {answer}"
                formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} problems from GSM8K")
        return formatted

    except Exception as e:
        print(f"✗ Error loading GSM8K: {e}")
        return []


def load_gsm8k_chunked(num_tokens=5_000_000_000, avg_tokens=400, chunk_size=5000):
    """
    Load GSM8K in chunks to avoid RAM overflow.
    Yields batches of problems instead of loading everything at once.
    NO SHUFFLE - maintains deterministic order for checkpointing.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per problem
        chunk_size: Number of problems per chunk

    Yields:
        List of text strings (chunk_size problems at a time)
    """
    print("\n" + "="*60)
    print("Loading GSM8K (openai/gsm8k) in chunks...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Chunk size: {chunk_size:,} problems")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "openai/gsm8k",
            "main",
            split="train",
            streaming=True
        )
        ds = ds.take(num_examples)

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing GSM8K", total=num_examples):
            question = example.get("question", "").strip()
            answer = example.get("answer", "").strip()

            if question and answer:
                text = f"Question: {question}\n\nAnswer: {answer}"
                chunk.append(text)

            if len(chunk) >= chunk_size:
                yield chunk
                total_processed += len(chunk)
                chunk = []

        if chunk:
            yield chunk
            total_processed += len(chunk)

        print(f"✓ Processed {total_processed:,} problems from GSM8K")

    except Exception as e:
        print(f"✗ Error loading GSM8K: {e}")
        yield []


def load_gsm8k_enhanced(num_tokens=10_000_000_000, avg_tokens=500):
    """
    Load enhanced GSM8K (gabrielaltay/gsm8k-math-reasoning)
    GSM8K with improved reasoning explanations.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per problem

    Returns:
        List of text strings
    """
    print("\n" + "="*60)
    print("Loading GSM8K Enhanced (gabrielaltay/gsm8k-math-reasoning)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "gabrielaltay/gsm8k-math-reasoning",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing GSM8K Enhanced", total=num_examples):
            question = example.get("question", "").strip()
            reasoning = example.get("reasoning", "").strip()
            answer = example.get("answer", "").strip()

            if question:
                text = f"Question: {question}"
                if reasoning:
                    text += f"\n\nReasoning: {reasoning}"
                if answer:
                    text += f"\n\nAnswer: {answer}"
                formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} problems from GSM8K Enhanced")
        return formatted

    except Exception as e:
        print(f"✗ Error loading GSM8K Enhanced: {e}")
        return []


def load_gsm8k_enhanced_chunked(num_tokens=10_000_000_000, avg_tokens=500, chunk_size=5000):
    """
    Load enhanced GSM8K in chunks to avoid RAM overflow.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per problem
        chunk_size: Number of problems per chunk

    Yields:
        List of text strings (chunk_size problems at a time)
    """
    print("\n" + "="*60)
    print("Loading GSM8K Enhanced (gabrielaltay/gsm8k-math-reasoning) in chunks...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Chunk size: {chunk_size:,} problems")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "gabrielaltay/gsm8k-math-reasoning",
            split="train",
            streaming=True
        )
        ds = ds.take(num_examples)

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing GSM8K Enhanced", total=num_examples):
            question = example.get("question", "").strip()
            reasoning = example.get("reasoning", "").strip()
            answer = example.get("answer", "").strip()

            if question:
                text = f"Question: {question}"
                if reasoning:
                    text += f"\n\nReasoning: {reasoning}"
                if answer:
                    text += f"\n\nAnswer: {answer}"
                chunk.append(text)

            if len(chunk) >= chunk_size:
                yield chunk
                total_processed += len(chunk)
                chunk = []

        if chunk:
            yield chunk
            total_processed += len(chunk)

        print(f"✓ Processed {total_processed:,} problems from GSM8K Enhanced")

    except Exception as e:
        print(f"✗ Error loading GSM8K Enhanced: {e}")
        yield []


if __name__ == "__main__":
    data = load_gsm8k(num_tokens=1_000_000)
    print(f"\nSample:\n{data[0][:500]}...")
