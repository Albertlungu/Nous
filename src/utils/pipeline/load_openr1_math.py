"""
OpenR1-Math Dataset Loader
220K mathematical problems with reasoning traces from DeepSeek R1 (2025).
Each problem verified with Math Verify for correctness.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_openr1_math(num_tokens=20_000_000_000, avg_tokens=1500):
    """
    Load OpenR1-Math (open-r1/OpenR1-Math-220k)
    220K math problems with 2-4 reasoning traces each.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per problem

    Returns:
        List of text strings (problem + reasoning traces)
    """
    print("\n" + "="*60)
    print("Loading OpenR1-Math (open-r1/OpenR1-Math-220k)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "open-r1/OpenR1-Math-220k",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing OpenR1-Math", total=num_examples):
            problem = example.get("problem", "").strip()
            solution = example.get("solution", "").strip()
            answer = example.get("answer", "").strip()

            if problem and solution:
                text = f"Problem: {problem}\n\nSolution: {solution}"
                if answer:
                    text += f"\n\nAnswer: {answer}"
                formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} problems from OpenR1-Math")
        return formatted

    except Exception as e:
        print(f"✗ Error loading OpenR1-Math: {e}")
        return []


def load_openr1_math_chunked(num_tokens=20_000_000_000, avg_tokens=1500, chunk_size=5000):
    """
    Load OpenR1-Math in chunks to avoid RAM overflow.
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
    print("Loading OpenR1-Math (open-r1/OpenR1-Math-220k) in chunks...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Chunk size: {chunk_size:,} problems")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "open-r1/OpenR1-Math-220k",
            split="train",
            streaming=True
        )
        ds = ds.take(num_examples)

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing OpenR1-Math", total=num_examples):
            problem = example.get("problem", "").strip()
            solution = example.get("solution", "").strip()
            answer = example.get("answer", "").strip()

            if problem and solution:
                text = f"Problem: {problem}\n\nSolution: {solution}"
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

        print(f"✓ Processed {total_processed:,} problems from OpenR1-Math")

    except Exception as e:
        print(f"✗ Error loading OpenR1-Math: {e}")
        yield []


if __name__ == "__main__":
    data = load_openr1_math(num_tokens=1_000_000)
    print(f"\nSample:\n{data[0][:500]}...")
