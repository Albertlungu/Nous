"""
OpenMathReasoning Dataset Loader
306K unique mathematical problems with solutions from NVIDIA (2025).
Generated using DeepSeek-R1 and QwQ-32B, sourced from AoPS forums.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_openmath_reasoning(num_tokens=20_000_000_000, avg_tokens=1200):
    """
    Load OpenMathReasoning (nvidia/OpenMathReasoning)
    306K mathematical problems with detailed solutions.

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per problem

    Returns:
        List of text strings (problem + solution)
    """
    print("\n" + "="*60)
    print("Loading OpenMathReasoning (nvidia/OpenMathReasoning)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "nvidia/OpenMathReasoning",
            split="cot",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_examples)

        formatted = []
        for example in tqdm(ds, desc="Processing OpenMathReasoning", total=num_examples):
            problem = example.get("problem", "").strip()
            solution = example.get("generated_solution", "").strip()

            if problem and solution:
                text = f"Problem: {problem}\n\nSolution: {solution}"
                formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} problems from OpenMathReasoning")
        return formatted

    except Exception as e:
        print(f"✗ Error loading OpenMathReasoning: {e}")
        return []


def load_openmath_reasoning_chunked(num_tokens=20_000_000_000, avg_tokens=1200, chunk_size=5000):
    """
    Load OpenMathReasoning in chunks to avoid RAM overflow.
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
    print("Loading OpenMathReasoning (nvidia/OpenMathReasoning) in chunks...")
    print(f"Target tokens: {num_tokens:,}")
    print(f"Chunk size: {chunk_size:,} problems")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)

    try:
        ds = load_dataset(
            "nvidia/OpenMathReasoning",
            split="cot",
            streaming=True
        )
        ds = ds.take(num_examples)

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing OpenMathReasoning", total=num_examples):
            problem = example.get("problem", "").strip()
            solution = example.get("generated_solution", "").strip()

            if problem and solution:
                text = f"Problem: {problem}\n\nSolution: {solution}"
                chunk.append(text)

            if len(chunk) >= chunk_size:
                yield chunk
                total_processed += len(chunk)
                chunk = []

        if chunk:
            yield chunk
            total_processed += len(chunk)

        print(f"✓ Processed {total_processed:,} problems from OpenMathReasoning")

    except Exception as e:
        print(f"✗ Error loading OpenMathReasoning: {e}")
        yield []


if __name__ == "__main__":
    data = load_openmath_reasoning(num_tokens=1_000_000)
    print(f"\nSample:\n{data[0][:500]}...")
