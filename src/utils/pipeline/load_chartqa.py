"""
ChartQA Dataset Loader
Chart question-answering dataset with 32K+ chart-question-answer triplets.
Includes both human-authored and machine-generated questions.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_chartqa(num_pairs=33_000):
    """
    Load ChartQA (HuggingFaceM4/ChartQA)
    32.7K chart images with question-answer pairs.

    Args:
        num_pairs: Target number of chart QA pairs to load

    Returns:
        List of dicts with 'image', 'question', and 'answer' keys
    """
    print("\n" + "="*60)
    print("Loading ChartQA (HuggingFaceM4/ChartQA)...")
    print(f"Target pairs: {num_pairs:,}")
    print("="*60)

    try:
        ds = load_dataset(
            "HuggingFaceM4/ChartQA",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_pairs)

        formatted = []
        for example in tqdm(ds, desc="Processing ChartQA", total=num_pairs):
            image = example.get("image", None)
            question = example.get("question", "").strip()
            answer = example.get("answer", "").strip()

            if image is not None and question and answer:
                formatted.append({
                    "image": image,
                    "question": question,
                    "answer": answer
                })

        print(f"✓ Loaded {len(formatted):,} chart QA pairs from ChartQA")
        return formatted

    except Exception as e:
        print(f"✗ Error loading ChartQA: {e}")
        return []


def load_chartqa_chunked(num_pairs=33_000, chunk_size=2000):
    """
    Load ChartQA in chunks to avoid RAM overflow.
    Yields batches of chart QA pairs instead of loading everything at once.
    NO SHUFFLE - maintains deterministic order for checkpointing.

    Args:
        num_pairs: Target number of chart QA pairs to load
        chunk_size: Number of pairs per chunk

    Yields:
        List of dicts with 'image', 'question', and 'answer' keys
    """
    print("\n" + "="*60)
    print("Loading ChartQA (HuggingFaceM4/ChartQA) in chunks...")
    print(f"Target pairs: {num_pairs:,}")
    print(f"Chunk size: {chunk_size:,} pairs")
    print("="*60)

    try:
        ds = load_dataset(
            "HuggingFaceM4/ChartQA",
            split="train",
            streaming=True
        )
        ds = ds.take(num_pairs)

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing ChartQA", total=num_pairs):
            image = example.get("image", None)
            question = example.get("question", "").strip()
            answer = example.get("answer", "").strip()

            if image is not None and question and answer:
                chunk.append({
                    "image": image,
                    "question": question,
                    "answer": answer
                })

            if len(chunk) >= chunk_size:
                yield chunk
                total_processed += len(chunk)
                chunk = []

        if chunk:
            yield chunk
            total_processed += len(chunk)

        print(f"✓ Processed {total_processed:,} pairs from ChartQA")

    except Exception as e:
        print(f"✗ Error loading ChartQA: {e}")
        yield []


if __name__ == "__main__":
    data = load_chartqa(num_pairs=100)
    print(f"\nLoaded {len(data)} pairs")
    print(f"Sample question: {data[0]['question']}")
    print(f"Sample answer: {data[0]['answer']}")
