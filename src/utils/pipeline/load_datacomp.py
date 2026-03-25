"""
DataComp Dataset Loader
1B image-text pairs from DataComp benchmark.
High-quality image-caption pairs for vision-language pretraining.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_datacomp(num_pairs=1_000_000):
    """
    Load DataComp-1B (mlfoundations/datacomp_1b)
    Subset of 1B image-text pairs from DataComp.

    Args:
        num_pairs: Target number of image-caption pairs to load

    Returns:
        List of dicts with 'image' and 'caption' keys
    """
    print("\n" + "="*60)
    print("Loading DataComp-1B (mlfoundations/datacomp_1b)...")
    print(f"Target pairs: {num_pairs:,}")
    print("="*60)

    try:
        ds = load_dataset(
            "mlfoundations/datacomp_1b",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_pairs)

        formatted = []
        for example in tqdm(ds, desc="Processing DataComp", total=num_pairs):
            image = example.get("image", None)
            caption = example.get("text", "").strip()

            if image is not None and caption:
                formatted.append({
                    "image": image,
                    "caption": caption
                })

        print(f"✓ Loaded {len(formatted):,} image-caption pairs from DataComp")
        return formatted

    except Exception as e:
        print(f"✗ Error loading DataComp: {e}")
        return []


def load_datacomp_chunked(num_pairs=1_000_000, chunk_size=10000):
    """
    Load DataComp-1B in chunks to avoid RAM overflow.
    Yields batches of image-caption pairs instead of loading everything at once.
    NO SHUFFLE - maintains deterministic order for checkpointing.

    Args:
        num_pairs: Target number of image-caption pairs to load
        chunk_size: Number of pairs per chunk

    Yields:
        List of dicts with 'image' and 'caption' keys
    """
    print("\n" + "="*60)
    print("Loading DataComp-1B (mlfoundations/datacomp_1b) in chunks...")
    print(f"Target pairs: {num_pairs:,}")
    print(f"Chunk size: {chunk_size:,} pairs")
    print("="*60)

    try:
        ds = load_dataset(
            "mlfoundations/datacomp_1b",
            split="train",
            streaming=True
        )
        ds = ds.take(num_pairs)

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing DataComp", total=num_pairs):
            image = example.get("image", None)
            caption = example.get("text", "").strip()

            if image is not None and caption:
                chunk.append({
                    "image": image,
                    "caption": caption
                })

            if len(chunk) >= chunk_size:
                yield chunk
                total_processed += len(chunk)
                chunk = []

        if chunk:
            yield chunk
            total_processed += len(chunk)

        print(f"✓ Processed {total_processed:,} pairs from DataComp")

    except Exception as e:
        print(f"✗ Error loading DataComp: {e}")
        yield []


if __name__ == "__main__":
    data = load_datacomp(num_pairs=100)
    print(f"\nLoaded {len(data)} pairs")
    print(f"Sample caption: {data[0]['caption']}")
