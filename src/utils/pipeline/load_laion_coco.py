"""
LAION-COCO Aesthetic Dataset Loader
8.5M image-caption pairs from LAION-COCO filtered for aesthetic quality.
10% subset of LAION-COCO with text and image filtering.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_laion_coco(num_pairs=5_000_000):
    """
    Load LAION-COCO Aesthetic (guangyil/laion-coco-aesthetic)
    8.5M high-quality image-caption pairs filtered from LAION-COCO.

    Args:
        num_pairs: Target number of image-caption pairs to load

    Returns:
        List of dicts with 'image' and 'caption' keys
    """
    print("\n" + "="*60)
    print("Loading LAION-COCO Aesthetic (guangyil/laion-coco-aesthetic)...")
    print(f"Target pairs: {num_pairs:,}")
    print("="*60)

    try:
        ds = load_dataset(
            "guangyil/laion-coco-aesthetic",
            split="train",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_pairs)

        formatted = []
        for example in tqdm(ds, desc="Processing LAION-COCO", total=num_pairs):
            image = example.get("image", None)
            caption = example.get("caption", "").strip()

            if image is not None and caption:
                formatted.append({
                    "image": image,
                    "caption": caption
                })

        print(f"✓ Loaded {len(formatted):,} image-caption pairs from LAION-COCO")
        return formatted

    except Exception as e:
        print(f"✗ Error loading LAION-COCO Aesthetic: {e}")
        return []


def load_laion_coco_chunked(num_pairs=5_000_000, chunk_size=10000):
    """
    Load LAION-COCO Aesthetic in chunks to avoid RAM overflow.
    Yields batches of image-caption pairs instead of loading everything at once.
    NO SHUFFLE - maintains deterministic order for checkpointing.

    Args:
        num_pairs: Target number of image-caption pairs to load
        chunk_size: Number of pairs per chunk

    Yields:
        List of dicts with 'image' and 'caption' keys
    """
    print("\n" + "="*60)
    print("Loading LAION-COCO Aesthetic (guangyil/laion-coco-aesthetic) in chunks...")
    print(f"Target pairs: {num_pairs:,}")
    print(f"Chunk size: {chunk_size:,} pairs")
    print("="*60)

    try:
        ds = load_dataset(
            "guangyil/laion-coco-aesthetic",
            split="train",
            streaming=True
        )
        ds = ds.take(num_pairs)

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing LAION-COCO", total=num_pairs):
            image = example.get("image", None)
            caption = example.get("caption", "").strip()

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

        print(f"✓ Processed {total_processed:,} pairs from LAION-COCO")

    except Exception as e:
        print(f"✗ Error loading LAION-COCO Aesthetic: {e}")
        yield []


if __name__ == "__main__":
    data = load_laion_coco(num_pairs=100)
    print(f"\nLoaded {len(data)} pairs")
    print(f"Sample caption: {data[0]['caption']}")
