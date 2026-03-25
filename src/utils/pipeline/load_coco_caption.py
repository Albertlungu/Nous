"""
COCO Caption Dataset Loader
MS COCO image captions without legacy scripts.
330K images with 5 captions per image (1.5M total captions).
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_coco_caption(num_pairs=500_000):
    """
    Load COCO-Caption (lmms-lab/COCO-Caption)
    MS COCO captions formatted for training.

    Args:
        num_pairs: Target number of image-caption pairs to load

    Returns:
        List of dicts with 'image' and 'caption' keys
    """
    print("\n" + "="*60)
    print("Loading COCO-Caption (lmms-lab/COCO-Caption)...")
    print(f"Target pairs: {num_pairs:,}")
    print("="*60)

    try:
        ds = load_dataset(
            "lmms-lab/COCO-Caption",
            split="val",
            streaming=True
        )
        ds = ds.shuffle(seed=42).take(num_pairs)

        formatted = []
        for example in tqdm(ds, desc="Processing COCO-Caption", total=num_pairs):
            image = example.get("image", None)
            # COCO-Caption has "answer" field with list of captions
            captions = example.get("answer", [])

            if image is not None and captions and len(captions) > 0:
                # Use the first caption from the list
                caption = captions[0].strip() if isinstance(captions, list) else str(captions).strip()

                if caption:
                    formatted.append({
                        "image": image,
                        "caption": caption
                    })

        print(f"✓ Loaded {len(formatted):,} image-caption pairs from COCO-Caption")
        return formatted

    except Exception as e:
        print(f"✗ Error loading COCO-Caption: {e}")
        return []


def load_coco_caption_chunked(num_pairs=500_000, chunk_size=10000):
    """
    Load COCO-Caption in chunks to avoid RAM overflow.
    Yields batches of image-caption pairs instead of loading everything at once.
    NO SHUFFLE - maintains deterministic order for checkpointing.

    Args:
        num_pairs: Target number of image-caption pairs to load
        chunk_size: Number of pairs per chunk

    Yields:
        List of dicts with 'image' and 'caption' keys
    """
    print("\n" + "="*60)
    print("Loading COCO-Caption (lmms-lab/COCO-Caption) in chunks...")
    print(f"Target pairs: {num_pairs:,}")
    print(f"Chunk size: {chunk_size:,} pairs")
    print("="*60)

    try:
        ds = load_dataset(
            "lmms-lab/COCO-Caption",
            split="val",
            streaming=True
        )
        ds = ds.take(num_pairs)

        chunk = []
        total_processed = 0

        for example in tqdm(ds, desc="Processing COCO-Caption", total=num_pairs):
            image = example.get("image", None)
            # COCO-Caption has "answer" field with list of captions
            captions = example.get("answer", [])

            if image is not None and captions and len(captions) > 0:
                # Use the first caption from the list
                caption = captions[0].strip() if isinstance(captions, list) else str(captions).strip()

                if caption:
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

        print(f"✓ Processed {total_processed:,} pairs from COCO-Caption")

    except Exception as e:
        print(f"✗ Error loading COCO-Caption: {e}")
        yield []


if __name__ == "__main__":
    data = load_coco_caption(num_pairs=100)
    print(f"\nLoaded {len(data)} pairs")
    print(f"Sample caption: {data[0]['caption']}")
