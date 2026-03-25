"""
Vision Dataset Combiner for Multimodal Base Model
Pre-training vision corpus - scientific figures, charts, captions.
Saves to /Volumes/Extreme SSD/Nous/training_data/vision/ as pickle chunks.

Mix (7.3M image-caption pairs total):
  LAION-COCO Aesthetic  5.0M  (68%) - high-quality filtered images
  DataComp-1B           1.0M  (14%) - diverse web images
  SciCap MLBCAP         0.4M  ( 5%) - scientific figures (enhanced)
  SciCap                0.4M  ( 5%) - scientific figures (original)
  COCO-Caption          0.5M  ( 7%) - object-focused captions
  ChartQA              33.0K  (<1%) - chart understanding
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import json
import pickle
from src.utils.pipeline.load_laion_coco import load_laion_coco_chunked
from src.utils.pipeline.load_datacomp import load_datacomp_chunked
from src.utils.pipeline.load_scicap import load_scicap_mlbcap_chunked, load_scicap_chunked
from src.utils.pipeline.load_coco_caption import load_coco_caption_chunked
from src.utils.pipeline.load_chartqa import load_chartqa_chunked


def stream_vision_datasets_to_disk(output_dir, target_pairs=7_300_000, chunk_size=100000, checkpoint_frequency=10000):
    """
    Stream vision datasets to disk in 100k-pair chunks with checkpointing.
    Saves as pickle files for each chunk (image-caption pairs).

    Args:
        output_dir: Directory to save vision chunks
        target_pairs: Target total image-caption pairs (default 7.3M)
        chunk_size: Number of pairs per chunk file (default 100k)
        checkpoint_frequency: Save checkpoint every N examples (default 10k)
    """
    checkpoint_path = os.path.join(output_dir, "vision_checkpoint.json")

    print("\n" + "=" * 80)
    print("STREAMING VISION DATASETS TO DISK (CHECKPOINTED MODE)")
    print(f"Target: {target_pairs:,} image-caption pairs")
    print(f"Chunk size: {chunk_size:,} pairs per pickle file")
    print(f"Checkpoint frequency: every {checkpoint_frequency:,} examples")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)

    # (name, target_pairs, loader_function)
    datasets_config = [
        ("LAION-COCO",    5_000_000, lambda: load_laion_coco_chunked(5_000_000, chunk_size=chunk_size)),
        ("DataComp-1B",   1_000_000, lambda: load_datacomp_chunked(1_000_000, chunk_size=chunk_size)),
        ("SciCap MLBCAP",   400_000, lambda: load_scicap_mlbcap_chunked(400_000, chunk_size=chunk_size)),
        ("SciCap",          400_000, lambda: load_scicap_chunked(400_000, chunk_size=chunk_size)),
        ("COCO-Caption",    500_000, lambda: load_coco_caption_chunked(500_000, chunk_size=chunk_size)),
        ("ChartQA",          33_000, lambda: load_chartqa_chunked(33_000, chunk_size=chunk_size)),
    ]

    # Load checkpoint if exists
    checkpoint = {
        "current_dataset_index": 0,
        "current_dataset_example": 0,
        "total_examples": 0,
        "current_chunk_file": 0
    }

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        print(f"\n✓ Resuming from checkpoint:")
        print(f"  Dataset: {datasets_config[checkpoint['current_dataset_index']][0]}")
        print(f"  Example within dataset: {checkpoint['current_dataset_example']:,}")
        print(f"  Total examples written: {checkpoint['total_examples']:,}")
        print(f"  Current chunk file: {checkpoint['current_chunk_file']}")
    else:
        print("\n✓ Starting fresh (no checkpoint found)")

    total_examples = checkpoint["total_examples"]
    start_dataset_idx = checkpoint["current_dataset_index"]
    start_example = checkpoint["current_dataset_example"]
    chunk_file_idx = checkpoint["current_chunk_file"]

    current_chunk = []

    for dataset_idx, (name, pair_target, loader_generator) in enumerate(datasets_config):
        # Skip datasets we've already completed
        if dataset_idx < start_dataset_idx:
            print(f"\n⏭  Skipping {name} (already completed)")
            continue

        print(f"\n{'=' * 80}")
        print(f"Streaming {name} (target: {pair_target:,} pairs)...")
        print(f"{'=' * 80}")

        try:
            dataset_examples = 0
            examples_to_skip = start_example if dataset_idx == start_dataset_idx else 0

            if examples_to_skip > 0:
                print(f"  Skipping first {examples_to_skip:,} examples (already written)...")

            # Process chunks as they're yielded
            for chunk in loader_generator():
                for item in chunk:
                    # Skip examples we've already written
                    if dataset_examples < examples_to_skip:
                        dataset_examples += 1
                        continue

                    # Add to current chunk
                    current_chunk.append(item)
                    dataset_examples += 1
                    total_examples += 1

                    # When chunk reaches size limit, save it
                    if len(current_chunk) >= chunk_size:
                        chunk_filename = os.path.join(output_dir, f"vision_chunk_{chunk_file_idx:06d}.pkl")
                        with open(chunk_filename, 'wb') as cf:
                            pickle.dump(current_chunk, cf)

                        print(f"  Saved {len(current_chunk):,} pairs to vision_chunk_{chunk_file_idx:06d}.pkl")

                        # Clear chunk from RAM
                        current_chunk = []
                        chunk_file_idx += 1

                    # Save checkpoint every N examples
                    if total_examples % checkpoint_frequency == 0:
                        checkpoint["current_dataset_index"] = dataset_idx
                        checkpoint["current_dataset_example"] = dataset_examples
                        checkpoint["total_examples"] = total_examples
                        checkpoint["current_chunk_file"] = chunk_file_idx

                        with open(checkpoint_path, 'w') as chk:
                            json.dump(checkpoint, chk, indent=2)

                # Progress update after each loader chunk
                if dataset_examples % (chunk_size * 10) == 0:
                    print(f"  Processed {dataset_examples:,} examples from {name}... (total: {total_examples:,})")

            print(f"✓ {name}: {dataset_examples:,} examples processed")
            print(f"  Running total: {total_examples:,} examples")

            # Reset start_example for next dataset
            start_example = 0

            # Save checkpoint at end of dataset
            checkpoint["current_dataset_index"] = dataset_idx + 1
            checkpoint["current_dataset_example"] = 0
            checkpoint["total_examples"] = total_examples
            checkpoint["current_chunk_file"] = chunk_file_idx

            with open(checkpoint_path, 'w') as chk:
                json.dump(checkpoint, chk, indent=2)

            print(f"  Checkpoint saved ✓")

        except Exception as e:
            print(f"\n✗ {name}: Failed — {e}")
            print(f"  Checkpoint saved at {total_examples:,} examples.")
            print(f"  You can resume by running the script again.")

            # Save any remaining items in current chunk before crashing
            if current_chunk:
                chunk_filename = os.path.join(output_dir, f"vision_chunk_{chunk_file_idx:06d}.pkl")
                with open(chunk_filename, 'wb') as cf:
                    pickle.dump(current_chunk, cf)
                print(f"  Saved partial chunk with {len(current_chunk):,} pairs")

            # Save checkpoint before exiting
            checkpoint["current_dataset_index"] = dataset_idx
            checkpoint["current_dataset_example"] = dataset_examples
            checkpoint["total_examples"] = total_examples
            checkpoint["current_chunk_file"] = chunk_file_idx

            with open(checkpoint_path, 'w') as chk:
                json.dump(checkpoint, chk, indent=2)
            raise

    # Save any remaining items in final chunk
    if current_chunk:
        chunk_filename = os.path.join(output_dir, f"vision_chunk_{chunk_file_idx:06d}.pkl")
        with open(chunk_filename, 'wb') as cf:
            pickle.dump(current_chunk, cf)
        print(f"\n✓ Saved final chunk with {len(current_chunk):,} pairs to vision_chunk_{chunk_file_idx:06d}.pkl")
        chunk_file_idx += 1

    # Calculate total size
    total_size_gb = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if f.endswith('.pkl')
    ) / (1024 ** 3)

    print(f"\n{'=' * 80}")
    print("VISION STREAMING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total examples: {total_examples:,}")
    print(f"Total chunk files: {chunk_file_idx}")
    print(f"Total size: {total_size_gb:.2f} GB")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 80}")
    print("\nNOTE: Data is not shuffled. Datasets maintain deterministic order.")
    print("Memory usage kept minimal by processing in 100k-pair chunks.")
    print(f"Checkpoints saved every {checkpoint_frequency:,} examples for fine-grained resume.")
    print("\nEach pickle file contains List[Dict] with keys:")
    print("  - 'image': PIL.Image")
    print("  - 'caption': str")
    print("  - 'question': str (ChartQA only)")
    print("  - 'answer': str (ChartQA only)")
    print(f"{'=' * 80}")

    # Clean up checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\n✓ Checkpoint file removed (pipeline completed successfully)")


def main():
    print("=" * 80)
    print("VISION CORPUS BUILDER FOR MULTIMODAL BASE MODEL")
    print("Pre-training mix: scientific figures, charts, captions")
    print("Target: 7.3M image-caption pairs")
    print("Memory-efficient: processes data in 100k pair chunks")
    print("Fine-grained checkpointing: Resume from exact position")
    print("Deterministic order: No shuffling, same order every run")
    print("=" * 80)

    # Path to Extreme SSD
    output_dir = "/Volumes/Extreme SSD/Nous/training_data/vision"
    os.makedirs(output_dir, exist_ok=True)

    stream_vision_datasets_to_disk(
        output_dir,
        target_pairs=7_300_000,
        chunk_size=100000,  # 100k pairs per pickle file
        checkpoint_frequency=10000  # Save checkpoint every 10k examples
    )

    print("\n" + "=" * 80)
    print("VISION CORPUS BUILDING COMPLETE!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("\nNext step: Preprocess images for ViT encoder")
    print("  - Resize to 224x224")
    print("  - Normalize pixel values")
    print("  - Prepare for vision pretraining")
    print("\nReady for vision encoder pretraining!")
    print("=" * 80)


if __name__ == "__main__":
    main()
