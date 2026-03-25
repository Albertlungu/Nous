"""
Dataset Combiner
Pre-training corpus — general web crawl, code, books, science, Wikipedia.
Saves to training_data/nous_corpus.txt

Mix (100B tokens total):
  FineWeb-Edu   50B  (50%) — educational web text from Common Crawl
  DCLM-Baseline 20B  (20%) — filtered general web text
  The Stack     15B  (15%) — deduplicated source code
  Wikipedia      5B  ( 5%) — encyclopedic content
  peS2o          5B  ( 5%) — scientific papers (Semantic Scholar)
  PG-19          5B  ( 5%) — public domain books
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import random
import gzip
import json
from api.paths import get_training_data_path

from src.utils.pipeline.load_fineweb import load_fineweb, load_fineweb_chunked
from src.utils.pipeline.load_dclm import load_dclm, load_dclm_chunked
from src.utils.pipeline.load_code import load_all_code, load_all_code_chunked
from src.utils.pipeline.load_wikipedia import load_wikipedia, load_wikipedia_chunked
from src.utils.pipeline.load_science import load_science, load_science_chunked
from src.utils.pipeline.load_books import load_books, load_books_chunked


def stream_datasets_to_disk_chunked_with_checkpoint(output_path, target_tokens=100_000_000_000, chunk_size=10000, checkpoint_frequency=1000):
    """
    Stream datasets to disk in chunks with fine-grained checkpointing.
    Saves checkpoint every N examples to enable resume from exact position.

    Args:
        output_path: Path to save the corpus (.txt)
        target_tokens: Target total tokens (default 100B)
        chunk_size: Number of documents per chunk (default 10000)
        checkpoint_frequency: Save checkpoint every N examples (default 1000)
    """
    checkpoint_path = output_path.replace('.txt', '_checkpoint.json')

    print("\n" + "=" * 80)
    print("STREAMING ALL DATASETS TO DISK (CHECKPOINTED MODE)")
    print(f"Target: {target_tokens:,} tokens")
    print(f"Chunk size: {chunk_size:,} documents")
    print(f"Checkpoint frequency: every {checkpoint_frequency:,} examples")
    print(f"Output: {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)

    # (name, target_tokens, avg_tokens, loader_function)
    datasets_config = [
        ("FineWeb-Edu",    50_000_000_000, 650, lambda: load_fineweb_chunked(50_000_000_000, chunk_size=chunk_size)),
        ("DCLM-Baseline",  20_000_000_000, 700, lambda: load_dclm_chunked(20_000_000_000, chunk_size=chunk_size)),
        ("The Stack",      15_000_000_000, 600, lambda: load_all_code_chunked(15_000_000_000, chunk_size=chunk_size)),
        ("Wikipedia",       5_000_000_000, 500, lambda: load_wikipedia_chunked(5_000_000_000, chunk_size=chunk_size)),
        ("peS2o Science",   5_000_000_000, 1200, lambda: load_science_chunked(5_000_000_000, chunk_size=chunk_size)),
        ("PG-19 Books",     5_000_000_000, 150000, lambda: load_books_chunked(5_000_000_000, chunk_size=100)),
    ]

    # Load checkpoint if exists
    checkpoint = {
        "current_dataset_index": 0,
        "current_dataset_example": 0,
        "total_examples": 0
    }

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        print(f"\n✓ Resuming from checkpoint:")
        print(f"  Dataset: {datasets_config[checkpoint['current_dataset_index']][0]}")
        print(f"  Example within dataset: {checkpoint['current_dataset_example']:,}")
        print(f"  Total examples written: {checkpoint['total_examples']:,}")
        mode = "a"  # append mode
    else:
        print("\n✓ Starting fresh (no checkpoint found)")
        mode = "w"  # write mode

    total_examples = checkpoint["total_examples"]
    start_dataset_idx = checkpoint["current_dataset_index"]
    start_example = checkpoint["current_dataset_example"]

    with open(output_path, mode, encoding="utf-8", buffering=1024*1024) as f:  # 1MB buffer
        for dataset_idx, (name, token_target, avg_tokens, loader_generator) in enumerate(datasets_config):
            # Skip datasets we've already completed
            if dataset_idx < start_dataset_idx:
                print(f"\n⏭  Skipping {name} (already completed)")
                continue

            print(f"\n{'=' * 80}")
            print(f"Streaming {name} (target: {token_target:,} tokens)...")
            print(f"{'=' * 80}")

            try:
                dataset_examples = 0
                examples_to_skip = start_example if dataset_idx == start_dataset_idx else 0

                if examples_to_skip > 0:
                    print(f"  Skipping first {examples_to_skip:,} examples (already written)...")

                # Process chunks as they're yielded
                for chunk in loader_generator():
                    for text in chunk:
                        # Skip examples we've already written
                        if dataset_examples < examples_to_skip:
                            dataset_examples += 1
                            continue

                        # Write the example
                        f.write(text)
                        f.write("\n\n")
                        dataset_examples += 1
                        total_examples += 1

                        # Save checkpoint every N examples
                        if total_examples % checkpoint_frequency == 0:
                            checkpoint["current_dataset_index"] = dataset_idx
                            checkpoint["current_dataset_example"] = dataset_examples
                            checkpoint["total_examples"] = total_examples

                            with open(checkpoint_path, 'w') as chk:
                                json.dump(checkpoint, chk, indent=2)

                            # Flush to disk
                            f.flush()
                            os.fsync(f.fileno())

                    # Progress update after each chunk
                    if dataset_examples % (chunk_size * 10) == 0:
                        print(f"  Written {dataset_examples:,} examples from {name}... (total: {total_examples:,})")

                print(f"✓ {name}: {dataset_examples:,} examples written to disk")
                print(f"  Running total: {total_examples:,} examples")

                # Reset start_example for next dataset
                start_example = 0

                # Save checkpoint at end of dataset
                checkpoint["current_dataset_index"] = dataset_idx + 1
                checkpoint["current_dataset_example"] = 0
                checkpoint["total_examples"] = total_examples

                with open(checkpoint_path, 'w') as chk:
                    json.dump(checkpoint, chk, indent=2)

                print(f"  Checkpoint saved ✓")

            except Exception as e:
                print(f"\n✗ {name}: Failed — {e}")
                print(f"  Checkpoint saved at {total_examples:,} examples.")
                print(f"  You can resume by running the script again.")
                # Save checkpoint before exiting
                checkpoint["current_dataset_index"] = dataset_idx
                checkpoint["current_dataset_example"] = dataset_examples
                checkpoint["total_examples"] = total_examples

                with open(checkpoint_path, 'w') as chk:
                    json.dump(checkpoint, chk, indent=2)
                raise  # Re-raise to stop execution

    file_size_gb = os.path.getsize(output_path) / (1024 ** 3)

    print(f"\n{'=' * 80}")
    print("STREAMING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total examples: {total_examples:,}")
    print(f"File size: {file_size_gb:.2f} GB (uncompressed)")
    print(f"Output: {output_path}")
    print(f"{'=' * 80}")
    print("\nNOTE: Data is not shuffled. Datasets maintain deterministic order.")
    print("Memory usage kept minimal by processing in chunks.")
    print("Checkpoints saved every 1,000 examples for fine-grained resume.")
    print(f"\nTo compress: python src/utils/pipeline/compress_corpus.py")
    print(f"{'=' * 80}")

    # Clean up checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\n✓ Checkpoint file removed (pipeline completed successfully)")


def stream_datasets_to_disk(output_path, target_tokens=100_000_000_000):
    """
    DEPRECATED: Use stream_datasets_to_disk_chunked() instead to avoid RAM issues.

    Stream datasets directly to disk without loading everything into RAM.

    Args:
        output_path: Path to save the corpus
        target_tokens: Target total tokens (default 100B)
    """
    print("\n" + "=" * 80)
    print("WARNING: This function still loads each full dataset into RAM!")
    print("Use stream_datasets_to_disk_chunked() instead for true low-memory operation.")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("STREAMING ALL DATASETS TO DISK")
    print(f"Target: {target_tokens:,} tokens")
    print(f"Output: {output_path}")
    print("=" * 80)

    # (name, target_tokens, loader_function)
    datasets_config = [
        ("FineWeb-Edu",    50_000_000_000, lambda: load_fineweb(50_000_000_000)),
        ("DCLM-Baseline",  20_000_000_000, lambda: load_dclm(20_000_000_000)),
        ("The Stack",      15_000_000_000, lambda: load_all_code(15_000_000_000)),
        ("Wikipedia",       5_000_000_000, lambda: load_wikipedia(5_000_000_000)),
        ("peS2o Science",   5_000_000_000, lambda: load_science(5_000_000_000)),
        ("PG-19 Books",     5_000_000_000, lambda: load_books(5_000_000_000)),
    ]

    total_examples = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for name, token_target, loader_func in datasets_config:
            print(f"\n{'=' * 80}")
            print(f"Streaming {name} (target: {token_target:,} tokens)...")
            print(f"{'=' * 80}")

            try:
                texts = loader_func()

                # Write each text immediately, then clear from memory
                for i, text in enumerate(texts):
                    f.write(text)
                    f.write("\n\n")

                    if (i + 1) % 10000 == 0:
                        print(f"  Written {i + 1:,} examples from {name}...")

                dataset_examples = len(texts)
                total_examples += dataset_examples

                print(f"✓ {name}: {dataset_examples:,} examples written to disk")
                print(f"  Running total: {total_examples:,} examples")

                # Clear texts from memory
                del texts

            except Exception as e:
                print(f"✗ {name}: Failed — {e}")
                continue

    file_size_gb = os.path.getsize(output_path) / (1024 ** 3)

    print(f"\n{'=' * 80}")
    print("STREAMING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total examples: {total_examples:,}")
    print(f"File size: {file_size_gb:.2f} GB")
    print(f"Output: {output_path}")
    print(f"{'=' * 80}")
    print("\nNOTE: Data is not shuffled. Each dataset is written sequentially.")
    print("If you need shuffled data, you'll need to shuffle the file afterward.")
    print(f"{'=' * 80}")


def combine_all_datasets(target_tokens=100_000_000_000):
    """
    DEPRECATED: Use stream_datasets_to_disk() instead to avoid RAM issues.

    Load and combine all datasets into a single pre-training corpus.

    Args:
        target_tokens: Target total tokens (default 100B)

    Returns:
        List of all text examples, shuffled
    """
    print("\n" + "=" * 80)
    print("WARNING: This function loads everything into RAM!")
    print("Consider using stream_datasets_to_disk() instead.")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("COMBINING ALL DATASETS FOR NOUS CORPUS")
    print(f"Target: {target_tokens:,} tokens")
    print("=" * 80)

    all_texts = []

    # (name, target_tokens, loader_function)
    datasets_config = [
        ("FineWeb-Edu",    50_000_000_000, lambda: load_fineweb(50_000_000_000)),
        ("DCLM-Baseline",  20_000_000_000, lambda: load_dclm(20_000_000_000)),
        ("The Stack",      15_000_000_000, lambda: load_all_code(15_000_000_000)),
        ("Wikipedia",       5_000_000_000, lambda: load_wikipedia(5_000_000_000)),
        ("peS2o Science",   5_000_000_000, lambda: load_science(5_000_000_000)),
        ("PG-19 Books",     5_000_000_000, lambda: load_books(5_000_000_000)),
    ]

    for name, token_target, loader_func in datasets_config:
        print(f"\n{'=' * 80}")
        print(f"Loading {name} (target: {token_target:,} tokens)...")
        print(f"{'=' * 80}")

        try:
            texts = loader_func()
            all_texts.extend(texts)
            print(f"✓ {name}: {len(texts):,} examples added")
            print(f"  Running total: {len(all_texts):,} examples")

        except Exception as e:
            print(f"✗ {name}: Failed — {e}")
            continue

    print(f"\n{'=' * 80}")
    print("Shuffling all examples...")
    print(f"{'=' * 80}")

    random.seed(42)
    random.shuffle(all_texts)

    print(f"\n{'=' * 80}")
    print("DATASET COMBINATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total examples: {len(all_texts):,}")
    print(f"{'=' * 80}")

    return all_texts


def save_corpus_txt(texts, output_path):
    """
    Save combined corpus to text file.

    Args:
        texts: List of text examples
        output_path: Path to save the .txt file
    """
    print(f"\n{'=' * 80}")
    print(f"Saving corpus to {output_path}...")
    print(f"{'=' * 80}")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            f.write(text)
            f.write("\n\n")

            if (i + 1) % 10000 == 0:
                print(f"  Written {i + 1:,} examples...")

    print(f"\n✓ Saved {len(texts):,} examples to {output_path}")

    file_size_gb = os.path.getsize(output_path) / (1024 ** 3)
    print(f"  File size: {file_size_gb:.2f} GB")


def main():
    print("=" * 80)
    print("NOUS CORPUS BUILDER (CHECKPOINTED STREAMING MODE)")
    print("Pre-training mix: web crawl, code, books, science, Wikipedia")
    print("Target: 100B tokens")
    print("Memory-efficient: processes data in 10K document chunks")
    print("Fine-grained checkpointing: Resume from exact position")
    print("Deterministic order: No shuffling, same order every run")
    print("=" * 80)

    # Hardcoded path to Seagate HDD
    output_path = "/Volumes/Seagate HDD/training_data/nous_corpus.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stream_datasets_to_disk_chunked_with_checkpoint(
        output_path,
        target_tokens=100_000_000_000,
        chunk_size=10000,
        checkpoint_frequency=1000  # Save checkpoint every 1000 examples
    )

    print("\n" + "=" * 80)
    print("CORPUS BUILDING COMPLETE!")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print("\nTo compress the corpus:")
    print("  python src/utils/pipeline/compress_corpus.py")
    print("\nReady for tokenization!")
    print("=" * 80)


if __name__ == "__main__":
    main()
