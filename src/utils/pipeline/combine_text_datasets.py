"""
Text Dataset Combiner for Multimodal Base Model
Pre-training text corpus - math, reasoning, science, code, books, web.
Saves to /Volumes/Extreme SSD/Nous/training_data/text/corpus.txt

Mix (260B tokens total):
  FineWeb-Edu        80B  (31%) - educational web text
  OpenWebMath        30B  (12%) - mathematical web text
  DCLM-Baseline      30B  (12%) - filtered general web
  OpenMathReasoning  20B  ( 8%) - NVIDIA math with reasoning
  OpenR1-Math        20B  ( 8%) - DeepSeek R1 math reasoning
  ML-ArXiv           20B  ( 8%) - machine learning papers
  Wikipedia          15B  ( 6%) - encyclopedic content
  PG-19              15B  ( 6%) - public domain books
  GSM8K Enhanced     15B  ( 6%) - grade school math
  The Stack          10B  ( 4%) - deduplicated source code
  Code Contests       5B  ( 2%) - competitive programming
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import json
from src.utils.pipeline.load_fineweb import load_fineweb_chunked
from src.utils.pipeline.load_openwebmath import load_openwebmath_chunked
from src.utils.pipeline.load_dclm import load_dclm_chunked
from src.utils.pipeline.load_openmath_reasoning import load_openmath_reasoning_chunked
from src.utils.pipeline.load_openr1_math import load_openr1_math_chunked
from src.utils.pipeline.load_ml_arxiv import load_ml_arxiv_chunked
from src.utils.pipeline.load_wikipedia import load_wikipedia_chunked
from src.utils.pipeline.load_pg19 import load_pg19_chunked
from src.utils.pipeline.load_gsm8k import load_gsm8k_enhanced_chunked
from src.utils.pipeline.load_code import load_all_code_chunked


def stream_text_datasets_to_disk(output_path, target_tokens=260_000_000_000, chunk_size=100000, checkpoint_frequency=10000):
    """
    Stream text datasets to disk in 100k-example chunks with checkpointing.
    Loads chunks into RAM, writes to disk, then clears RAM.

    Args:
        output_path: Path to save the corpus (.txt)
        target_tokens: Target total tokens (default 260B)
        chunk_size: Number of documents per chunk (default 100k)
        checkpoint_frequency: Save checkpoint every N examples (default 10k)
    """
    checkpoint_path = output_path.replace('.txt', '_checkpoint.json')

    print("\n" + "=" * 80)
    print("STREAMING TEXT DATASETS TO DISK (CHECKPOINTED MODE)")
    print(f"Target: {target_tokens:,} tokens (260B for base model)")
    print(f"Chunk size: {chunk_size:,} documents")
    print(f"Checkpoint frequency: every {checkpoint_frequency:,} examples")
    print(f"Output: {output_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)

    # (name, target_tokens, avg_tokens, loader_function)
    datasets_config = [
        ("FineWeb-Edu",         80_000_000_000,  650, lambda: load_fineweb_chunked(80_000_000_000, chunk_size=chunk_size)),
        ("OpenWebMath",         30_000_000_000,  800, lambda: load_openwebmath_chunked(30_000_000_000, chunk_size=chunk_size)),
        ("DCLM-Baseline",       30_000_000_000,  700, lambda: load_dclm_chunked(30_000_000_000, chunk_size=chunk_size)),
        ("OpenMathReasoning",   20_000_000_000, 1200, lambda: load_openmath_reasoning_chunked(20_000_000_000, chunk_size=chunk_size)),
        ("OpenR1-Math",         20_000_000_000, 1500, lambda: load_openr1_math_chunked(20_000_000_000, chunk_size=chunk_size)),
        ("ML-ArXiv",            20_000_000_000, 1500, lambda: load_ml_arxiv_chunked(20_000_000_000, chunk_size=chunk_size)),
        ("Wikipedia",           15_000_000_000,  500, lambda: load_wikipedia_chunked(15_000_000_000, chunk_size=chunk_size)),
        ("PG-19 Books",         15_000_000_000, 2000, lambda: load_pg19_chunked(15_000_000_000, chunk_size=chunk_size)),
        ("GSM8K Enhanced",      15_000_000_000,  500, lambda: load_gsm8k_enhanced_chunked(15_000_000_000, chunk_size=chunk_size)),
        ("The Stack",           10_000_000_000,  600, lambda: load_all_code_chunked(10_000_000_000, chunk_size=chunk_size)),
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
        mode = "a"
    else:
        print("\n✓ Starting fresh (no checkpoint found)")
        mode = "w"

    total_examples = checkpoint["total_examples"]
    start_dataset_idx = checkpoint["current_dataset_index"]
    start_example = checkpoint["current_dataset_example"]

    with open(output_path, mode, encoding="utf-8", buffering=1024*1024) as f:
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
                raise

    file_size_gb = os.path.getsize(output_path) / (1024 ** 3)

    print(f"\n{'=' * 80}")
    print("TEXT STREAMING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total examples: {total_examples:,}")
    print(f"File size: {file_size_gb:.2f} GB (uncompressed)")
    print(f"Output: {output_path}")
    print(f"{'=' * 80}")
    print("\nNOTE: Data is not shuffled. Datasets maintain deterministic order.")
    print("Memory usage kept minimal by processing in 100k-example chunks.")
    print(f"Checkpoints saved every {checkpoint_frequency:,} examples for fine-grained resume.")
    print(f"{'=' * 80}")

    # Clean up checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("\n✓ Checkpoint file removed (pipeline completed successfully)")


def main():
    print("=" * 80)
    print("TEXT CORPUS BUILDER FOR MULTIMODAL BASE MODEL")
    print("Pre-training mix: math, reasoning, science, code, books, web")
    print("Target: 260B tokens")
    print("Memory-efficient: processes data in 100k document chunks")
    print("Fine-grained checkpointing: Resume from exact position")
    print("Deterministic order: No shuffling, same order every run")
    print("=" * 80)

    # Path to Extreme SSD
    output_path = "/Volumes/Extreme SSD/Nous/training_data/text/corpus.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stream_text_datasets_to_disk(
        output_path,
        target_tokens=260_000_000_000,
        chunk_size=100000,  # 100k examples per chunk
        checkpoint_frequency=10000  # Save checkpoint every 10k examples
    )

    print("\n" + "=" * 80)
    print("TEXT CORPUS BUILDING COMPLETE!")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print("\nNext step: Tokenize the corpus")
    print("  python src/utils/pipeline/tokenize_corpus.py")
    print("\nReady for text pretraining!")
    print("=" * 80)


if __name__ == "__main__":
    main()
