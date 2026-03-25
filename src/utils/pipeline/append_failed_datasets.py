"""
Append Failed Datasets to Corpus
Appends DCLM, OpenMathReasoning, and OpenR1-Math to existing corpus.txt
Run this after the main pipeline to add the datasets that failed initially.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.utils.pipeline.load_dclm import load_dclm_chunked
from src.utils.pipeline.load_openmath_reasoning import load_openmath_reasoning_chunked
from src.utils.pipeline.load_openr1_math import load_openr1_math_chunked
from tqdm import tqdm


def append_failed_datasets(output_path):
    """
    Append the three failed datasets to existing corpus.

    Args:
        output_path: Path to corpus.txt file
    """
    print("=" * 80)
    print("APPENDING FAILED DATASETS TO CORPUS")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print("\nDatasets to append:")
    print("  1. DCLM-Baseline      30B tokens")
    print("  2. OpenMathReasoning  20B tokens")
    print("  3. OpenR1-Math        20B tokens")
    print("=" * 80)

    # Configuration for the three failed datasets
    datasets_config = [
        ("DCLM-Baseline",      30_000_000_000,  700, lambda: load_dclm_chunked(30_000_000_000, chunk_size=100000)),
        ("OpenMathReasoning",  20_000_000_000, 1200, lambda: load_openmath_reasoning_chunked(20_000_000_000, chunk_size=100000)),
        ("OpenR1-Math",        20_000_000_000, 1500, lambda: load_openr1_math_chunked(20_000_000_000, chunk_size=100000)),
    ]

    total_examples = 0

    # Append mode - will add to existing file
    with open(output_path, 'a', encoding='utf-8', buffering=1024*1024) as f:
        for name, token_target, avg_tokens, loader_generator in datasets_config:
            print(f"\n{'=' * 80}")
            print(f"Appending {name} (target: {token_target:,} tokens)...")
            print(f"{'=' * 80}")

            try:
                dataset_examples = 0

                # Process chunks as they're yielded
                for chunk in loader_generator():
                    for text in chunk:
                        f.write(text)
                        f.write("\n\n")
                        dataset_examples += 1
                        total_examples += 1

                    # Progress update after each chunk
                    if dataset_examples % 1000000 == 0:
                        file_size_gb = os.path.getsize(output_path) / (1024 ** 3)
                        print(f"  Written {dataset_examples:,} examples from {name}... ({file_size_gb:.2f} GB total)")

                # Flush to disk after each dataset
                f.flush()
                os.fsync(f.fileno())

                file_size_gb = os.path.getsize(output_path) / (1024 ** 3)
                print(f"✓ {name}: {dataset_examples:,} examples appended")
                print(f"  File size: {file_size_gb:.2f} GB")

            except Exception as e:
                print(f"\n✗ {name}: Failed — {e}")
                import traceback
                traceback.print_exc()
                print(f"  Continuing to next dataset...")

    file_size_gb = os.path.getsize(output_path) / (1024 ** 3)

    print(f"\n{'=' * 80}")
    print("APPEND COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total new examples appended: {total_examples:,}")
    print(f"Final file size: {file_size_gb:.2f} GB")
    print(f"Output: {output_path}")
    print(f"{'=' * 80}")


def main():
    print("=" * 80)
    print("APPEND FAILED DATASETS TO TEXT CORPUS")
    print("=" * 80)
    print("This script appends the 3 datasets that failed in the initial run:")
    print("  - DCLM-Baseline (30B tokens)")
    print("  - OpenMathReasoning (20B tokens)")
    print("  - OpenR1-Math (20B tokens)")
    print("=" * 80)

    # Path to Extreme SSD
    output_path = "/Volumes/Extreme SSD/Nous/training_data/text/corpus.txt"

    if not os.path.exists(output_path):
        print(f"\n✗ ERROR: Corpus file not found at {output_path}")
        print("Make sure the SSD is mounted and the path is correct.")
        return

    # Show current file size
    current_size_gb = os.path.getsize(output_path) / (1024 ** 3)
    print(f"\nCurrent corpus size: {current_size_gb:.2f} GB")

    append_failed_datasets(output_path)

    print("\n" + "=" * 80)
    print("APPEND COMPLETE!")
    print("=" * 80)
    print(f"Corpus file: {output_path}")
    print("\nNext step: Tokenize the corpus")
    print("  python src/utils/pipeline/tokenize_corpus.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
