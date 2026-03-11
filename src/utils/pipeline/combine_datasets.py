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
from api.paths import get_training_data_path

from src.utils.pipeline.load_fineweb import load_fineweb
from src.utils.pipeline.load_dclm import load_dclm
from src.utils.pipeline.load_code import load_all_code
from src.utils.pipeline.load_wikipedia import load_wikipedia
from src.utils.pipeline.load_science import load_science
from src.utils.pipeline.load_books import load_books


def combine_all_datasets(target_tokens=100_000_000_000):
    """
    Load and combine all datasets into a single pre-training corpus.

    Args:
        target_tokens: Target total tokens (default 100B)

    Returns:
        List of all text examples, shuffled
    """
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
    print("NOUS CORPUS BUILDER")
    print("Pre-training mix: web crawl, code, books, science, Wikipedia")
    print("Target: 100B tokens")
    print("=" * 80)

    all_texts = combine_all_datasets(target_tokens=100_000_000_000)

    output_path = get_training_data_path("nous_corpus.txt")
    save_corpus_txt(all_texts, output_path)

    print("\n" + "=" * 80)
    print("CORPUS BUILDING COMPLETE!")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print(f"Total examples: {len(all_texts):,}")
    print("Ready for tokenization!")
    print("=" * 80)


if __name__ == "__main__":
    main()
