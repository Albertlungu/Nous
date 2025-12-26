"""
Dataset Combiner
Loads all datasets and combines them into a single corpus
Saves to training_data/nous_corpus.txt
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import random
from api.paths import get_training_data_path

# Import all dataset loaders
from src.utils.pipeline.load_openorca import load_openorca
from src.utils.pipeline.load_flan import load_flan
from src.utils.pipeline.load_ultrachat import load_ultrachat
from src.utils.pipeline.load_sharegpt import load_sharegpt
from src.utils.pipeline.load_code import load_all_code
from src.utils.pipeline.load_math import load_all_math


def combine_all_datasets(target_tokens=46_000_000_000):
    """
    Load and combine all datasets into a single corpus

    Args:
        target_tokens: Target total tokens (default 46B)

    Returns:
        List of all text examples, shuffled
    """
    print("\n" + "="*80)
    print("COMBINING ALL DATASETS FOR NOUS CORPUS")
    print(f"Target: {target_tokens:,} tokens")
    print("="*80)

    all_texts = []

    # Dataset configuration: (name, target_tokens, loader_function)
    datasets_config = [
        ("OpenOrca", 10_000_000_000, lambda: load_openorca(10_000_000_000)),
        ("FLAN v2", 8_000_000_000, lambda: load_flan(8_000_000_000)),
        ("UltraChat", 10_000_000_000, lambda: load_ultrachat(10_000_000_000)),
        ("ShareGPT", 4_000_000_000, lambda: load_sharegpt(4_000_000_000)),
        ("Code (Stack + Evol)", 9_000_000_000, lambda: load_all_code(7_000_000_000, 2_000_000_000)),
        ("Math (Orca + Meta)", 5_000_000_000, lambda: load_all_math(3_000_000_000, 2_000_000_000)),
    ]

    # Load each dataset
    for name, target_tokens, loader_func in datasets_config:
        print(f"\n{'='*80}")
        print(f"Loading {name} (target: {target_tokens:,} tokens)...")
        print(f"{'='*80}")

        try:
            texts = loader_func()
            all_texts.extend(texts)
            print(f"✓ {name}: {len(texts):,} examples added")
            print(f"  Running total: {len(all_texts):,} examples")

        except Exception as e:
            print(f"✗ {name}: Failed - {e}")
            continue

    # Shuffle all texts
    print(f"\n{'='*80}")
    print("Shuffling all examples...")
    print(f"{'='*80}")

    random.seed(42)
    random.shuffle(all_texts)

    print(f"\n{'='*80}")
    print("DATASET COMBINATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total examples: {len(all_texts):,}")
    print(f"{'='*80}")

    return all_texts


def save_corpus_txt(texts, output_path):
    """
    Save combined corpus to text file

    Args:
        texts: List of text examples
        output_path: Path to save the .txt file
    """
    print(f"\n{'='*80}")
    print(f"Saving corpus to {output_path}...")
    print(f"{'='*80}")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            # Separate examples with double newline
            f.write(text)
            f.write("\n\n")

            if (i + 1) % 10000 == 0:
                print(f"  Written {i + 1:,} examples...")

    print(f"\n✓ Saved {len(texts):,} examples to {output_path}")

    # Print file size
    file_size_bytes = os.path.getsize(output_path)
    file_size_gb = file_size_bytes / (1024 ** 3)
    print(f"  File size: {file_size_gb:.2f} GB")


def main():
    """
    Main function to combine all datasets and save to nous_corpus.txt
    """
    print("="*80)
    print("NOUS CORPUS BUILDER")
    print("Combining 8 datasets for 46B tokens")
    print("="*80)

    # Combine all datasets
    all_texts = combine_all_datasets(target_tokens=46_000_000_000)

    # Save to training_data/nous_corpus.txt
    output_path = get_training_data_path('nous_corpus.txt')
    save_corpus_txt(all_texts, output_path)

    print("\n" + "="*80)
    print("CORPUS BUILDING COMPLETE!")
    print("="*80)
    print(f"Output file: {output_path}")
    print(f"Total examples: {len(all_texts):,}")
    print(f"Ready for tokenization!")
    print("="*80)


if __name__ == "__main__":
    main()
