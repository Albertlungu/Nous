"""
Code Dataset Loaders
Target: 9B tokens total (7B from GitHub Code + 2B from Evol-CodeAlpaca)
Programming capabilities across multiple languages
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from datasets import load_dataset
from tqdm import tqdm


def load_github_code(num_tokens=7_000_000_000, avg_tokens=600):
    """
    Load GitHub Code dataset (codeparrot/github-code - 115M files, 1TB)

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per code file

    Returns:
        List of formatted code strings
    """
    print("\n" + "="*60)
    print("Loading GitHub Code (codeparrot/github-code)...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    num_examples = int(num_tokens / avg_tokens)
    languages = ["Python", "JavaScript", "Java", "TypeScript", "C++", "Go", "Rust"]
    examples_per_lang = num_examples // len(languages)

    formatted = []

    try:
        for lang in languages:
            print(f"\nLoading {lang} code...")
            try:
                ds = load_dataset(
                    "codeparrot/github-code",
                    languages=[lang],
                    split="train",
                    streaming=True
                )
                ds = ds.take(examples_per_lang)

                for example in tqdm(ds, desc=f"Processing {lang}", total=examples_per_lang):
                    code = example['code']
                    # Format as instruction-following
                    text = f"Instruction: Write {lang} code.\nOutput: {code}"
                    formatted.append(text)

            except Exception as e:
                print(f"Warning: Could not load {lang} - {e}")
                continue

        print(f"\nLoaded {len(formatted):,} code examples from GitHub Code")
        return formatted

    except Exception as e:
        print(f"Error loading GitHub Code: {e}")
        print("Falling back to Evol-CodeAlpaca only...")
        return load_evol_code_alpaca(num_tokens)


def load_evol_code_alpaca(num_tokens=2_000_000_000, avg_tokens=700):
    """
    Load Evol-CodeAlpaca (instruction-tuned code)

    Args:
        num_tokens: Target number of tokens to load
        avg_tokens: Average tokens per example

    Returns:
        List of formatted code instruction strings
    """
    print("\n" + "="*60)
    print("Loading Evol-CodeAlpaca...")
    print(f"Target tokens: {num_tokens:,}")
    print("="*60)

    try:
        ds = load_dataset("theblackcat102/evol-codealpaca-v1", split="train", streaming=True)

        formatted = []
        for example in tqdm(ds, desc="Processing Evol-CodeAlpaca"):
            text = f"Instruction: {example['instruction']}\nOutput: {example['output']}"
            formatted.append(text)

        print(f"✓ Loaded {len(formatted):,} code examples from Evol-CodeAlpaca")
        return formatted

    except Exception as e:
        print(f"✗ Error loading Evol-CodeAlpaca: {e}")
        return []


def load_all_code(github_tokens=7_000_000_000, evol_tokens=2_000_000_000):
    """
    Load all code datasets

    Returns:
        Combined list of code examples
    """
    all_code = []

    # Load GitHub Code
    github_data = load_github_code(github_tokens)
    all_code.extend(github_data)

    # Add Evol-CodeAlpaca
    evol_data = load_evol_code_alpaca(evol_tokens)
    all_code.extend(evol_data)

    print(f"\n{'='*60}")
    print(f"Total code examples: {len(all_code):,}")
    print(f"{'='*60}")

    return all_code


if __name__ == "__main__":
    # Test the loader
    data = load_evol_code_alpaca(num_tokens=100_000)  # Test with smaller dataset
    print(f"\nSample code example:\n{data[0][:500]}...")
