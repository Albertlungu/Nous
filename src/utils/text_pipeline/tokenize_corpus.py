"""
Corpus Tokenizer
Tokenizes nous_corpus.txt using TikToken
Saves to training_data/nous_corpus.pkl
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pickle
from tqdm import tqdm
from api.paths import get_training_data_path
from src.tokenizer.tiktoken_tokenizer import TikToken


def tokenize_corpus(input_path, output_path, target_tokens=46_000_000_000):
    """
    Tokenize the text corpus and save as pickle file

    Args:
        input_path: Path to nous_corpus.txt
        output_path: Path to save nous_corpus.pkl
        target_tokens: Maximum tokens to process (default 46B)

    Returns:
        List of tokenized sequences
    """
    print("="*80)
    print("TOKENIZING NOUS CORPUS")
    print("="*80)

    # Initialize tokenizer
    print("\nLoading TikToken tokenizer...")
    tokenizer = TikToken()
    print(f"✓ Loaded tokenizer (vocab size: {tokenizer.vocab_size:,})")

    # Read the corpus
    print(f"\nReading corpus from {input_path}...")

    if not os.path.exists(input_path):
        print(f"✗ Error: File not found at {input_path}")
        print("Please run combine_datasets.py first!")
        return None

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by double newlines to get individual examples
    print("\nSplitting into examples...")
    examples = [ex.strip() for ex in content.split('\n\n') if ex.strip()]
    print(f"✓ Found {len(examples):,} examples")

    # Tokenize all examples
    print(f"\n{'='*80}")
    print("TOKENIZING ALL EXAMPLES...")
    print(f"{'='*80}")

    token_ids = []
    total_tokens = 0
    skipped = 0

    for i, text in enumerate(tqdm(examples, desc="Tokenizing")):
        try:
            # Encode the text
            ids = tokenizer.encode(text)

            # Add EOS token
            ids.append(tokenizer.eos_token_id)

            # Track tokens
            total_tokens += len(ids)

            # Add to list
            token_ids.append(ids)

            # Stop if we hit target
            if total_tokens >= target_tokens:
                print(f"\n✓ Reached target of {target_tokens:,} tokens")
                break

            # Progress update every 10k examples
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1:,} examples | {total_tokens:,} tokens so far")

        except Exception as e:
            skipped += 1
            if skipped <= 10:  # Only print first 10 errors
                print(f"\nWarning: Skipped example {i} due to error: {e}")
            continue

    # Print statistics
    print(f"\n{'='*80}")
    print("TOKENIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total examples: {len(token_ids):,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens/example: {total_tokens / len(token_ids):.1f}")
    print(f"Skipped examples: {skipped:,}")
    print(f"{'='*80}")

    # Calculate token distribution
    token_lengths = [len(ids) for ids in token_ids]
    print(f"\nToken Distribution:")
    print(f"  Min tokens: {min(token_lengths):,}")
    print(f"  Max tokens: {max(token_lengths):,}")
    print(f"  Median tokens: {sorted(token_lengths)[len(token_lengths)//2]:,}")

    # Save tokenized data
    print(f"\n{'='*80}")
    print(f"Saving tokenized data to {output_path}...")
    print(f"{'='*80}")

    with open(output_path, "wb") as f:
        pickle.dump(token_ids, f)

    # Print file size
    file_size_bytes = os.path.getsize(output_path)
    file_size_gb = file_size_bytes / (1024 ** 3)
    print(f"✓ Saved successfully!")
    print(f"  File size: {file_size_gb:.2f} GB")

    return token_ids


def verify_tokenized_data(pkl_path):
    """
    Load and verify the tokenized data

    Args:
        pkl_path: Path to nous_corpus.pkl
    """
    print(f"\n{'='*80}")
    print("VERIFYING TOKENIZED DATA")
    print(f"{'='*80}")

    print(f"Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        token_ids = pickle.load(f)

    print(f"✓ Loaded successfully!")
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {len(token_ids):,}")
    print(f"  Total tokens: {sum(len(ids) for ids in token_ids):,}")
    print(f"  Average tokens/example: {sum(len(ids) for ids in token_ids) / len(token_ids):.1f}")

    # Show first example
    print(f"\nFirst example (first 10 token IDs):")
    print(f"  {token_ids[0][:10]}...")

    # Initialize tokenizer to decode
    tokenizer = TikToken()

    # Decode first example
    decoded = tokenizer.decode(token_ids[0][:100])  # First 100 tokens
    print(f"\nFirst example decoded (first 100 tokens):")
    print(f"  {decoded}...")


def main():
    """
    Main function to tokenize nous_corpus.txt
    """
    print("="*80)
    print("NOUS CORPUS TOKENIZER")
    print("Using TikToken tokenizer")
    print("="*80)

    # Paths
    input_path = get_training_data_path('nous_corpus.txt')
    output_path = get_training_data_path('nous_corpus.pkl')

    # Tokenize
    token_ids = tokenize_corpus(
        input_path=input_path,
        output_path=output_path,
        target_tokens=46_000_000_000
    )

    if token_ids is not None:
        # Verify
        verify_tokenized_data(output_path)

        print("\n" + "="*80)
        print("TOKENIZATION PIPELINE COMPLETE!")
        print("="*80)
        print(f"Text corpus: {input_path}")
        print(f"Tokenized corpus: {output_path}")
        print(f"Ready for training!")
        print("="*80)


if __name__ == "__main__":
    main()
