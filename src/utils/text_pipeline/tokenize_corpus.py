"""
Corpus Tokenizer
Tokenizes nous_corpus.txt using TikToken
Saves to training_data/nous_corpus.pkl
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pickle
from api.paths import get_training_data_path
from src.tokenizer.tiktoken_tokenizer import TikToken


def tokenize_corpus(input_path, output_path, target_tokens=46_000_000_000):
    """
    Tokenize the text corpus and save as pickle file (STREAMING VERSION)

    Processes the corpus in a streaming fashion:
    - Reads line by line (not all at once)
    - Tokenizes one example at a time
    - Writes each tokenized example to disk immediately
    - Never keeps the full token list in memory

    Args:
        input_path: Path to nous_corpus.txt
        output_path: Path to save nous_corpus.pkl
        target_tokens: Maximum tokens to process (default 46B)

    Returns:
        Dictionary with statistics (not the actual tokens)
    """
    print("="*80)
    print("TOKENIZING NOUS CORPUS (STREAMING MODE)")
    print("="*80)

    # Initialize tokenizer
    print("\nLoading TikToken tokenizer...")
    tokenizer = TikToken()
    print(f"✓ Loaded tokenizer (vocab size: {tokenizer.vocab_size:,})")

    # Check input file exists
    print(f"\nPreparing to stream from {input_path}...")
    if not os.path.exists(input_path):
        print(f"✗ Error: File not found at {input_path}")
        print("Please run combine_datasets.py first!")
        return None

    # Streaming tokenization
    print(f"\n{'='*80}")
    print("STREAMING TOKENIZATION...")
    print(f"{'='*80}")

    total_tokens = 0
    total_examples = 0
    skipped = 0
    token_lengths = []  # For statistics (just lengths, not actual tokens)
    min_tokens = float('inf')
    max_tokens = 0

    # Open output file for writing tokenized examples one by one
    with open(output_path, "wb") as out_f:
        with open(input_path, "r", encoding="utf-8") as in_f:
            current_example_lines = []

            for line in in_f:
                # Remove trailing newline
                line = line.rstrip('\n')

                # Empty line indicates end of example (examples separated by \n\n)
                if line == '':
                    if current_example_lines:
                        # Join lines to reconstruct the example
                        text = '\n'.join(current_example_lines).strip()

                        if text:  # Only process non-empty examples
                            try:
                                # Tokenize this example
                                ids = tokenizer.encode(text)
                                ids.append(tokenizer.eos_token_id)

                                # Write immediately to disk (streaming write)
                                pickle.dump(ids, out_f)

                                # Update statistics (keep only counts, not tokens)
                                token_count = len(ids)
                                total_tokens += token_count
                                total_examples += 1
                                min_tokens = min(min_tokens, token_count)
                                max_tokens = max(max_tokens, token_count)
                                token_lengths.append(token_count)

                                # Progress update every 10k examples
                                if total_examples % 10000 == 0:
                                    print(f"  Processed {total_examples:,} examples | {total_tokens:,} tokens so far")

                            except Exception as e:
                                skipped += 1
                                if skipped <= 10:
                                    print(f"\nWarning: Skipped example due to error: {e}")

                        # Clear the buffer - this is key for streaming!
                        current_example_lines = []
                else:
                    # Accumulate lines for current example
                    current_example_lines.append(line)

            # Handle last example if file doesn't end with blank line
            if current_example_lines:
                text = '\n'.join(current_example_lines).strip()
                if text:
                    try:
                        ids = tokenizer.encode(text)
                        ids.append(tokenizer.eos_token_id)
                        pickle.dump(ids, out_f)

                        token_count = len(ids)
                        total_tokens += token_count
                        total_examples += 1
                        min_tokens = min(min_tokens, token_count)
                        max_tokens = max(max_tokens, token_count)
                        token_lengths.append(token_count)

                    except Exception as e:
                        skipped += 1

    # Print statistics
    print(f"\n{'='*80}")
    print("TOKENIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total examples: {total_examples:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens/example: {total_tokens / total_examples:.1f}" if total_examples > 0 else "N/A")
    print(f"Skipped examples: {skipped:,}")
    print(f"{'='*80}")

    # Calculate token distribution
    if token_lengths:
        print(f"\nToken Distribution:")
        print(f"  Min tokens: {min_tokens:,}")
        print(f"  Max tokens: {max_tokens:,}")
        print(f"  Median tokens: {sorted(token_lengths)[len(token_lengths)//2]:,}")

    # Print file size
    print(f"\n{'='*80}")
    print(f"Saved tokenized data to {output_path}")
    print(f"{'='*80}")
    file_size_bytes = os.path.getsize(output_path)
    file_size_gb = file_size_bytes / (1024 ** 3)
    print(f"✓ Saved successfully!")
    print(f"  File size: {file_size_gb:.2f} GB")

    # Return statistics instead of the actual token data
    return {
        'total_examples': total_examples,
        'total_tokens': total_tokens,
        'skipped': skipped,
        'min_tokens': min_tokens if min_tokens != float('inf') else 0,
        'max_tokens': max_tokens
    }


def verify_tokenized_data(pkl_path):
    """
    Load and verify the tokenized data (STREAMING VERSION)

    The pickle file contains multiple pickled objects (one per example),
    so we read them one at a time without loading everything into memory.

    Args:
        pkl_path: Path to nous_corpus.pkl
    """
    print(f"\n{'='*80}")
    print("VERIFYING TOKENIZED DATA (STREAMING MODE)")
    print(f"{'='*80}")

    print(f"Streaming verification from {pkl_path}...")

    total_examples = 0
    total_tokens = 0
    first_example_ids = None

    # Stream through the pickle file without loading all data
    with open(pkl_path, "rb") as f:
        while True:
            try:
                # Load one example at a time
                ids = pickle.load(f)

                # Capture first example for display
                if first_example_ids is None:
                    first_example_ids = ids

                # Update stats
                total_examples += 1
                total_tokens += len(ids)

            except EOFError:
                # End of file reached
                break

    print(f"✓ Verified successfully!")
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {total_examples:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens/example: {total_tokens / total_examples:.1f}" if total_examples > 0 else "N/A")

    # Show first example if we have one
    if first_example_ids:
        print(f"\nFirst example (first 10 token IDs):")
        print(f"  {first_example_ids[:10]}...")

        # Initialize tokenizer to decode
        tokenizer = TikToken()

        # Decode first example
        decoded = tokenizer.decode(first_example_ids[:100])  # First 100 tokens
        print(f"\nFirst example decoded (first 100 tokens):")
        print(f"  {decoded}...")


def main():
    """
    Main function to tokenize nous_corpus.txt (STREAMING VERSION)
    """
    print("="*80)
    print("NOUS CORPUS TOKENIZER (STREAMING MODE)")
    print("Using TikToken tokenizer")
    print("="*80)

    # Paths
    input_path = get_training_data_path('nous_corpus.txt')
    output_path = get_training_data_path('nous_corpus.pkl')

    # Tokenize (returns statistics dict, not actual tokens)
    stats = tokenize_corpus(
        input_path=input_path,
        output_path=output_path,
        target_tokens=46_000_000_000
    )

    if stats is not None:
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
