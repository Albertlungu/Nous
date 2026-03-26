"""
Estimate Token Count in Corpus
Samples 10 chunks from different positions in the corpus and estimates total tokens.
Uses tiktoken for accurate token counting.
"""

import os
import sys
import tiktoken
import random


def sample_from_position(file_path, position, sample_size=10000):
    """
    Read sample_size bytes starting from position in file.

    Args:
        file_path: Path to corpus file
        position: Byte position to start reading
        sample_size: Number of bytes to read

    Returns:
        String of text from that position
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        f.seek(position)
        # Read a bit extra to avoid cutting mid-word
        text = f.read(sample_size + 1000)
        # Truncate to sample_size characters (not bytes, for cleaner boundaries)
        return text[:sample_size]


def estimate_tokens_in_corpus(corpus_path, num_samples=10, sample_size=10000):
    """
    Estimate total tokens by sampling from different positions.

    Args:
        corpus_path: Path to corpus.txt
        num_samples: Number of samples to take (default 10)
        sample_size: Size of each sample in characters (default 10k chars)

    Returns:
        Estimated total tokens
    """
    print("=" * 80)
    print("ESTIMATING TOKENS IN CORPUS")
    print("=" * 80)

    # Get file size
    file_size_bytes = os.path.getsize(corpus_path)
    file_size_gb = file_size_bytes / (1024 ** 3)

    print(f"Corpus file: {corpus_path}")
    print(f"File size: {file_size_gb:.2f} GB ({file_size_bytes:,} bytes)")
    print(f"Sampling strategy: {num_samples} samples of {sample_size:,} characters each")
    print("=" * 80)

    # Initialize tokenizer (using tiktoken cl100k_base - same as GPT-4)
    enc = tiktoken.get_encoding("cl100k_base")

    # Calculate sample positions (evenly distributed throughout file)
    # Avoid the very end to prevent incomplete documents
    max_position = file_size_bytes - (sample_size * 2)
    positions = [int(i * max_position / (num_samples - 1)) for i in range(num_samples)]

    print("\nSampling positions:")
    for i, pos in enumerate(positions):
        pos_gb = pos / (1024 ** 3)
        pos_pct = (pos / file_size_bytes) * 100
        print(f"  Sample {i+1}: {pos_gb:.2f} GB ({pos_pct:.1f}% through file)")

    print("\n" + "=" * 80)
    print("TOKENIZING SAMPLES...")
    print("=" * 80)

    token_counts = []
    byte_counts = []

    for i, position in enumerate(positions):
        # Read sample
        sample_text = sample_from_position(corpus_path, position, sample_size)

        # Count tokens
        tokens = enc.encode(sample_text)
        num_tokens = len(tokens)
        num_bytes = len(sample_text.encode('utf-8'))

        token_counts.append(num_tokens)
        byte_counts.append(num_bytes)

        # Calculate ratio for this sample
        tokens_per_byte = num_tokens / num_bytes if num_bytes > 0 else 0

        print(f"\nSample {i+1}:")
        print(f"  Characters: {len(sample_text):,}")
        print(f"  Bytes: {num_bytes:,}")
        print(f"  Tokens: {num_tokens:,}")
        print(f"  Tokens/byte: {tokens_per_byte:.6f}")
        print(f"  First 100 chars: {sample_text[:100]}...")

    print("\n" + "=" * 80)
    print("ESTIMATION RESULTS")
    print("=" * 80)

    # Calculate statistics
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_bytes = sum(byte_counts) / len(byte_counts)
    avg_tokens_per_byte = sum(token_counts) / sum(byte_counts)

    # Calculate standard deviation for confidence interval
    import statistics
    std_tokens_per_byte = statistics.stdev([tc/bc for tc, bc in zip(token_counts, byte_counts)])

    print(f"\nSample Statistics:")
    print(f"  Average tokens per sample: {avg_tokens:,.0f}")
    print(f"  Average bytes per sample: {avg_bytes:,.0f}")
    print(f"  Average tokens/byte: {avg_tokens_per_byte:.6f}")
    print(f"  Std dev tokens/byte: {std_tokens_per_byte:.6f}")

    # Estimate total tokens
    estimated_total_tokens = int(file_size_bytes * avg_tokens_per_byte)
    estimated_total_billions = estimated_total_tokens / 1_000_000_000

    # Calculate confidence interval (95% confidence)
    margin_of_error = 1.96 * std_tokens_per_byte
    lower_bound = int(file_size_bytes * (avg_tokens_per_byte - margin_of_error))
    upper_bound = int(file_size_bytes * (avg_tokens_per_byte + margin_of_error))
    lower_bound_billions = lower_bound / 1_000_000_000
    upper_bound_billions = upper_bound / 1_000_000_000

    print("\n" + "=" * 80)
    print("FINAL ESTIMATE")
    print("=" * 80)
    print(f"Estimated total tokens: {estimated_total_tokens:,}")
    print(f"Estimated total tokens: {estimated_total_billions:.2f}B")
    print(f"\n95% Confidence Interval:")
    print(f"  Lower bound: {lower_bound:,} ({lower_bound_billions:.2f}B)")
    print(f"  Upper bound: {upper_bound:,} ({upper_bound_billions:.2f}B)")

    # Additional metrics
    avg_chars_per_token = 1 / avg_tokens_per_byte
    print(f"\nAdditional Metrics:")
    print(f"  Average characters per token: {avg_chars_per_token:.2f}")
    print(f"  Compression ratio (bytes/token): {1/avg_tokens_per_byte:.2f}")

    print("=" * 80)

    return estimated_total_tokens, lower_bound, upper_bound


def main():
    corpus_path = "/Volumes/Extreme SSD/Nous/training_data/text/corpus.txt"

    if not os.path.exists(corpus_path):
        print(f"ERROR: Corpus file not found at {corpus_path}")
        print("Make sure the SSD is mounted.")
        return

    # Estimate with 10 samples of 10k characters each
    estimate_tokens_in_corpus(corpus_path, num_samples=10, sample_size=10000)


if __name__ == "__main__":
    main()
