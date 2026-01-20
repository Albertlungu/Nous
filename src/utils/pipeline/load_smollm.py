"""
SmolLM-Corpus Pipeline
Streams directly from HuggingFace -> tokenize -> save to pkl
No intermediate .txt file, minimal disk usage

Datasets:
- fineweb-edu-dedup: 220B tokens (educational web content)
- cosmopedia-v2: 28B tokens (synthetic textbooks & stories)
- python-edu: 4B tokens (educational Python code)

Total: ~252B tokens
"""

import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from datasets import load_dataset
from tqdm import tqdm

from src.tokenizer.tiktoken_tokenizer import TikToken


def stream_smollm_corpus(
    target_tokens=100_000_000_000, output_path=None, num_workers=4, batch_size=1000
):
    """
    Stream SmolLM-Corpus directly to tokenized pkl file.
    No intermediate files, no memory accumulation.
    Uses batched tokenization for speed.

    Args:
        target_tokens: Target number of tokens (default 100B)
        output_path: Path to save tokenized data (default: /Volumes/SanDisk128G/Nous/training_data)
        num_workers: Number of parallel tokenization workers (default 4)
        batch_size: Batch size for tokenization (default 1000)

    Returns:
        Statistics dict
    """
    if output_path is None:
        output_path = "/Volumes/SanDisk128G/Nous/training_data/nous_corpus.pkl"

    print("=" * 80)
    print("SMOLLM-CORPUS STREAMING PIPELINE (FAST MODE)")
    print(f"Target: {target_tokens:,} tokens")
    print(f"Output: {output_path}")
    print(f"Workers: {num_workers} | Batch size: {batch_size}")
    print("=" * 80)

    # Initialize tokenizer
    print("\nLoading tokenizer...")
    tokenizer = TikToken()
    print(f"Tokenizer ready (vocab size: {tokenizer.vocab_size:,})")

    # Dataset configuration
    # Proportions roughly match SmolLM training mix
    datasets_config = [
        {
            "name": "fineweb-edu-dedup",
            "dataset": "HuggingFaceTB/smollm-corpus",
            "subset": "fineweb-edu-dedup",
            "text_field": "text",
            "proportion": 0.75,  # 75% educational web
        },
        {
            "name": "cosmopedia-v2",
            "dataset": "HuggingFaceTB/smollm-corpus",
            "subset": "cosmopedia-v2",
            "text_field": "text",
            "proportion": 0.15,  # 15% synthetic textbooks
        },
        {
            "name": "python-edu",
            "dataset": "HuggingFaceTB/smollm-corpus",
            "subset": "python-edu",
            "text_field": "text",
            "proportion": 0.10,  # 10% code
        },
    ]

    # Calculate tokens per dataset
    for config in datasets_config:
        config["target_tokens"] = int(target_tokens * config["proportion"])

    # Stats
    total_tokens = 0
    total_examples = 0
    stats_per_dataset = {}

    # Batch tokenization function
    def tokenize_batch(texts):
        """Tokenize a batch of texts"""
        results = []
        for text in texts:
            if not text or len(text.strip()) < 50:
                continue
            try:
                ids = tokenizer.encode(text)
                ids.append(tokenizer.eos_token_id)
                results.append(ids)
            except Exception:
                continue
        return results

    # Open output file for streaming writes
    print(f"\nStreaming to {output_path}...")

    with open(output_path, "wb") as out_f:
        for config in datasets_config:
            name = config["name"]
            dataset_tokens = 0
            dataset_examples = 0
            target = config["target_tokens"]

            print(f"\n{'=' * 60}")
            print(f"Loading {name}...")
            print(f"Target: {target:,} tokens")
            print("=" * 60)

            try:
                # Load with streaming - no download!
                ds = load_dataset(
                    config["dataset"], config["subset"], split="train", streaming=True
                )

                # Process in batches for speed
                pbar = tqdm(desc=f"Processing {name}", unit=" examples")
                batch = []

                for example in ds:
                    text = example.get(config["text_field"], "")
                    batch.append(text)

                    # Process batch when full
                    if len(batch) >= batch_size:
                        tokenized = tokenize_batch(batch)
                        for ids in tokenized:
                            pickle.dump(ids, out_f)
                            token_count = len(ids)
                            dataset_tokens += token_count
                            dataset_examples += 1
                            total_tokens += token_count
                            total_examples += 1

                        pbar.update(len(batch))
                        pbar.set_postfix(
                            {
                                "tokens": f"{dataset_tokens:,}",
                                "tok/s": f"{total_tokens / (pbar.format_dict['elapsed'] + 0.001):.0f}",
                            }
                        )
                        batch = []

                        # Stop when we hit target
                        if dataset_tokens >= target:
                            break

                # Process remaining batch
                if batch and dataset_tokens < target:
                    tokenized = tokenize_batch(batch)
                    for ids in tokenized:
                        pickle.dump(ids, out_f)
                        token_count = len(ids)
                        dataset_tokens += token_count
                        dataset_examples += 1
                        total_tokens += token_count
                        total_examples += 1
                    pbar.update(len(batch))

                pbar.close()

            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue

            stats_per_dataset[name] = {
                "examples": dataset_examples,
                "tokens": dataset_tokens,
            }
            print(
                f"Finished {name}: {dataset_examples:,} examples, {dataset_tokens:,} tokens"
            )

    # Final stats
    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print("\nPer-dataset breakdown:")
    for name, stats in stats_per_dataset.items():
        print(f"  {name}: {stats['examples']:,} examples, {stats['tokens']:,} tokens")
    print(f"\nTotal: {total_examples:,} examples, {total_tokens:,} tokens")

    # File size
    file_size_gb = os.path.getsize(output_path) / (1024**3)
    print(f"Output file: {file_size_gb:.2f} GB")
    print("=" * 80)

    return {
        "total_examples": total_examples,
        "total_tokens": total_tokens,
        "per_dataset": stats_per_dataset,
        "output_path": output_path,
    }


def main():
    """Run the SmolLM pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="SmolLM-Corpus Streaming Pipeline")
    parser.add_argument(
        "--tokens",
        type=int,
        default=100_000_000_000,
        help="Target tokens (default: 100B)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: training_data/nous_corpus.pkl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for tokenization (default: 1000)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of workers (default: 4)"
    )
    args = parser.parse_args()

    stats = stream_smollm_corpus(
        target_tokens=args.tokens,
        output_path=args.output,
        num_workers=args.workers,
        batch_size=args.batch_size,
    )

    print("\nReady for training!")
    print(f"Total tokens: {stats['total_tokens']:,}")


if __name__ == "__main__":
    main()
