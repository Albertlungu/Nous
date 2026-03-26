"""
Upload Corpus to HuggingFace Hub for Streaming Training
Compresses and uploads in chunks to work with 32GB storage on Vast AI.
"""

import os
import sys
import glob
from huggingface_hub import HfApi, create_repo
import zstandard as zstd


def compress_corpus(input_path: str, output_path: str, compression_level: int = 3):
    """
    Compress corpus.txt with zstandard for efficient upload.

    Args:
        input_path: Path to corpus.txt (651GB)
        output_path: Path to save compressed file
        compression_level: Zstandard compression level (3 = balanced)

    Returns:
        Compression ratio
    """
    print("=" * 80)
    print("COMPRESSING CORPUS WITH ZSTANDARD")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Compression level: {compression_level}")
    print("=" * 80)

    original_size = os.path.getsize(input_path)
    print(f"\nOriginal size: {original_size / (1024**3):.2f} GB")

    # Compress
    cctx = zstd.ZstdCompressor(level=compression_level)

    with open(input_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            with cctx.stream_writer(f_out) as compressor:
                # Read in chunks to show progress
                chunk_size = 100 * 1024 * 1024  # 100MB chunks
                bytes_read = 0

                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break

                    compressor.write(chunk)
                    bytes_read += len(chunk)

                    if bytes_read % (1024 * 1024 * 1024) == 0:  # Every 1GB
                        progress = (bytes_read / original_size) * 100
                        print(f"  Progress: {progress:.1f}% ({bytes_read / (1024**3):.1f} GB)")

    compressed_size = os.path.getsize(output_path)
    ratio = original_size / compressed_size

    print("\n" + "=" * 80)
    print("COMPRESSION COMPLETE")
    print("=" * 80)
    print(f"Compressed size: {compressed_size / (1024**3):.2f} GB")
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"Space saved: {(original_size - compressed_size) / (1024**3):.2f} GB")
    print("=" * 80)

    return ratio


def upload_to_hub(
    file_path: str,
    repo_id: str,
    repo_type: str = "dataset",
    private: bool = True,
):
    """
    Upload compressed corpus to HuggingFace Hub.

    Args:
        file_path: Path to compressed corpus file
        repo_id: HuggingFace repo ID (e.g., "your-username/nous-corpus")
        repo_type: Repository type
        private: Make repo private
    """
    print("\n" + "=" * 80)
    print("UPLOADING TO HUGGINGFACE HUB")
    print("=" * 80)
    print(f"File: {file_path}")
    print(f"Repo: {repo_id}")
    print(f"Private: {private}")
    print("=" * 80)

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
        print(f"\n✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"\n✗ Error creating repository: {e}")
        return

    # Upload file
    try:
        print(f"\nUploading {os.path.basename(file_path)}...")
        print("This may take several hours for large files...")

        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
            repo_type=repo_type,
        )

        print(f"\n✓ Upload complete!")
        print(f"View at: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        raise


def create_dataset_card(repo_id: str, total_tokens: int, corpus_size_gb: float):
    """
    Create README.md dataset card for the repository.

    Args:
        repo_id: HuggingFace repo ID
        total_tokens: Total number of tokens
        corpus_size_gb: Corpus size in GB
    """
    readme_content = f"""# Nous Training Corpus

## Dataset Description

This dataset contains the pre-training corpus for the Nous multimodal model.

- **Total tokens:** {total_tokens:,} (~{total_tokens/1e9:.1f}B)
- **Uncompressed size:** {corpus_size_gb:.1f} GB
- **Tokenizer:** TikToken cl100k_base
- **Format:** Plain text, double-newline separated documents

## Data Sources

Mix of high-quality text data:
- FineWeb-Edu (80B tokens)
- OpenWebMath (30B tokens)
- DCLM-Baseline (30B tokens)
- OpenMathReasoning (20B tokens)
- OpenR1-Math (20B tokens)
- ML-ArXiv (20B tokens)
- Wikipedia (15B tokens)
- PG-19 Books (15B tokens)
- GSM8K Enhanced (15B tokens)
- The Stack (10B tokens)

## Usage

### Streaming (Recommended for 32GB storage)

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}", split="train", streaming=True)

for example in dataset:
    text = example["text"]
    # Tokenize and train...
```

### With Streaming Dataloader

```python
from src.data.streaming_dataset import create_dataloader

dataloader = create_dataloader(
    repo_id="{repo_id}",
    batch_size=8,
    seq_length=4096,
    rank=0,
    world_size=4,
)

for batch in dataloader:
    input_ids = batch["input_ids"]  # [8, 4096]
    labels = batch["labels"]
    # Train...
```

## License

This dataset is a compilation of publicly available sources. Each component retains its original license.

## Citation

If you use this dataset, please cite the original sources.
"""

    # Upload README
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"\n✓ Dataset card created")


def main():
    """
    Main function to compress and upload corpus.
    """
    print("=" * 80)
    print("CORPUS UPLOAD PIPELINE")
    print("=" * 80)
    print("This script will:")
    print("  1. Compress corpus.txt with zstandard")
    print("  2. Upload to HuggingFace Hub")
    print("  3. Create dataset card")
    print("=" * 80)

    # Paths
    input_path = "/Volumes/Extreme SSD/Nous/training_data/text/corpus.txt"
    output_path = "/Volumes/Extreme SSD/Nous/training_data/text/corpus.txt.zst"

    # HuggingFace repo
    repo_id = input("\nEnter HuggingFace repo ID (e.g., your-username/nous-corpus): ").strip()

    if not repo_id:
        print("✗ Error: Repository ID is required")
        return

    # Check if corpus exists
    if not os.path.exists(input_path):
        print(f"✗ Error: Corpus not found at {input_path}")
        return

    corpus_size_gb = os.path.getsize(input_path) / (1024**3)
    print(f"\n✓ Found corpus: {corpus_size_gb:.2f} GB")

    # Step 1: Compress
    if not os.path.exists(output_path):
        print("\nStep 1: Compressing corpus...")
        compress_corpus(input_path, output_path, compression_level=3)
    else:
        print(f"\n✓ Compressed corpus already exists: {output_path}")
        compressed_size = os.path.getsize(output_path) / (1024**3)
        print(f"  Size: {compressed_size:.2f} GB")

    # Step 2: Upload
    print("\nStep 2: Uploading to HuggingFace Hub...")
    upload_to_hub(output_path, repo_id, private=True)

    # Step 3: Create dataset card
    print("\nStep 3: Creating dataset card...")
    create_dataset_card(repo_id, total_tokens=170_756_572_334, corpus_size_gb=corpus_size_gb)

    print("\n" + "=" * 80)
    print("UPLOAD COMPLETE!")
    print("=" * 80)
    print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
    print("\nYou can now use this dataset for training on Vast AI with:")
    print(f'  repo_id="{repo_id}"')
    print("=" * 80)


if __name__ == "__main__":
    main()
