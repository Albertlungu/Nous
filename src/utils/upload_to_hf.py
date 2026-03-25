"""
src/utils/upload_to_hf.py

Streams nous_corpus.pkl from HDD → Parquet shards → HuggingFace Hub.

Never loads more than one shard into RAM at a time. Resumes from where it
left off if interrupted.

Prerequisites:
    pip install huggingface_hub pyarrow tqdm

Usage:
    python src/utils/upload_to_hf.py \\
        --pkl "/Volumes/Seagate HDD/training_data/nous_corpus.pkl" \\
        --repo your-username/nous-corpus \\
        --token hf_xxxxxxxxxxxxxxxxxxxx

    # Resume after interruption (reads upload_progress.json automatically):
    python src/utils/upload_to_hf.py \\
        --pkl "/Volumes/Seagate HDD/training_data/nous_corpus.pkl" \\
        --repo your-username/nous-corpus \\
        --token hf_xxxxxxxxxxxxxxxxxxxx \\
        --resume

    # Dry run to count examples without uploading:
    python src/utils/upload_to_hf.py \\
        --pkl "/Volumes/Seagate HDD/training_data/nous_corpus.pkl" \\
        --dry-run
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
import time

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import HfHubHTTPError
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SHARD_SIZE    = 500_000          # examples per shard (~1-2 GB Parquet each)
PROGRESS_FILE = "upload_progress.json"
TEMP_DIR      = tempfile.gettempdir()


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"shard_idx": 0, "file_position": 0, "total_uploaded": 0}


def save_progress(shard_idx, file_position, total_uploaded):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "shard_idx": shard_idx,
            "file_position": file_position,
            "total_uploaded": total_uploaded,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------

def write_shard_to_parquet(buffer: list, path: str):
    """Write a list of token_id sequences to a Parquet file."""
    array = pa.array(buffer, type=pa.large_list(pa.int32()))
    table = pa.table({"input_ids": array})
    pq.write_table(table, path, compression="snappy")


# ---------------------------------------------------------------------------
# Core streaming loop
# ---------------------------------------------------------------------------

def stream_pkl(pkl_path: str, file_position: int = 0):
    """
    Generator that yields (token_ids, file_position_after_read) from the pkl.

    Uses f.seek() for instant resume — no need to deserialize skipped examples.

    Args:
        pkl_path:      Path to nous_corpus.pkl
        file_position: Byte offset to seek to before reading (0 = start)
    """
    with open(pkl_path, "rb") as f:
        if file_position > 0:
            f.seek(file_position)
        while True:
            try:
                ids = pickle.load(f)
                yield ids, f.tell()
            except EOFError:
                return


def upload_dataset(pkl_path: str, repo_id: str, token: str, resume: bool, shard_size: int = SHARD_SIZE):
    api = HfApi(token=token)

    try:
        create_repo(repo_id, repo_type="dataset", token=token, private=True, exist_ok=True)
        print(f"Dataset repo ready: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Warning: could not create repo (may already exist): {e}")

    progress       = load_progress() if resume else {"shard_idx": 0, "file_position": 0, "total_uploaded": 0}
    shard_idx      = progress["shard_idx"]
    file_position  = progress.get("file_position", 0)
    total_uploaded = progress["total_uploaded"]

    # Migrate old progress files that only have examples_skipped (no file_position).
    # Scan forward to find the byte offset, showing progress so it doesn't look frozen.
    if resume and file_position == 0 and progress.get("examples_skipped", 0) > 0:
        skip_count = progress["examples_skipped"]
        print(f"Old progress format detected — scanning to example {skip_count:,} to find byte offset...")
        print("(This only happens once; future resumes will be instant)")
        with open(pkl_path, "rb") as f:
            for _ in tqdm(range(skip_count), desc="Scanning to resume point", unit="ex", dynamic_ncols=True):
                try:
                    pickle.load(f)
                except EOFError:
                    break
            file_position = f.tell()
        save_progress(shard_idx, file_position, total_uploaded)
        print(f"Resume point found at byte {file_position:,}. Saved for instant future resumes.")

    if resume and file_position > 0:
        print(f"Resuming from shard {shard_idx} (seeking to byte {file_position:,})")

    buffer            = []
    last_file_position = file_position

    stream = stream_pkl(pkl_path, file_position=file_position)
    pbar   = tqdm(stream, desc="Streaming examples", unit="ex", dynamic_ncols=True)

    for ids, pos in pbar:
        buffer.append(ids)
        last_file_position = pos

        if len(buffer) >= shard_size:
            shard_path  = os.path.join(TEMP_DIR, f"shard_{shard_idx:05d}.parquet")
            remote_path = f"data/shard_{shard_idx:05d}.parquet"

            write_shard_to_parquet(buffer, shard_path)
            shard_mb = os.path.getsize(shard_path) / 1e6

            t0 = time.time()
            api.upload_file(
                path_or_fileobj=shard_path,
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="dataset",
            )
            elapsed  = time.time() - t0
            speed_mb = shard_mb / elapsed

            os.remove(shard_path)

            total_uploaded += len(buffer)
            shard_idx      += 1

            save_progress(shard_idx, last_file_position, total_uploaded)
            pbar.set_postfix(
                shard=shard_idx,
                total=f"{total_uploaded:,}",
                speed=f"{speed_mb:.1f}MB/s",
            )
            buffer = []

    # Upload final (incomplete) shard
    if buffer:
        shard_path  = os.path.join(TEMP_DIR, f"shard_{shard_idx:05d}.parquet")
        remote_path = f"data/shard_{shard_idx:05d}.parquet"
        write_shard_to_parquet(buffer, shard_path)
        api.upload_file(
            path_or_fileobj=shard_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        os.remove(shard_path)
        total_uploaded += len(buffer)
        shard_idx      += 1
        save_progress(shard_idx, last_file_position, total_uploaded)

    print(f"\nDone. Uploaded {total_uploaded:,} examples across {shard_idx} shards.")
    print(f"Dataset: https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Dry run — just count examples, no upload
# ---------------------------------------------------------------------------

def dry_run(pkl_path: str):
    print(f"Dry run: counting examples in {pkl_path}")
    total        = 0
    total_tokens = 0
    with open(pkl_path, "rb") as f:
        with tqdm(desc="Counting", unit="ex", dynamic_ncols=True) as pbar:
            while True:
                try:
                    ids           = pickle.load(f)
                    total        += 1
                    total_tokens += len(ids)
                    pbar.update(1)
                except EOFError:
                    break
    print(f"\nTotal examples : {total:,}")
    print(f"Total tokens   : {total_tokens:,}")
    print(f"Avg tokens/ex  : {total_tokens / total:.1f}" if total else "N/A")
    print(f"Est. shards    : {(total + SHARD_SIZE - 1) // SHARD_SIZE} (at {SHARD_SIZE:,} ex/shard)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Upload nous_corpus.pkl to HuggingFace Hub")
    parser.add_argument("--pkl", type=str,
                        default="/Volumes/Seagate HDD/training_data/nous_corpus.pkl",
                        help="Path to nous_corpus.pkl")
    parser.add_argument("--repo", type=str, default=None,
                        help="HuggingFace dataset repo ID, e.g. your-username/nous-corpus")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace write token (or set HF_TOKEN env var)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from upload_progress.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count examples only, do not upload")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE,
                        help=f"Examples per shard (default: {SHARD_SIZE:,})")
    args = parser.parse_args()

    if not os.path.exists(args.pkl):
        print(f"Error: pkl not found at {args.pkl}")
        sys.exit(1)

    if args.dry_run:
        dry_run(args.pkl)
        return

    if not args.repo:
        print("Error: --repo is required (e.g. your-username/nous-corpus)")
        sys.exit(1)

    if not args.token:
        print("Error: --token or HF_TOKEN env var required")
        sys.exit(1)

    upload_dataset(args.pkl, args.repo, args.token, args.resume, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
