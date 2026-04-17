"""
src/training/train_vastai.py

Trains the model on Vast.ai by streaming the dataset shard-by-shard from
HuggingFace Hub. Requires only ~2-4 GB RAM per shard — no full dataset
download needed.

Architecture defaults target ~3.47B parameters:
    embedding_dim=1024, num_blocks=48, num_heads=16, num_experts=8
    → 50304×1024 (embed) + 48 × 68 × 1024² (blocks) = ~3.47B params

Setup on Vast.ai instance:
    pip install huggingface_hub datasets pyarrow tiktoken tqdm
    pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    git clone https://github.com/YOUR/Nous.git && cd Nous

Usage:
    # Fresh training run:
    python src/training/train_vastai.py \\
        --repo your-username/nous-corpus \\
        --token hf_xxxxxxxxxxxxxxxxxxxx

    # Resume from checkpoint:
    python src/training/train_vastai.py \\
        --repo your-username/nous-corpus \\
        --token hf_xxxxxxxxxxxxxxxxxxxx \\
        --checkpoint artifacts/models/checkpoint.pkl

    # Start from a specific shard (useful after a crash):
    python src/training/train_vastai.py \\
        --repo your-username/nous-corpus \\
        --token hf_xxxxxxxxxxxxxxxxxxxx \\
        --start-shard 12
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from src.training.train import Trainer
from src.tokenizer.tiktoken_tokenizer import TikToken
from api.paths import get_models_path


# ---------------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------------

def list_shards(repo_id: str, token: str) -> list:
    """Returns a sorted list of parquet shard paths from the repo's data/ folder."""
    api = HfApi(token=token)
    files = [
        item.path
        for item in api.list_repo_tree(repo_id=repo_id, repo_type="dataset", recursive=True)
        if item.path.endswith(".parquet")
    ]
    return sorted(files)


def load_shard(repo_id: str, remote_path: str, token: str) -> list:
    """
    Downloads one Parquet shard, reads it into a list of token_id lists,
    then deletes the local file to keep disk usage near zero.
    """
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=remote_path,
        repo_type="dataset",
        token=token,
        local_dir="/tmp/hf_shards",
    )
    table     = pq.read_table(local_path)
    token_ids = table["input_ids"].to_pylist()
    os.remove(local_path)
    return token_ids


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

PROGRESS_FILE = "vastai_shard_progress.json"


def load_shard_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"last_completed_shard": -1, "total_examples_trained": 0}


def save_shard_progress(last_completed_shard: int, total_examples_trained: int):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "last_completed_shard": last_completed_shard,
            "total_examples_trained": total_examples_trained,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    tokenizer = TikToken()

    print("Listing shards in HF dataset repo...")
    all_shards = list_shards(args.repo, args.token)
    print(f"Found {len(all_shards)} shards.")

    if not all_shards:
        print("No shards found. Did the upload complete?")
        sys.exit(1)

    # Determine which shard to start from
    progress    = load_shard_progress()
    start_shard = args.start_shard
    if start_shard == 0 and progress["last_completed_shard"] >= 0:
        start_shard = progress["last_completed_shard"] + 1
        print(f"Auto-resuming from shard {start_shard} "
              f"(last completed: {progress['last_completed_shard']})")

    total_examples_trained = progress["total_examples_trained"]

    # Build trainer with no token_ids — we'll inject them shard-by-shard
    trainer = Trainer(
        tokenizer         = tokenizer,
        token_ids         = None,
        num_blocks        = args.num_blocks,
        num_heads         = args.num_heads,
        embedding_dim     = args.embedding_dim,
        max_seq_length    = args.max_seq_length,
        use_moe           = args.use_moe,
        num_experts       = args.num_experts,
        experts_per_token = args.experts_per_token,
        lr                = args.lr,
        min_lr            = args.min_lr,
        use_lr_schedule   = True,
        warmup_steps      = args.warmup_steps,
    )

    checkpoint_path = args.checkpoint or get_models_path("checkpoint.pkl")

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    elif args.checkpoint:
        print(f"Warning: checkpoint not found at {args.checkpoint}, starting fresh.")

    # Epoch × shard loop
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        # On epoch 0 honour start_shard; subsequent epochs train on all shards
        shards_this_epoch = all_shards[start_shard:] if epoch == 0 else all_shards

        for remote_path in shards_this_epoch:
            global_shard_idx = all_shards.index(remote_path)
            print(f"\nShard {global_shard_idx + 1}/{len(all_shards)}  [{remote_path}]")

            print("  Downloading shard...")
            token_ids = load_shard(args.repo, remote_path, args.token)
            print(f"  Loaded {len(token_ids):,} examples")

            # Swap in this shard's data and train one pass over it
            trainer.token_ids = token_ids
            trainer.train(
                epochs          = 1,
                checkpoint_path = checkpoint_path,
                batch_size      = args.batch_size,
                save_every      = 1,
            )

            total_examples_trained += len(token_ids)
            save_shard_progress(global_shard_idx, total_examples_trained)
            print(f"  Cumulative examples trained: {total_examples_trained:,}")

            # Release shard memory before downloading the next one
            trainer.token_ids = None
            del token_ids

    print(f"\nTraining complete.")
    print(f"Total examples trained : {total_examples_trained:,}")
    print(f"Final checkpoint       : {checkpoint_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Nous (~3.47B params) on Vast.ai from HuggingFace Hub"
    )

    # Data / auth
    parser.add_argument("--repo", type=str, required=True,
                        help="HuggingFace dataset repo ID, e.g. your-username/nous-corpus")
    parser.add_argument("--token", type=str, default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace read token (or set HF_TOKEN env var)")
    parser.add_argument("--start-shard", type=int, default=0,
                        help="Shard index to start from (overrides auto-resume)")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to load/save checkpoint .pkl")

    # Training
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of full passes over the entire dataset")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (reduce if OOM)")

    # Model architecture  — ~3.47B params with these defaults
    # embedding_dim=1024, num_blocks=48, num_heads=16, 8 experts
    # Param count: 50304×1024 + 48×(4+64)×1024² ≈ 3.47B
    parser.add_argument("--num-blocks",        type=int,   default=48)
    parser.add_argument("--num-heads",         type=int,   default=16)
    parser.add_argument("--embedding-dim",     type=int,   default=1024)
    parser.add_argument("--max-seq-length",    type=int,   default=1024)
    parser.add_argument("--use-moe",           type=bool,  default=True)
    parser.add_argument("--num-experts",       type=int,   default=8)
    parser.add_argument("--experts-per-token", type=int,   default=2)

    # Optimizer
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--min-lr",        type=float, default=3e-5)
    parser.add_argument("--warmup-steps",  type=int,   default=2000)

    args = parser.parse_args()

    if not args.token:
        print("Error: --token or HF_TOKEN env var required")
        sys.exit(1)

    train(args)


if __name__ == "__main__":
    main()
