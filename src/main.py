import os

# import numpy as np
import sys

# Suppress verbose XLA compilation logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# import jax
# import jax.numpy as jnp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import time

from tqdm import tqdm

from api.paths import get_models_path, get_tokenizer_path, get_training_data_path

# from datasets import load_dataset
from src.tokenizer.tiktoken_tokenizer import TikToken
from training.train import Trainer


def save_token_ids(output_path):
    """
    Save token ids to a pickled file in order to avoid 10 minutes of prep time during training.
    """

    tokenizer_path = get_tokenizer_path("tokenizer_alpaca.pkl")
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
            if hasattr(tokenizer, "_ensure_vocab"):
                tokenizer._ensure_vocab()
    else:
        tokenizer = TikToken()

    with open(get_training_data_path("alpaca.txt"), "r") as f:
        content = f.read()

    # Split by double newlines to get complete instruction-response pairs
    training_texts = [doc.strip() for doc in content.split("\n\n") if doc.strip()]

    token_ids = []
    for text in tqdm(training_texts):
        ids = tokenizer.encode(text)
        ids.append(tokenizer.eos_token_id)
        token_ids.append(ids)

    with open(output_path, "wb") as f:
        pickle.dump(token_ids, f)


def train():
    """Train a new model from scratch with JAX architecture."""

    print("This code is running.")
    # Load tokenizer - using TikToken
    tokenizer = TikToken()
    print(f"Loaded TikToken tokenizer with vocab size: {tokenizer.vocab_size}")

    from src.data.jax_streaming_loader import JAXStreamingLoader

    # Training configuration
    batch_size = 8
    seq_length = 1536
    target_tokens = 6_000_000_000  # 6B tokens

    streaming_loader = JAXStreamingLoader(
        repo_id="albertlungu/final-nous-corpus",
        filename="corpus.txt.zst",
        tokenizer_name="cl100k_base",
        batch_size=batch_size,
        seq_length=seq_length,
        shuffle=True,  # Shuffle to mix code, math, and text
    )

    trainer = Trainer(
        tokenizer=tokenizer,
        lr=4e-4,  # Increased from 3e-4 for faster loss decline
        num_blocks=16,  # 700M model
        num_heads=16,
        embedding_dim=1024,
        max_seq_length=1536,
        use_moe=True,
        num_experts=4,
        experts_per_token=2,
        dropout=0.0,
        use_lr_schedule=True,  # Warmup + cosine decay within epoch
        warmup_steps=2000,
        min_lr=3e-5,
        load_balance_coef=0.01,
        use_multi_gpu=False,
        gradient_accumulation_steps=1,
    )

    # Print model architecture summary
    print("=" * 60)
    print("MODEL SUMMARY")
    trainer.print_model_summary()
    print("=" * 60)

    print("Training model.")
    train_time = time.time()

    # Calculate batches needed for target tokens
    tokens_per_batch = batch_size * seq_length
    max_batches = int(target_tokens / tokens_per_batch)
    actual_tokens = max_batches * tokens_per_batch

    print(f"\nTraining configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Tokens per batch: {tokens_per_batch:,}")
    print(f"  Target tokens: {target_tokens:,} ({target_tokens/1e9:.1f}B)")
    print(f"  Batches needed: {max_batches:,}")
    print(f"  Actual tokens: {actual_tokens:,} ({actual_tokens/1e9:.2f}B)")
    print(f"  Dataset: 221.62 GB compressed (~174.9B tokens available)")
    print(f"  Estimated time at 4.13 it/s: {max_batches / 4.13 / 3600:.1f} hours\n")

    trainer.train(
        data_loader=streaming_loader,
        epochs=1,
        checkpoint_path="artifacts/models/nous_700m_6b_tokens.pkl",
        save_every=1,
        prompt="The meaning of life is",
        max_batches=max_batches,
    )

    end_train = time.time() - train_time
    print("Finished training model.")
    print("=" * 60)
    print(f"Train time: {end_train:.4f}s")

    # Save lightweight model-only checkpoint (no optimizer state)
    print("\nCreating lightweight checkpoint for inference...")
    # trainer.save_model_only(get_models_path("alpaca200_model_only.pkl"))
    print("Lightweight checkpoint saved!")

    # Test generation
    prompt = "Instruction: List three best practices for starting a conversation.\nInput: \nOutput:"
    generated_text = trainer.generate(prompt, max_length=200)
    print("=" * 60)
    print("Generated text:")
    print(generated_text)
    print("=" * 60)


def extend():
    tokenizer = TikToken()
    print(f"Loaded TikToken tokenizer with vocab size: {tokenizer.vocab_size}")

    from src.data.jax_streaming_loader import JAXStreamingLoader

    streaming_loader = JAXStreamingLoader(
        repo_id="albertlungu/final-nous-corpus",
        filename="corpus.txt.zst",
        tokenizer_name="cl100k_base",
        batch_size=64,
        seq_length=256,
    )

    trainer = Trainer(
        tokenizer=tokenizer,
        lr=1.2e-3,  # Base learning rate
        num_blocks=8,  # Must match checkpoint!
        num_heads=8,  # Must match checkpoint!
        embedding_dim=512,  # Must match checkpoint!
        max_seq_length=256,
        use_moe=True,
        num_experts=8,
        experts_per_token=2,
        dropout=0.0,
        use_lr_schedule=True,
        warmup_steps=500,
        min_lr=5e-6,
        load_balance_coef=0.01,
    )

    trainer.extend_training(
        data_loader=streaming_loader,
        checkpoint_path="",
        epochs=50,
        save_every=1,
        prompt="Instruction: List three best practices for starting a conversation.\nInput: \nOutput:",
    )


def main():
    """Load a trained model and generate text."""

    print("Loading tokenizer...")
    # with open("artifacts/tokenizer/tokenizer_alpaca.pkl", "rb") as f:
    #     tokenizer = pickle.load(f)
    #     tokenizer._ensure_vocab()
    tokenizer = TikToken()
    print(f"Loaded TikToken tokenizer with vocab size: {tokenizer.vocab_size}")

    # Load token_ids to match train() function
    # with open("training_data/tiktoken_combined_new.pkl", "rb") as f:
    #     token_ids = pickle.load(f)

    # Create trainer with same architecture as checkpoint
    trainer = Trainer(
        tokenizer=tokenizer,
        lr=1.2e-3,  # Base learning rate
        num_blocks=8,  # Must match checkpoint!
        num_heads=8,  # Must match checkpoint!
        embedding_dim=512,  # Must match checkpoint!
        max_seq_length=256,
        use_moe=True,
        num_experts=8,
        experts_per_token=2,
        dropout=0.0,
        use_lr_schedule=True,
        warmup_steps=500,
        min_lr=5e-6,
        load_balance_coef=0.01,
    )

    # Use latest checkpoint
    checkpoint_path = get_models_path("epoch155.pkl")
    print(f"Loading checkpoint from: {checkpoint_path}")

    try:
        trainer.load_checkpoint(checkpoint_path)
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("You need to train a new model first!")
        print("Run: train() function to create a new JAX checkpoint")
        return

    # Test multiple prompts - simpler examples from the dataset
    prompts = [
        "Instruction: List three best practices for starting a conversation.\nInput: \nOutput: ",
        "Instruction: Describe an interesting article you read recently.\nInput: \nOutput: ",
    ]

    for prompt in prompts:
        print("=" * 60)
        print(f"Prompt: {prompt}")
        print("=" * 60)

        generated_text = trainer.generate(
            prompt,
            max_length=400,
        )

        # print(f"Generated: '{generated_text[0]}'")
        print()


def user_input():
    print(
        "You will be asked two questions, one for the instruction the model is meant to complete, and the other for the input it requires. Respond appropriately."
    )

    time.sleep(2)

    instruction = input("Please enter your instruction here: ")
    argument = input("Please enter your input here, or leave it blank: ")

    prompt = f"Instruction: {instruction}\n" + f"Input: {argument}\n" + "Output: \n"

    print("Loading tokenizer...")
    tokenizer = TikToken()

    trainer = Trainer(
        tokenizer=tokenizer,
        # token_ids=token_ids,
        lr=6e-4,  # Slightly higher base LR with schedule
        num_blocks=8,  # Must match checkpoint!
        num_heads=8,  # Must match checkpoint!
        embedding_dim=512,  # Must match checkpoint!
        max_seq_length=256,  # Chunk long sequences to avoid memory issues
        dropout=0.0,
        use_lr_schedule=True,  # Enable warmup + cosine decay
        warmup_steps=500,  # Warmup for first 500 steps
        min_lr=1e-5,  # Minimum learning rate floor
    )

    checkpoint_path = get_models_path("epoch155.pkl")
    print("Loaded checkpoint.")

    try:
        trainer.load_checkpoint(checkpoint_path)
    except FileNotFoundError:
        print(
            f"ERROR: No checkpoint found at {checkpoint_path}. Please verify to ensure it exists."
        )

    print("=" * 60)
    print("Your prompt: \n")
    print(prompt)
    print("=" * 60)
    print("Generated: \n")
    generated_text = trainer.generate(
        prompt,
        max_length=400,
    )
    print("\n")


if __name__ == "__main__":
    print("Hello World - Starting PyGPT")
    start = time.time()

    main_or_train = input("""
To train the model from scratch, please enter 't'
To extend from a previous checkpoint, please enter 'e'
To use the main function, where the model responds to harcoded inputs that come directly from the training data, please enter 'm'
Or, to enter your own user input, please enter 'i':
""")
    if main_or_train.lower() == "t":
        train()
    elif main_or_train.lower() == "m":
        main()
    elif main_or_train.lower() == "e":
        extend()
    elif main_or_train.lower() == "ids":
        save_token_ids(get_training_data_path("alpaca_tokenized.pkl"))
    else:
        user_input()

    # main()

    end = time.time()
    print("=" * 60)
    print(f"Total execution time: {end - start:.4f} s")
    print("=" * 60)
