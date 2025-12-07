import os
# import numpy as np
import sys
# import jax
# import jax.numpy as jnp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import pickle
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
# from datasets import load_dataset

import src.utils.config
from embeddings.embeddings import EmbeddingLayer
from src.tokenizer.tiktoken_tokenizer import TikToken
from tokenizer.tokenizer_class import BPETokenizer
from training.train import Trainer

def save_token_ids(output_path):
    """
    Save token ids to a pickled file in order to avoid 10 minutes of prep time during training.
    """

    with open("artifacts/tokenizer/tokenizer_alpaca.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        tokenizer._ensure_vocab()

    with open("training_data/alpaca.txt", "r") as f:
        content = f.read()

    # Split by double newlines to get complete instruction-response pairs
    training_texts = [doc.strip() for doc in content.split('\n\n') if doc.strip()]

    token_ids = []
    for text in tqdm(training_texts):
        ids = tokenizer.encode(text)
        ids.append(tokenizer.eos_token_id)
        token_ids.append(ids)

    with open(output_path, "wb") as f:
        pickle.dump(token_ids, f)

def inspect_model(model_path="artifacts/models/model.pkl"):
    """
    Inspect a model checkpoint and display its metadata.
    """
    print("="*80)
    print(f"INSPECTING MODEL: {model_path}")
    print("="*80)

    try:
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)

        # Display available keys
        print("\nCheckpoint contains:")
        for key in checkpoint.keys():
            print(f"  - {key}")

        # Display metadata if available
        if 'metadata' in checkpoint:
            meta = checkpoint['metadata']
            print("\n" + "="*80)
            print("MODEL METADATA")
            print("="*80)

            print(f"\nModel Info:")
            print(f"  Name: {meta['model_info']['name']}")
            print(f"  Version: {meta['model_info']['version']}")
            print(f"  Last Updated: {meta['model_info']['last_updated']}")

            print(f"\nArchitecture:")
            print(f"  Total Parameters: {meta['architecture']['total_parameters']:,}")
            print(f"  Embedding Dimension: {meta['architecture']['embedding_dim']}")
            print(f"  Blocks: {meta['architecture']['num_blocks']}")
            print(f"  Attention Heads: {meta['architecture']['num_heads']}")
            print(f"  Vocabulary Size: {meta['architecture']['vocab_size']:,}")

            print(f"\nTraining Data:")
            print(f"  Total Examples: {meta['training_data']['total_examples']:,}")
            print(f"  Total Tokens: {meta['training_data']['total_tokens']:,}")
            print(f"  Avg Tokens/Example: {meta['training_data']['avg_tokens_per_example']}")

            print(f"\nTraining History:")
            print(f"  Epochs Completed: {meta['training_history']['epochs_completed']}")
            print(f"  Total Steps: {meta['training_history']['total_steps']:,}")
            if meta['training_history']['initial_loss']:
                print(f"  Initial Loss: {meta['training_history']['initial_loss']:.4f}")
            if meta['training_history']['final_loss']:
                print(f"  Final Loss: {meta['training_history']['final_loss']:.4f}")

            if meta['training_history']['losses']:
                print(f"\n  Recent losses (last 10 epochs):")
                for i, loss in enumerate(meta['training_history']['losses']):
                    print(f"    Epoch -{len(meta['training_history']['losses'])-i}: {loss:.4f}")

        else:
            print("\nNo metadata found in checkpoint.")

        # Display training history if available
        if 'training_history' in checkpoint:
            hist = checkpoint['training_history']
            print("\n" + "="*80)
            print("TRAINING HISTORY")
            print("="*80)
            print(f"Epochs Completed: {hist['epochs_completed']}")
            print(f"Total Steps: {hist['total_steps']}")
            print(f"Losses tracked: {len(hist['losses'])}")
            if hist['losses']:
                print(f"Loss range: {min(hist['losses']):.4f} - {max(hist['losses']):.4f}")

        print("\n" + "="*80)

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")

def analyze_training():
    """
    Analyze and visualize training progress across all training runs.
    Creates a loss curve and displays model statistics.
    """
    print("="*80)
    print("TRAINING ANALYSIS & MODEL STATISTICS")
    print("="*80)

    # Define training log files and how many epochs to take from each
    training_logs = [
        ("training_logs/train_medium_size.txt", 45, "Initial Training (1-45)"),
        ("training_logs/extend_medium.txt", 50, "Extended from epoch 45 (46-95)"),
        ("training_logs/extend_medium_partt_2.txt", 25, "Extended from epoch 50 (51-75)"),
        ("training_logs/extend_medium_epoch120_condt.txt", 35, "Final Run from epoch 25 (26-60)")
    ]

    # Extract loss values from training logs
    all_losses = []
    all_epochs = []
    epoch_counter = 0

    for log_file, num_epochs, label in training_logs:
        try:
            with open(log_file, 'r') as f:
                content = f.read()

            # Extract epoch and loss using regex
            pattern = r'Epoch (\d+)/\d+ complete\. Avg loss: ([\d.]+)'
            matches = re.findall(pattern, content)

            # Take only the specified number of epochs
            for i, (epoch_num, loss) in enumerate(matches[:num_epochs]):
                epoch_counter += 1
                all_epochs.append(epoch_counter)
                all_losses.append(float(loss))

            print(f"\nLoaded {len(matches[:num_epochs])} epochs from {log_file.split('/')[-1]}")

        except FileNotFoundError:
            print(f"Warning: {log_file} not found, skipping...")

    # Create the loss curve plot
    plt.figure(figsize=(14, 8))

    # Plot the loss curve
    plt.subplot(2, 1, 1)
    plt.plot(all_epochs, all_losses, linewidth=2, color='#2E86AB', marker='o', markersize=3)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Average Loss', fontsize=12, fontweight='bold')
    plt.title('Training Loss Over Time - PyGPT Model', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add annotations for key milestones
    if len(all_losses) > 0:
        plt.axhline(y=all_losses[0], color='r', linestyle='--', alpha=0.3, label=f'Initial: {all_losses[0]:.4f}')
        plt.axhline(y=all_losses[-1], color='g', linestyle='--', alpha=0.3, label=f'Final: {all_losses[-1]:.4f}')
        plt.legend()

    # Display model statistics in a text box
    plt.subplot(2, 1, 2)
    plt.axis('off')

    stats_text = f"""
MODEL SPECIFICATIONS

Architecture:
  • Total Parameters: 76,895,360
  • Embedding Dimension: 512
  • Number of Blocks: 8
  • Attention Heads: 8
  • Vocabulary Size: 50,304 (TikToken)
  • Max Sequence Length: 256

Training Data:
  • Total Examples: 448,382
  • Total Tokens: 39,657,127
  • Average Tokens/Example: 88.4
  • Datasets: Alpaca (51,974), WizardLM (70,004),
              FLAN-50k (50,000), GPT-Teacher (89,260)

Training Configuration:
  • Batch Size: 64
  • Learning Rate: 0.0012 → 5e-06 (cosine decay)
  • Total Training Steps: 1,191,020
  • Backend: JAX on RTX 5090

Performance:
  • Initial Loss: {all_losses[0]:.4f}
  • Final Loss: {all_losses[-1]:.4f}
  • Loss Reduction: {((all_losses[0] - all_losses[-1]) / all_losses[0] * 100):.1f}%
  • Total Epochs Analyzed: {len(all_epochs)}
"""

    plt.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # Save the plot
    output_path = "artifacts/training_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plot saved to: {output_path}")
    print(f"{'='*80}")

    # Display the plot
    plt.show()

    # Print summary statistics
    print("\nTRAINING SUMMARY:")
    print(f"  Total epochs analyzed: {len(all_epochs)}")
    print(f"  Initial loss: {all_losses[0]:.4f}")
    print(f"  Final loss: {all_losses[-1]:.4f}")
    print(f"  Loss reduction: {all_losses[0] - all_losses[-1]:.4f} ({((all_losses[0] - all_losses[-1]) / all_losses[0] * 100):.1f}%)")
    print(f"  Average loss per epoch: {sum(all_losses) / len(all_losses):.4f}")
    print("="*80)

def train():
    """Train a new model from scratch with JAX architecture."""

    print("This code is running.")
    # Load tokenizer - using TikToken
    tokenizer = TikToken()
    print(f"Loaded TikToken tokenizer with vocab size: {tokenizer.vocab_size}")

    # with open("artifacts/tokenizer/tokenizer_alpaca.pkl", "rb") as f:
    #     tokenizer = pickle.load(f)
    #     tokenizer._ensure_vocab()

    # Load the new tokenized dataset
    with open("training_data/tiktoken_combined_new.pkl", "rb") as f:
        token_ids = pickle.load(f)

    # with open("training_data/alpaca_tokenized.pkl", "rb") as f:
    #     token_ids = pickle.load(f)

    print("="*60)
    print("Appended training texts to list")
    print("="*60)

    trainer = Trainer(
        tokenizer=tokenizer,
        token_ids=token_ids,
        lr=1.1e-3,  # Base learning rate
        num_blocks=8,
        num_heads=8,
        embedding_dim=512,  # Must be divisible by num_heads
        max_seq_length=256,  # Covers 95% of examples (most are under 200 tokens)
        dropout=0.0,
        use_lr_schedule=True,  # Enable warmup + cosine decay
        warmup_steps=500,  # Warmup for first 500 steps
        min_lr=5e-6  # Minimum learning rate floor
    )


    # Print model architecture summary
    print("="*60)
    print("MODEL SUMMARY")
    trainer.print_model_summary()
    print("="*60)

    print("Training model.")
    train_time = time.time()

    # Train with automatic checkpointing
    trainer.train(
        epochs=75,
        batch_size=64,
        checkpoint_path="artifacts/model/alpaca284.pkl",
        save_every=1,
        prompt="Instruction: List three best practices for starting a conversation.\nInput: \nOutput:"
    )

    end_train = time.time() - train_time
    print("Finished training model.")
    print("="*60)
    print(f"Train time: {end_train:.4f}s")

    # Save lightweight model-only checkpoint (no optimizer state)
    print("\nCreating lightweight checkpoint for inference...")
    trainer.save_model_only("artifacts/model/alpaca200_model_only.pkl")
    print("Lightweight checkpoint saved!")

    # Test generation
    prompt = "Instruction: List three best practices for starting a conversation.\nInput: \nOutput:"
    generated_text = trainer.generate(prompt, max_length=200)
    print("="*60)
    print("Generated text:")
    print(generated_text)
    print("="*60)

def extend():

    tokenizer = TikToken()
    print(f"Loaded TikToken tokenizer with vocab size: {tokenizer.vocab_size}")

    # Load token_ids to match train() function
    with open("training_data/tiktoken_combined_new.pkl", "rb") as f:
        token_ids = pickle.load(f)

    trainer = Trainer(
        tokenizer=tokenizer,
        token_ids=token_ids,
        lr=1.2e-3,  # Base learning rate
        num_blocks=8,  # Must match checkpoint!
        num_heads=8,   # Must match checkpoint!
        embedding_dim=512,  # Must match checkpoint!
        max_seq_length=256,
        dropout=0.0,
        use_lr_schedule=True,
        warmup_steps=500,
        min_lr=5e-6
    )

    trainer.extend_training(
        checkpoint_path="",
        epochs=50,
        batch_size=64,
        save_every=1,
        prompt="Instruction: List three best practices for starting a conversation.\nInput: \nOutput:"
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
        # token_ids=token_ids,
        lr=6e-4,  # Slightly higher base LR with schedule
        num_blocks=8,  # Must match checkpoint!
        num_heads=8,   # Must match checkpoint!
        embedding_dim=512,  # Must match checkpoint!
        max_seq_length=256,  # Chunk long sequences to avoid memory issues
        dropout=0.0,
        use_lr_schedule=True,  # Enable warmup + cosine decay
        warmup_steps=500,  # Warmup for first 500 steps
        min_lr=1e-5  # Minimum learning rate floor
    )

    # Use latest checkpoint
    checkpoint_path = "artifacts/models/epoch155.pkl"
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
        "Instruction: Describe an interesting article you read recently.\nInput: \nOutput: "
    ]

    for prompt in prompts:
        print("="*60)
        print(f"Prompt: {prompt}")
        print("="*60)

        generated_text = trainer.generate(
            prompt,
            max_length=400,
        )

        # print(f"Generated: '{generated_text[0]}'")
        print()


if __name__ == "__main__":
    print("Hello World - Starting PyGPT")
    start = time.time()

    main_or_train = input("M/T/E/A/I/ids? ")
    if main_or_train.lower() == 't':
        train()
    elif main_or_train.lower() == 'e':
        extend()
    elif main_or_train.lower() == 'a':
        analyze_training()
    elif main_or_train.lower() == 'i':
        inspect_model()
    elif main_or_train.lower() == "ids":
        save_token_ids("training_data/alpaca_tokenized.pkl")
    else:
        main()

    # main()

    end = time.time()
    print("="*60)
    print(f"Total execution time: {end-start:.4f} s")
    print("="*60)
