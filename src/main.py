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
from api.paths import get_training_data_path, get_tokenizer_path, get_models_path

def save_token_ids(output_path):
    """
    Save token ids to a pickled file in order to avoid 10 minutes of prep time during training.
    """

    tokenizer_path = get_tokenizer_path('tokenizer_alpaca.pkl')
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
            if hasattr(tokenizer, '_ensure_vocab'):
                tokenizer._ensure_vocab()
    else:
        tokenizer = TikToken()

    with open(get_training_data_path('alpaca.txt'), "r") as f:
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
    output_path = get_models_path('training_analysis.png')
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
    with open(get_training_data_path('tiktoken_combined_new.pkl'), "rb") as f:
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
        checkpoint_path=get_models_path('alpaca284.pkl'),
        save_every=1,
        prompt="Instruction: List three best practices for starting a conversation.\nInput: \nOutput:"
    )

    end_train = time.time() - train_time
    print("Finished training model.")
    print("="*60)
    print(f"Train time: {end_train:.4f}s")

    # Save lightweight model-only checkpoint (no optimizer state)
    print("\nCreating lightweight checkpoint for inference...")
    trainer.save_model_only(get_models_path('alpaca200_model_only.pkl'))
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
    with open(get_training_data_path('tiktoken_combined_new.pkl'), "rb") as f:
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
    checkpoint_path = get_models_path('epoch155.pkl')
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

def user_input():

    print("You will be asked two questions, one for the instruction the model is meant to complete, and the other for the input it requires. Respond appropriately.")

    time.sleep(2)

    instruction = input("Please enter your instruction here: ")
    argument = input("Please enter your input here, or leave it blank: ")

    prompt = f"Instruction: {instruction}\n" + f"Input: {argument}\n" + f"Output: \n"

    print("Loading tokenizer...")
    tokenizer = TikToken()

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

    checkpoint_path = get_models_path('epoch155.pkl')
    print("Loaded checkpoint.")

    try:
        trainer.load_checkpoint(checkpoint_path)
    except FileNotFoundError():
        print(f"ERROR: No checkpoint found at {checkpoint_path}. Please verify to ensure it exists.")

    print("="*60)
    print(f"Your prompt: \n")
    print(prompt)
    print("="*60)
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
To analyze the current model, please enter 'a'
Or, to enter your own user input, please enter 'i':
""")
    if main_or_train.lower() == 't':
        train()
    elif main_or_train.lower() == 'm':
        main()
    elif main_or_train.lower() == 'e':
        extend()
    elif main_or_train.lower() == 'a':
        analyze_training()
    elif main_or_train.lower() == "ids":
        save_token_ids(get_training_data_path('alpaca_tokenized.pkl'))
    else:
        user_input()

    # main()

    end = time.time()
    print("="*60)
    print(f"Total execution time: {end-start:.4f} s")
    print("="*60)
