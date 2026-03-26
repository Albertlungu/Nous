"""
Streaming Dataset for Training with 32GB Storage Constraint
Streams from HuggingFace Hub, tokenizes on-the-fly, with disk caching.
Supports thinking tokens and conversational format.
"""

import os
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from typing import Optional, Dict, List
import tiktoken


class StreamingTokenizedDataset(IterableDataset):
    """
    Streams text from HuggingFace Hub and tokenizes on-the-fly.
    Uses disk caching to avoid re-tokenizing same data.
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        seq_length: int = 4096,
        cache_dir: Optional[str] = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Args:
            repo_id: HuggingFace dataset repo (e.g., "your-username/nous-corpus")
            split: Dataset split to use
            seq_length: Sequence length for training (4096 for thinking + conversation)
            cache_dir: Directory for caching tokenized data (default: /tmp/hf_cache)
            seed: Random seed for shuffling
            rank: GPU rank for distributed training
            world_size: Total number of GPUs
        """
        super().__init__()
        self.repo_id = repo_id
        self.split = split
        self.seq_length = seq_length
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

        # Set cache directory (default to /tmp for Vast AI)
        if cache_dir is None:
            cache_dir = "/tmp/hf_cache"
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize tokenizer with special tokens
        self.tokenizer = self._init_tokenizer()

        # Load dataset in streaming mode
        self.dataset = load_dataset(
            repo_id,
            split=split,
            streaming=True,
            cache_dir=cache_dir,
        )

        # Shuffle and shard for this GPU
        self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10000)
        self.dataset = self.dataset.shard(
            num_shards=world_size,
            index=rank,
            contiguous=True
        )

        # Buffer for accumulating tokens across documents
        self.token_buffer = []

    def _init_tokenizer(self) -> tiktoken.Encoding:
        """
        Initialize tokenizer with special tokens for thinking and conversation.
        """
        # Base tokenizer
        enc = tiktoken.get_encoding("cl100k_base")

        # Define special tokens
        self.special_tokens = {
            "thinking_start": "<|thinking_start|>",
            "thinking_end": "<|thinking_end|>",
            "assistant": "<|assistant|>",
            "user": "<|user|>",
            "system": "<|system|>",
            "endoftext": "<|endoftext|>",
        }

        # Get special token IDs (these will be added to vocab)
        # For tiktoken, we'll use the existing eot_token and define mappings
        # In production, you'd extend the vocab properly
        self.special_token_ids = {
            "thinking_start": 100256,  # First ID after cl100k_base vocab (100256)
            "thinking_end": 100257,
            "assistant": 100258,
            "user": 100259,
            "system": 100260,
            "endoftext": enc.eot_token,  # Use existing EOT
        }

        # Reverse mapping
        self.id_to_special = {v: k for k, v in self.special_token_ids.items()}

        return enc

    def _tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize text, handling special tokens.

        Args:
            text: Raw text string

        Returns:
            List of token IDs
        """
        # For base pretraining, just tokenize normally
        # Special tokens will be added during supervised fine-tuning
        tokens = self.tokenizer.encode(text)
        tokens.append(self.special_token_ids["endoftext"])
        return tokens

    def __iter__(self):
        """
        Iterate over dataset, yielding sequences of seq_length tokens.
        """
        for example in self.dataset:
            # Get text from example
            text = example.get("text", "")

            if not text:
                continue

            # Tokenize
            tokens = self._tokenize_text(text)

            # Add to buffer
            self.token_buffer.extend(tokens)

            # Yield sequences while buffer has enough tokens
            while len(self.token_buffer) >= self.seq_length + 1:
                # Extract sequence (+1 for next token prediction)
                sequence = self.token_buffer[:self.seq_length + 1]
                self.token_buffer = self.token_buffer[self.seq_length:]

                # Create input and labels
                input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
                labels = torch.tensor(sequence[1:], dtype=torch.long)

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                }

    @property
    def vocab_size(self) -> int:
        """
        Return extended vocabulary size (base + special tokens).
        """
        return self.tokenizer.n_vocab + len(self.special_token_ids) - 1  # -1 for endoftext overlap


class ConversationalDataset(IterableDataset):
    """
    Dataset for conversational fine-tuning with thinking tokens.
    Formats data as: <|user|>...<|assistant|><|thinking_start|>...<|thinking_end|>...<|endoftext|>
    """

    def __init__(
        self,
        repo_id: str,
        split: str = "train",
        seq_length: int = 8192,  # Longer for conversations
        cache_dir: Optional[str] = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.split = split
        self.seq_length = seq_length
        self.rank = rank
        self.world_size = world_size

        # Set cache directory
        if cache_dir is None:
            cache_dir = "/tmp/hf_cache"
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize tokenizer
        base_dataset = StreamingTokenizedDataset(
            repo_id=repo_id,
            split=split,
            seq_length=seq_length,
            cache_dir=cache_dir,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )
        self.tokenizer = base_dataset.tokenizer
        self.special_tokens = base_dataset.special_tokens
        self.special_token_ids = base_dataset.special_token_ids

        # Load conversational dataset
        self.dataset = load_dataset(
            repo_id,
            split=split,
            streaming=True,
            cache_dir=cache_dir,
        ).shuffle(seed=seed).shard(num_shards=world_size, index=rank)

    def _format_conversation(self, example: Dict) -> List[int]:
        """
        Format conversation with thinking tokens.

        Expected format in dataset:
        {
            "messages": [
                {"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "...", "thinking": "..."}
            ]
        }
        """
        tokens = []

        messages = example.get("messages", [])

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            thinking = msg.get("thinking", "")

            # Add role token
            if role == "system":
                tokens.append(self.special_token_ids["system"])
            elif role == "user":
                tokens.append(self.special_token_ids["user"])
            elif role == "assistant":
                tokens.append(self.special_token_ids["assistant"])

            # Add thinking tokens if present (before main content)
            if thinking:
                tokens.append(self.special_token_ids["thinking_start"])
                tokens.extend(self.tokenizer.encode(thinking))
                tokens.append(self.special_token_ids["thinking_end"])

            # Add content
            tokens.extend(self.tokenizer.encode(content))

        # Add end of text
        tokens.append(self.special_token_ids["endoftext"])

        return tokens

    def __iter__(self):
        """
        Iterate over conversations.
        """
        for example in self.dataset:
            tokens = self._format_conversation(example)

            # Truncate or pad to seq_length
            if len(tokens) > self.seq_length + 1:
                tokens = tokens[:self.seq_length + 1]
            elif len(tokens) < self.seq_length + 1:
                # Pad with endoftext
                pad_length = self.seq_length + 1 - len(tokens)
                tokens.extend([self.special_token_ids["endoftext"]] * pad_length)

            # Create input and labels
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            labels = torch.tensor(tokens[1:], dtype=torch.long)

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }


def create_dataloader(
    repo_id: str,
    batch_size: int = 8,
    seq_length: int = 4096,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    conversational: bool = False,
):
    """
    Create dataloader for training.

    Args:
        repo_id: HuggingFace dataset repo
        batch_size: Batch size per GPU
        seq_length: Sequence length (4096 for base, 8192 for conversation)
        num_workers: Number of dataloader workers
        rank: GPU rank
        world_size: Total GPUs
        conversational: Use conversational format (for fine-tuning)

    Returns:
        DataLoader instance
    """
    if conversational:
        dataset = ConversationalDataset(
            repo_id=repo_id,
            seq_length=seq_length,
            rank=rank,
            world_size=world_size,
        )
    else:
        dataset = StreamingTokenizedDataset(
            repo_id=repo_id,
            seq_length=seq_length,
            rank=rank,
            world_size=world_size,
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    return dataloader
