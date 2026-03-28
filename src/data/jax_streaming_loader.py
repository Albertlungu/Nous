"""
JAX-compatible streaming data loader for HuggingFace datasets.
Downloads compressed text, tokenizes on-the-fly, yields batches.
"""

import os
from typing import Iterator, Optional
import random
from collections import deque
from itertools import islice

import numpy as np
import tiktoken
import zstandard as zstd
from huggingface_hub import HfFileSystem
from tqdm import tqdm


class JAXStreamingLoader:
    """
    Streams tokenized batches from HuggingFace Hub for JAX training.
    Downloads compressed corpus in chunks, tokenizes on-the-fly.
    """

    def __init__(
        self,
        repo_id: str,
        filename: str,
        tokenizer_name: str = "cl100k_base",
        batch_size: int = 32,
        seq_length: int = 256,
        cache_dir: Optional[str] = "/tmp/hf_streaming_cache",
        buffer_size_mb: int = 100,
        shuffle: bool = False,
        shuffle_buffer_size: int = 10000,
    ):
        """
        Initialize streaming loader.

        Args:
            repo_id: HuggingFace repo ID (e.g., "albertlungu/final-nous-corpus")
            filename: Filename in repo (e.g., "corpus.txt.zst")
            tokenizer_name: TikToken tokenizer name (default: "cl100k_base")
            batch_size: Number of sequences per batch
            seq_length: Length of each sequence (must match model's max_seq_length)
            cache_dir: Local cache directory for downloaded chunks
            buffer_size_mb: Size of decompression buffer in MB
            shuffle: Whether to shuffle documents using a buffer
            shuffle_buffer_size: Number of documents to keep in shuffle buffer
        """
        self.repo_id = repo_id
        self.filename = filename
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.cache_dir = cache_dir
        self.buffer_size = buffer_size_mb * 1024 * 1024
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        # Create cache directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.tokenizer.n_vocab

        # Add special tokens (matching your TikToken wrapper)
        self.eos_token_id = self.tokenizer.eot_token

        # Token buffer for accumulating tokens across documents (using deque for efficiency)
        self.token_buffer = deque()

        # Set up HuggingFace filesystem for streaming (no download)
        print(f"Setting up streaming from {repo_id}/{filename}...")
        self.hffs = HfFileSystem()
        self.file_path = f"datasets/{repo_id}/{filename}"
        print(f"Will stream from HuggingFace: {self.file_path}")

    def _stream_decompress(self) -> Iterator[str]:
        """
        Stream and decompress the .zst file directly from HuggingFace.
        Yields text chunks without downloading entire file to disk.
        Uses HfFileSystem for efficient streaming.
        """
        print("Starting stream from HuggingFace...")

        # Open file stream from HuggingFace (binary mode for zstd)
        with self.hffs.open(self.file_path, "rb") as remote_file:
            # Decompress streaming data
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(remote_file) as reader:
                text_buffer = ""

                while True:
                    # Read decompressed chunk
                    chunk = reader.read(self.buffer_size)
                    if not chunk:
                        break

                    # Decode bytes to text
                    try:
                        text_buffer += chunk.decode("utf-8")
                    except UnicodeDecodeError:
                        # Handle partial UTF-8 sequences at chunk boundaries
                        continue

                    # Yield complete lines (split on newlines to avoid mid-word breaks)
                    lines = text_buffer.split("\n")
                    text_buffer = lines[-1]  # Keep incomplete line in buffer

                    for line in lines[:-1]:
                        if line.strip():  # Skip empty lines
                            yield line

                # Yield remaining buffer
                if text_buffer.strip():
                    yield text_buffer

    def _tokenize_text(self, text: str) -> list[int]:
        """
        Tokenize text using tiktoken.

        Args:
            text: Raw text string

        Returns:
            List of token IDs
        """
        # Allow special tokens to be encoded without raising errors
        tokens = self.tokenizer.encode(text, allowed_special="all")
        tokens.append(self.eos_token_id)
        return tokens

    def stream_batches(self) -> Iterator[np.ndarray]:
        """
        Stream batches from HuggingFace dataset.
        If shuffle=True, uses a document buffer to mix documents.

        Yields:
            np.ndarray: Batch of shape (batch_size, seq_length)
        """
        if self.shuffle:
            yield from self._stream_batches_shuffled()
        else:
            yield from self._stream_batches_sequential()

    def _stream_batches_sequential(self) -> Iterator[np.ndarray]:
        """Stream batches sequentially without shuffling."""
        batch = []

        print("Starting streaming from HuggingFace...")

        for text_chunk in self._stream_decompress():
            # Tokenize the text chunk
            tokens = self._tokenize_text(text_chunk)

            # Add to buffer
            self.token_buffer.extend(tokens)

            # Create sequences while buffer has enough tokens
            while len(self.token_buffer) >= self.seq_length:
                # Extract sequence from deque (more efficient than list slicing)
                sequence = list(islice(self.token_buffer, self.seq_length))
                # Remove consumed tokens from front of deque
                for _ in range(self.seq_length):
                    self.token_buffer.popleft()

                batch.append(sequence)

                # Yield batch when full
                if len(batch) == self.batch_size:
                    yield np.array(batch, dtype=np.int32)
                    batch = []

        # Yield final partial batch if exists
        if batch:
            # Pad final batch to batch_size
            while len(batch) < self.batch_size:
                # Pad with zeros (will be masked during loss calculation)
                batch.append([0] * self.seq_length)
            yield np.array(batch, dtype=np.int32)

    def _stream_batches_shuffled(self) -> Iterator[np.ndarray]:
        """Stream batches with document-level shuffling using a buffer."""
        print(f"Starting streaming from HuggingFace with shuffle (buffer size: {self.shuffle_buffer_size})...")

        document_buffer = []
        batch = []

        for text_chunk in self._stream_decompress():
            # Add document to shuffle buffer
            document_buffer.append(text_chunk)

            # Once buffer is full, start yielding shuffled documents
            if len(document_buffer) >= self.shuffle_buffer_size:
                # Shuffle buffer
                random.shuffle(document_buffer)

                # Process half the buffer (keep refilling for continuous shuffling)
                docs_to_process = document_buffer[: self.shuffle_buffer_size // 2]
                document_buffer = document_buffer[self.shuffle_buffer_size // 2 :]

                # Tokenize and create sequences from shuffled docs
                for doc in docs_to_process:
                    tokens = self._tokenize_text(doc)
                    self.token_buffer.extend(tokens)

                    # Create sequences
                    while len(self.token_buffer) >= self.seq_length:
                        # Extract sequence from deque
                        sequence = list(islice(self.token_buffer, self.seq_length))
                        # Remove consumed tokens
                        for _ in range(self.seq_length):
                            self.token_buffer.popleft()

                        batch.append(sequence)

                        if len(batch) == self.batch_size:
                            yield np.array(batch, dtype=np.int32)
                            batch = []

        # Process remaining buffer
        random.shuffle(document_buffer)
        for doc in document_buffer:
            tokens = self._tokenize_text(doc)
            self.token_buffer.extend(tokens)

            while len(self.token_buffer) >= self.seq_length:
                # Extract sequence from deque
                sequence = list(islice(self.token_buffer, self.seq_length))
                # Remove consumed tokens
                for _ in range(self.seq_length):
                    self.token_buffer.popleft()
                batch.append(sequence)

                if len(batch) == self.batch_size:
                    yield np.array(batch, dtype=np.int32)
                    batch = []

        # Yield final partial batch
        if batch:
            while len(batch) < self.batch_size:
                batch.append([0] * self.seq_length)
            yield np.array(batch, dtype=np.int32)

    def estimate_total_batches(
        self, compressed_size_gb: float, compression_ratio: float = 2.94
    ) -> int:
        """
        Estimate total number of batches in dataset.

        Args:
            compressed_size_gb: Size of compressed file in GB
            compression_ratio: Compression ratio (original_size / compressed_size)

        Returns:
            Estimated number of batches
        """
        # Estimate original size
        original_size_gb = compressed_size_gb * compression_ratio

        # Estimate total tokens (assuming ~4 chars per token for English text)
        chars_per_gb = 1024 * 1024 * 1024
        total_chars = original_size_gb * chars_per_gb
        total_tokens = total_chars / 4

        # Calculate sequences and batches
        total_sequences = int(total_tokens / self.seq_length)
        total_batches = total_sequences // self.batch_size

        return total_batches


class InfiniteStreamingLoader:
    """
    Infinite streaming loader that repeats the dataset indefinitely.
    Useful for training where you want to control epochs manually.
    """

    def __init__(self, base_loader: JAXStreamingLoader):
        """
        Args:
            base_loader: JAXStreamingLoader instance to wrap
        """
        self.base_loader = base_loader

    def stream_batches(self) -> Iterator[np.ndarray]:
        """
        Stream batches infinitely (repeats dataset).
        """
        while True:
            # Reset token buffer at start of each epoch
            self.base_loader.token_buffer = []

            for batch in self.base_loader.stream_batches():
                yield batch
