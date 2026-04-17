"""
JAX-compatible streaming data loader for HuggingFace datasets.
Downloads compressed text, tokenizes on-the-fly, yields batches.
"""

import codecs
import os
import random
from typing import Iterator, Optional

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

        # Token buffer as numpy ring buffer for efficient O(1) operations
        # Pre-allocate buffer that can hold multiple sequences
        buffer_capacity = max(seq_length * batch_size * 2, 100000)
        self.token_buffer = np.zeros(buffer_capacity, dtype=np.int32)
        self.buffer_start = 0
        self.buffer_end = 0
        self.buffer_capacity = buffer_capacity

        # Set up HuggingFace filesystem for streaming (no download)
        print(f"Setting up streaming from {repo_id}/{filename}...")
        self.hffs = HfFileSystem()
        self.file_path = f"datasets/{repo_id}/{filename}"
        print(f"Will stream from HuggingFace: {self.file_path}")

    def _buffer_size(self) -> int:
        """Get current number of tokens in buffer."""
        return (self.buffer_end - self.buffer_start) % self.buffer_capacity

    def _buffer_extend(self, tokens: list[int]) -> None:
        """Add tokens to ring buffer."""
        tokens_array = np.array(tokens, dtype=np.int32)
        num_tokens = len(tokens_array)

        # Check if buffer needs expansion
        if self._buffer_size() + num_tokens >= self.buffer_capacity:
            # Double buffer size and copy existing data
            new_capacity = self.buffer_capacity * 2
            new_buffer = np.zeros(new_capacity, dtype=np.int32)

            # Copy existing tokens to new buffer
            current_size = self._buffer_size()
            if current_size > 0:
                if self.buffer_end > self.buffer_start:
                    new_buffer[:current_size] = self.token_buffer[
                        self.buffer_start : self.buffer_end
                    ]
                else:
                    # Wrapped around
                    first_part = self.buffer_capacity - self.buffer_start
                    new_buffer[:first_part] = self.token_buffer[self.buffer_start :]
                    new_buffer[first_part:current_size] = self.token_buffer[
                        : self.buffer_end
                    ]

            self.token_buffer = new_buffer
            self.buffer_start = 0
            self.buffer_end = current_size
            self.buffer_capacity = new_capacity

        # Add new tokens
        for i, token in enumerate(tokens_array):
            self.token_buffer[self.buffer_end] = token
            self.buffer_end = (self.buffer_end + 1) % self.buffer_capacity

    def _buffer_extract(self, n: int) -> np.ndarray:
        """Extract n tokens from buffer and remove them."""
        if n > self._buffer_size():
            raise ValueError(
                f"Cannot extract {n} tokens, only {self._buffer_size()} available"
            )

        result = np.zeros(n, dtype=np.int32)

        # Extract tokens
        if self.buffer_start + n <= self.buffer_capacity:
            # No wrap-around
            result[:] = self.token_buffer[self.buffer_start : self.buffer_start + n]
        else:
            # Wrap-around case
            first_part = self.buffer_capacity - self.buffer_start
            result[:first_part] = self.token_buffer[self.buffer_start :]
            result[first_part:] = self.token_buffer[: n - first_part]

        # Update start pointer
        self.buffer_start = (self.buffer_start + n) % self.buffer_capacity

        return result

    def _stream_decompress(self) -> Iterator[str]:
        """
        Stream and decompress the .zst file directly from HuggingFace.
        Yields text lines (each gets EOS token for boundary learning).
        Uses HfFileSystem for efficient streaming with retry support.
        """
        print("Starting stream from HuggingFace...")

        max_retries = 5
        retry_count = 0
        lines_emitted = 0

        while retry_count < max_retries:
            try:
                # Recreate filesystem connection for each attempt
                self.hffs = HfFileSystem()

                # Open file stream from HuggingFace (binary mode for zstd)
                with self.hffs.open(self.file_path, "rb") as remote_file:
                    lines_to_skip = lines_emitted
                    if retry_count > 0 and lines_emitted > 0:
                        print(
                            f"Reconnecting and skipping {lines_emitted:,} completed lines..."
                        )

                    # Decompress streaming data
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(remote_file) as reader:
                        decoder = codecs.getincrementaldecoder("utf-8")()
                        text_buffer = ""
                        skipped_lines = 0

                        while True:
                            try:
                                # Read decompressed chunk
                                chunk = reader.read(self.buffer_size)
                                if not chunk:
                                    text_buffer += decoder.decode(b"", final=True)
                                    break

                                # Incremental decoder preserves partial UTF-8 sequences across chunks.
                                text_buffer += decoder.decode(chunk, final=False)

                                # Split on newlines to get lines
                                # Each line gets an EOS token for frequent boundary signals
                                lines = text_buffer.split("\n")
                                text_buffer = lines[
                                    -1
                                ]  # Keep incomplete line in buffer

                                for line in lines[:-1]:
                                    if not line.strip():
                                        continue

                                    if skipped_lines < lines_to_skip:
                                        skipped_lines += 1
                                        continue

                                    yield line
                                    lines_emitted += 1

                            except (TimeoutError, ConnectionError, OSError) as e:
                                print(f"Connection error during read: {e}")
                                print(
                                    "Will retry stream from start and skip previously yielded lines"
                                )
                                raise  # Re-raise to trigger outer retry logic

                        # Yield remaining buffer
                        if text_buffer.strip():
                            if skipped_lines < lines_to_skip:
                                skipped_lines += 1
                            else:
                                yield text_buffer
                                lines_emitted += 1

                        print("Stream completed successfully")

                        # If we got here, stream completed successfully
                        return

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Failed after {max_retries} retries. Last error: {e}")
                    raise

                import time

                wait_time = 2**retry_count  # Exponential backoff
                print(f"Stream error: {e}")
                print(f"Retry {retry_count}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)

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

            # Add to buffer using ring buffer
            self._buffer_extend(tokens)

            # Create sequences while buffer has enough tokens
            while self._buffer_size() >= self.seq_length:
                # Extract sequence from buffer (removes tokens automatically)
                sequence = self._buffer_extract(self.seq_length)

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
                batch.append(np.zeros(self.seq_length, dtype=np.int32))
            yield np.array(batch, dtype=np.int32)

    def _stream_batches_shuffled(self) -> Iterator[np.ndarray]:
        """Stream batches with document-level shuffling using a buffer."""
        print(
            f"Starting streaming from HuggingFace with shuffle (buffer size: {self.shuffle_buffer_size})..."
        )

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
                    self._buffer_extend(tokens)

                    # Create sequences
                    while self._buffer_size() >= self.seq_length:
                        # Extract sequence from buffer
                        sequence = self._buffer_extract(self.seq_length)

                        batch.append(sequence)

                        if len(batch) == self.batch_size:
                            yield np.array(batch, dtype=np.int32)
                            batch = []

        # Process remaining buffer
        random.shuffle(document_buffer)
        for doc in document_buffer:
            tokens = self._tokenize_text(doc)
            self._buffer_extend(tokens)

            while self._buffer_size() >= self.seq_length:
                # Extract sequence from buffer
                sequence = self._buffer_extract(self.seq_length)
                batch.append(sequence)

                if len(batch) == self.batch_size:
                    yield np.array(batch, dtype=np.int32)
                    batch = []

        # Yield final partial batch
        if batch:
            while len(batch) < self.batch_size:
                batch.append(np.zeros(self.seq_length, dtype=np.int32))
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
            self.base_loader.buffer_start = 0
            self.base_loader.buffer_end = 0

            for batch in self.base_loader.stream_batches():
                yield batch
