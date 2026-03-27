"""
./src/data/streaming_loader.py

Streams data loader for HuggingFace datasets.
Downloads compressed text, tokenizes on the fly, yields batches.
"""

import os
from typing import Iterator, Optional

import numpy as np
import tiktoken
import zstandard as zstd
from huggingface_hub import hf_hub_download
from tqdm import tqdm


class JAXStreamingLoader:
    def __init__(
        self,
        filename: str,
        repo_id: str = "albertlungu/final-nous-corpus",
        tokenizer_name: str = "cl100k_base",
        batch_size: int = 32,
        seq_length: int = 256,
        cache_dir: Optional[str] = "/tmp/hf_streaming_cache",
        buffer_size_mb: int = 100,
    ) -> None:
        """
        Initialize the streaming loader

        Args:
            filename (str): Filename in repo (e.g. "final_nous_corpus.txt.zst")
            repo_id (str, optional): Huggingface repo ID. Defaults to "albertlungu/final_nous_corpus".
            tokenizer_name (str, optional): Tiktoken tokenizer name. Defaults to "cl100k_base".
            batch_size (int, optional): Number of sequences per batch. Defaults to 32.
            seq_length (int, optional): Length of each sequence (must match model's max_seq_length). Defaults to 256.
            cache_dir (Optional[str], optional): Cache directory for downloaded chunks. Defaults to "/tmp/hf_streaming_cache".
            buffer_size_mb (int, optional): Size of decompression buffer in MB. Defaults to 100.
        """
        self.repo_id = repo_id
        self.filename = filename
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.cache_dir = cache_dir
        self.buffer_size = buffer_size_mb * 1024

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.tokenizer.n_vocab

        self.eos_token_id = self.tokenizer.eot_token

        self.token_buffer = []

        print(f"Downloading {filename} from {repo_id} ...")
        self.file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        print(f"File cached at {self.file_path}")

        def _stream_decompress(self) -> Iterator[str]:
            dctx = zstd.ZstdDecompressor()

            with open(self.file_path, "rb") as compressed_file:
                with dctx.stream_reader(compressed_file) as reader:
                    text_buffer = ""

                    while True:
                        chunk = reader.read(self.buffer_size)
                        if not chunk:
                            break

                        try:
                            text_buffer += chunk.decode("utf-8")
                        except UnicodeDecodeError:
                            continue

                        lines = text_buffer.split("\n")
                        text_buffer = lines[-1]

                        for line in lines[:-1]:
                            if line.strip():
                                yield line

                    if text_buffer.strip():
                        yield text_buffer

        def _tokenize_text(self, text: str) -> list[int]:
            """
            Tokenize using tiktoken

            Args:
                text (str): Raw text strings

            Returns:
                list[int]: List of token ids
            """
            tokens = self.tokenizer.encode(text)
            tokens.append(self.eos_token_id)
            return tokens

        def stream_batches(self) -> Iterator[np.ndarray]:
            """
            Stream batches from HF dataset.

            Yields:
                Iterator[np.ndarray]: Batch of shape (batch_size, seq_length)
            """
            batch = []
            print("Starting streaming")

            for text_chunk in self._stream_decompress():
                tokens = self._tokenize_text(text_chunk)
                self.token_buffer.extend(tokens)

                while len(self.token_buffer) >= self.seq_length:
                    sequence = self.token_buffer[: self.seq_length]
                    self.token_buffer = self.token_buffer[self.seq_length :]

                    batch.append(sequence)

                    if len(batch) == self.batch_size:
                        yield np.array(batch, dtype=np.int32)
                        batch = []
            if batch:
                while len(batch) < self.batch_size:
                    batch.append([0] * self.seq_length)
                yield np.array(batch, dtype=np.int32)

        def estimate_total_batches(
            self, compressed_size_gb: float, compression_ratio: float = 2.94
        ) -> int:
            """
            Estimate total number of batches in a dataset.

            Args:
                compressed_size_gb (float): Size of compressed file in GB
                compression_ratio (float, optional): Compression ratio (original size / compressed size). Defaults to 2.94.

            Returns:
                int: Estimated number of batches
            """
            original_size_gb = compressed_size_gb * compression_ratio

            chars_per_gb = 1024**3
            total_chars = original_size_gb * chars_per_gb
            total_tokens = total_chars / 4

            total_sequences = int(total_tokens / self.seq_length)
            total_batches = total_sequences // self.batch_size

            return total_batches


class InfiniteStreamingLoader:
    """
    Infinite streaming loader that repeats the dataset infinitely.
    """

    def __init__(self, base_loader: JAXStreamingLoader):
        """
        Init

        Args:
            base_loader (JAXStreamingLoader): The instance to wrap
        """
        self.base_loader = base_loader

    def stream_batches(self) -> Iterator[np.ndarray]:
        """
        Stream batches indefinitely (repeats dataset)
        """
        while True:
            self.base_loader.token_buffer = []
            for batch in self.base_loader.stream_batches():
                yield batch
