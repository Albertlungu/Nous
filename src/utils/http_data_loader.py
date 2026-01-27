"""
HTTP Data Loader for fetching training data from a remote server.
Allows training on remote GPUs without storing the full dataset locally.
"""

import pickle
import requests
from typing import List, Optional
import time


class HTTPDataLoader:
    """
    Loads training data from an HTTP server in chunks.
    """

    def __init__(self, server_url: str, cache_chunks: int = 3):
        """
        Initialize HTTP data loader.

        Args:
            server_url: Base URL of the HTTP data server (e.g., 'http://192.168.1.100:8000')
            cache_chunks: Number of chunks to keep in memory (default: 3)
        """
        self.server_url = server_url.rstrip('/')
        self.cache_chunks = cache_chunks
        self.chunk_cache = {}
        self.cache_order = []

        # Fetch dataset info
        self.info = self._fetch_info()
        self.total_examples = self.info['total_examples']
        self.chunk_size = self.info['chunk_size']
        self.total_chunks = self.info['total_chunks']

        print(f"Connected to HTTP data server at {self.server_url}")
        print(f"Total examples: {self.total_examples:,}")
        print(f"Chunk size: {self.chunk_size}")
        print(f"Total chunks: {self.total_chunks}")

    def _fetch_info(self) -> dict:
        """Fetch dataset information from server."""
        response = requests.get(f"{self.server_url}/info", timeout=10)
        response.raise_for_status()
        return response.json()

    def _fetch_chunk(self, chunk_id: int) -> List:
        """
        Fetch a chunk from the server.

        Args:
            chunk_id: The chunk index to fetch

        Returns:
            List of token_ids for the chunk
        """
        response = requests.get(
            f"{self.server_url}/chunk/{chunk_id}",
            timeout=30
        )
        response.raise_for_status()

        # Deserialize the chunk
        chunk_data = pickle.loads(response.content)
        return chunk_data

    def get_chunk(self, chunk_id: int) -> List:
        """
        Get a chunk, using cache if available.

        Args:
            chunk_id: The chunk index to fetch

        Returns:
            List of token_ids for the chunk
        """
        if chunk_id in self.chunk_cache:
            return self.chunk_cache[chunk_id]

        # Fetch from server
        print(f"Fetching chunk {chunk_id}/{self.total_chunks}...")
        chunk_data = self._fetch_chunk(chunk_id)

        # Add to cache
        self.chunk_cache[chunk_id] = chunk_data
        self.cache_order.append(chunk_id)

        # Evict oldest chunks if cache is full
        while len(self.cache_order) > self.cache_chunks:
            oldest_chunk = self.cache_order.pop(0)
            if oldest_chunk in self.chunk_cache:
                del self.chunk_cache[oldest_chunk]

        return chunk_data

    def get_all_data(self) -> List:
        """
        Fetch all training data by iterating through all chunks.
        Warning: This loads the entire dataset into memory!

        Returns:
            Complete list of all token_ids
        """
        print("Warning: Fetching all data will load the entire dataset into memory!")
        all_data = []

        for chunk_id in range(self.total_chunks):
            chunk = self.get_chunk(chunk_id)
            all_data.extend(chunk)
            print(f"Loaded chunk {chunk_id + 1}/{self.total_chunks} ({len(all_data):,} examples so far)")

        return all_data

    def stream_chunks(self):
        """
        Generator that yields chunks one at a time.
        Useful for streaming training.

        Yields:
            Chunk data (list of token_ids)
        """
        for chunk_id in range(self.total_chunks):
            yield self.get_chunk(chunk_id)

    def fetch_batch(self, batch_id: int, batch_size: int) -> List:
        """
        Fetch a specific batch directly from server.
        More granular than chunks.

        Args:
            batch_id: The batch index
            batch_size: Number of examples per batch

        Returns:
            List of token_ids for the batch
        """
        response = requests.get(
            f"{self.server_url}/batch/{batch_id}/{batch_size}",
            timeout=30
        )
        response.raise_for_status()

        batch_data = pickle.loads(response.content)
        return batch_data

    def health_check(self) -> bool:
        """
        Check if the server is healthy.

        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


def load_data_from_http(server_url: str, load_all: bool = False, cache_chunks: int = 3):
    """
    Load training data from HTTP server.

    Args:
        server_url: Base URL of the HTTP data server
        load_all: If True, load all data into memory at once. If False, return loader for streaming.
        cache_chunks: Number of chunks to cache in memory

    Returns:
        If load_all=True: List of all token_ids
        If load_all=False: HTTPDataLoader instance for streaming
    """
    loader = HTTPDataLoader(server_url, cache_chunks=cache_chunks)

    if load_all:
        return loader.get_all_data()
    else:
        return loader
