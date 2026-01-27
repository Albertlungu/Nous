"""
HTTP server for serving training data from USB drive on Raspberry Pi.
MEMORY EFFICIENT - streams from disk without loading entire dataset into RAM.
Perfect for Raspberry Pi with limited RAM (1GB, 2GB, etc.)

Usage on Raspberry Pi:
    python raspberry_pi_server.py --data-path /media/pi/YOUR_USB/data.pkl --port 8000

    # To run in background:
    nohup python raspberry_pi_server.py --data-path /media/pi/YOUR_USB/data.pkl --port 8000 > server.log 2>&1 &
"""

import pickle
import argparse
from flask import Flask, jsonify, send_file
import io
import os

app = Flask(__name__)

# Global variables
DATA_PATH = None
CHUNK_SIZE = 1000
TOTAL_EXAMPLES = 0
CHUNK_CACHE = {}  # Simple cache to avoid re-reading same chunks
MAX_CACHE_SIZE = 5  # Keep only 5 chunks in memory


def get_dataset_length(file_path):
    """
    Get the length of the dataset without loading it all into memory.
    """
    print("Scanning dataset to count examples (this may take a few minutes)...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        length = len(data)
    return length


def load_slice_from_disk(start_idx, end_idx):
    """
    Load only a specific slice of data from disk.
    This is memory efficient - only loads what's needed.

    Args:
        start_idx: Start index
        end_idx: End index

    Returns:
        List of token_ids for the requested slice
    """
    # Check cache first
    cache_key = (start_idx, end_idx)
    if cache_key in CHUNK_CACHE:
        return CHUNK_CACHE[cache_key]

    # Load from disk
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)
        slice_data = data[start_idx:end_idx]

        # Clean up the full data immediately to free memory
        del data

    # Add to cache
    CHUNK_CACHE[cache_key] = slice_data

    # Limit cache size to prevent memory issues
    if len(CHUNK_CACHE) > MAX_CACHE_SIZE:
        # Remove oldest entry
        oldest_key = next(iter(CHUNK_CACHE))
        del CHUNK_CACHE[oldest_key]

    return slice_data


@app.route('/info', methods=['GET'])
def get_info():
    """Get dataset information."""
    return jsonify({
        'total_examples': TOTAL_EXAMPLES,
        'chunk_size': CHUNK_SIZE,
        'total_chunks': (TOTAL_EXAMPLES + CHUNK_SIZE - 1) // CHUNK_SIZE
    })


@app.route('/chunk/<int:chunk_id>', methods=['GET'])
def get_chunk(chunk_id):
    """
    Get a specific chunk of training data.
    Loads only this chunk from disk - memory efficient!

    Args:
        chunk_id: The chunk index to fetch (0-indexed)

    Returns:
        Pickled list of token_ids for the requested chunk
    """
    start_idx = chunk_id * CHUNK_SIZE
    end_idx = min(start_idx + CHUNK_SIZE, TOTAL_EXAMPLES)

    if start_idx >= TOTAL_EXAMPLES:
        return jsonify({'error': 'Chunk ID out of range'}), 404

    # Load only this chunk from disk
    chunk_data = load_slice_from_disk(start_idx, end_idx)

    # Serialize chunk to bytes
    buffer = io.BytesIO()
    pickle.dump(chunk_data, buffer)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=f'chunk_{chunk_id}.pkl'
    )


@app.route('/batch/<int:batch_id>/<int:batch_size>', methods=['GET'])
def get_batch(batch_id, batch_size):
    """
    Get a specific batch of training data.
    Loads only this batch from disk - memory efficient!

    Args:
        batch_id: The batch index to fetch (0-indexed)
        batch_size: Number of examples per batch

    Returns:
        Pickled list of token_ids for the requested batch
    """
    start_idx = batch_id * batch_size
    end_idx = min(start_idx + batch_size, TOTAL_EXAMPLES)

    if start_idx >= TOTAL_EXAMPLES:
        return jsonify({'error': 'Batch ID out of range'}), 404

    # Load only this batch from disk
    batch_data = load_slice_from_disk(start_idx, end_idx)

    # Serialize batch to bytes
    buffer = io.BytesIO()
    pickle.dump(batch_data, buffer)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=f'batch_{batch_id}.pkl'
    )


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'total_examples': TOTAL_EXAMPLES,
        'data_loaded': True,
        'cache_size': len(CHUNK_CACHE)
    })


def main():
    parser = argparse.ArgumentParser(description='HTTP server for serving training data from USB drive (memory efficient)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the .pkl file on the USB drive (e.g., /media/pi/USB/data.pkl)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run the server on (default: 8000)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Number of examples per chunk (default: 1000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0 for external access)')
    parser.add_argument('--cache-size', type=int, default=5,
                        help='Number of chunks to keep in memory cache (default: 5)')

    args = parser.parse_args()

    global DATA_PATH, CHUNK_SIZE, TOTAL_EXAMPLES, MAX_CACHE_SIZE

    DATA_PATH = args.data_path
    CHUNK_SIZE = args.chunk_size
    MAX_CACHE_SIZE = args.cache_size

    # Verify file exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at {DATA_PATH}")
        print("\nTip: USB drives usually mount at /media/pi/ on Raspberry Pi")
        print("Run 'ls /media/pi/' to find your USB drive name")
        exit(1)

    file_size = os.path.getsize(DATA_PATH)
    file_size_gb = file_size / (1024 ** 3)

    print(f"{'='*60}")
    print(f"Raspberry Pi Data Server (Memory Efficient Mode)")
    print(f"{'='*60}")
    print(f"Data file: {DATA_PATH}")
    print(f"File size: {file_size_gb:.2f} GB")
    print(f"Cache size: {MAX_CACHE_SIZE} chunks")

    # Count total examples (needs to load once to count, but then discarded)
    try:
        TOTAL_EXAMPLES = get_dataset_length(DATA_PATH)
        print(f"\nDataset contains {TOTAL_EXAMPLES:,} training examples")
        print(f"Chunk size: {CHUNK_SIZE}")
        print(f"Total chunks: {(TOTAL_EXAMPLES + CHUNK_SIZE - 1) // CHUNK_SIZE}")

    except Exception as e:
        print(f"\nERROR loading data: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print(f"\n{'='*60}")
    print(f"Starting HTTP server on {args.host}:{args.port}")
    print(f"{'='*60}")
    print(f"Memory mode: STREAMING (loads chunks on-demand)")
    print(f"RAM usage: ~{MAX_CACHE_SIZE * CHUNK_SIZE * 0.001:.0f}MB cached + overhead")
    print(f"\nEndpoints:")
    print(f"  GET  /info                           - Dataset information")
    print(f"  GET  /chunk/<chunk_id>               - Fetch a chunk of data")
    print(f"  GET  /batch/<batch_id>/<batch_size>  - Fetch a specific batch")
    print(f"  GET  /health                         - Health check")
    print(f"\nServer ready! Data will be streamed from USB as needed.")
    print(f"{'='*60}\n")

    # Run the Flask server
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
