"""
HTTP server for serving training data in chunks.
This server allows remote GPUs to fetch training data over HTTP without storing the full dataset locally.

Usage:
    python src/utils/http_data_server.py --data-path /path/to/data.pkl --port 8000 --chunk-size 1000
"""

import pickle
import argparse
from flask import Flask, jsonify, send_file
import io

app = Flask(__name__)

# Global variables set by CLI args
DATA = None
CHUNK_SIZE = 1000
TOTAL_EXAMPLES = 0


@app.route('/info', methods=['GET'])
def get_info():
    """
    Get dataset information (total examples, chunk size).
    """
    return jsonify({
        'total_examples': TOTAL_EXAMPLES,
        'chunk_size': CHUNK_SIZE,
        'total_chunks': (TOTAL_EXAMPLES + CHUNK_SIZE - 1) // CHUNK_SIZE
    })


@app.route('/chunk/<int:chunk_id>', methods=['GET'])
def get_chunk(chunk_id):
    """
    Get a specific chunk of training data.

    Args:
        chunk_id: The chunk index to fetch (0-indexed)

    Returns:
        Pickled list of token_ids for the requested chunk
    """
    start_idx = chunk_id * CHUNK_SIZE
    end_idx = min(start_idx + CHUNK_SIZE, TOTAL_EXAMPLES)

    if start_idx >= TOTAL_EXAMPLES:
        return jsonify({'error': 'Chunk ID out of range'}), 404

    chunk_data = DATA[start_idx:end_idx]

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
    More granular than chunks - useful for on-demand batch fetching.

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

    batch_data = DATA[start_idx:end_idx]

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
    return jsonify({'status': 'healthy', 'data_loaded': DATA_LOADED})


def main():
    parser = argparse.ArgumentParser(description='HTTP server for serving training data')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to the pickled training data file (optional - can upload via /upload endpoint)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run the server on (default: 8000)')
    parser.add_argument('--chunk-size', type=int, default=1000,
                        help='Number of examples per chunk (default: 1000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')

    args = parser.parse_args()

    global DATA, CHUNK_SIZE, TOTAL_EXAMPLES, DATA_LOADED

    CHUNK_SIZE = args.chunk_size

    # Optionally pre-load data if path is provided
    if args.data_path:
        print(f"Loading training data from {args.data_path}...")
        with open(args.data_path, 'rb') as f:
            DATA = pickle.load(f)

        TOTAL_EXAMPLES = len(DATA)
        DATA_LOADED = True

        print(f"Loaded {TOTAL_EXAMPLES:,} training examples")
        print(f"Chunk size: {CHUNK_SIZE}")
        print(f"Total chunks: {(TOTAL_EXAMPLES + CHUNK_SIZE - 1) // CHUNK_SIZE}")
    else:
        print("No data pre-loaded. Waiting for file upload via /upload endpoint.")

    print(f"\nStarting HTTP server on {args.host}:{args.port}")
    print(f"Endpoints:")
    print(f"  - POST /upload - Upload a .pkl data file")
    print(f"  - GET /info - Dataset information")
    print(f"  - GET /chunk/<chunk_id> - Fetch a chunk of data")
    print(f"  - GET /batch/<batch_id>/<batch_size> - Fetch a specific batch")
    print(f"  - GET /health - Health check")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
