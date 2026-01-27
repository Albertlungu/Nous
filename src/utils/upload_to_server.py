"""
Upload training data to an HTTP data server.

Usage:
    python src/utils/upload_to_server.py --file /path/to/data.pkl --server http://192.168.1.100:8000
"""

import argparse
import requests
import os
from tqdm import tqdm


class TqdmUpload(object):
    """
    Wrapper to track upload progress with tqdm.
    """
    def __init__(self, filename, total_size):
        self.filename = filename
        self.total_size = total_size
        self.progress = tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Uploading {os.path.basename(filename)}"
        )
        self.uploaded = 0

    def __call__(self, monitor):
        # Update progress bar
        self.progress.update(monitor.bytes_read - self.uploaded)
        self.uploaded = monitor.bytes_read

    def close(self):
        self.progress.close()


def upload_file(file_path, server_url):
    """
    Upload a pickle file to the HTTP data server.

    Args:
        file_path: Path to the .pkl file to upload
        server_url: Base URL of the server (e.g., 'http://192.168.1.100:8000')
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False

    if not file_path.endswith('.pkl'):
        print("Error: Only .pkl files are supported")
        return False

    file_size = os.path.getsize(file_path)
    file_size_gb = file_size / (1024 ** 3)

    print(f"File: {file_path}")
    print(f"Size: {file_size_gb:.2f} GB ({file_size:,} bytes)")
    print(f"Server: {server_url}")
    print()

    # Check server health
    try:
        print("Checking server health...")
        health_response = requests.get(f"{server_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"Error: Server health check failed with status {health_response.status_code}")
            return False
        print("Server is healthy!")
    except Exception as e:
        print(f"Error: Cannot connect to server: {e}")
        return False

    # Upload file with progress bar
    try:
        print(f"\nUploading {os.path.basename(file_path)}...")

        with open(file_path, 'rb') as f:
            # Create progress bar
            with tqdm(
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Uploading {os.path.basename(file_path)}"
            ) as pbar:
                # Wrap file object to track progress
                class FileWrapper:
                    def __init__(self, file_obj, callback):
                        self.file_obj = file_obj
                        self.callback = callback

                    def read(self, size=-1):
                        data = self.file_obj.read(size)
                        self.callback(len(data))
                        return data

                    def __getattr__(self, attr):
                        return getattr(self.file_obj, attr)

                # Upload with progress tracking
                wrapped_file = FileWrapper(f, pbar.update)
                files = {'file': (os.path.basename(file_path), wrapped_file, 'application/octet-stream')}

                response = requests.post(
                    f"{server_url}/upload",
                    files=files,
                    timeout=3600  # 1 hour timeout for large files
                )

        if response.status_code == 200:
            result = response.json()
            print("\nUpload successful!")
            print(f"Total examples: {result['total_examples']:,}")
            print(f"Chunk size: {result['chunk_size']}")
            print(f"Total chunks: {result['total_chunks']}")
            return True
        else:
            print(f"\nUpload failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"\nError during upload: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Upload training data to HTTP server')
    parser.add_argument('--file', type=str, required=True,
                        help='Path to the .pkl file to upload')
    parser.add_argument('--server', type=str, required=True,
                        help='Server URL (e.g., http://192.168.1.100:8000)')

    args = parser.parse_args()

    server_url = args.server.rstrip('/')

    success = upload_file(args.file, server_url)

    if success:
        print("\nDone! The server is ready to serve training data.")
    else:
        print("\nUpload failed.")
        exit(1)


if __name__ == '__main__':
    main()
