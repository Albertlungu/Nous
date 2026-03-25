"""
Compress Corpus to GZIP
Compresses the uncompressed corpus file to .txt.gz for storage efficiency.
Provides 5-10x space savings with lossless compression.
"""

import os
import gzip
import shutil


def compress_corpus(input_path, output_path=None, compression_level=6, delete_original=False):
    """
    Compress a text corpus file to gzip format.

    Args:
        input_path: Path to uncompressed .txt file
        output_path: Path to save .txt.gz file (default: input_path + '.gz')
        compression_level: Compression level 1-9 (6=balanced, 9=max compression)
        delete_original: If True, delete uncompressed file after compression
    """
    if not os.path.exists(input_path):
        print(f"✗ Error: Input file not found: {input_path}")
        return

    if output_path is None:
        output_path = input_path + '.gz'

    input_size_gb = os.path.getsize(input_path) / (1024 ** 3)

    print("=" * 80)
    print("COMPRESSING CORPUS TO GZIP")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Input size: {input_size_gb:.2f} GB")
    print(f"Compression level: {compression_level}/9")
    print("=" * 80)
    print("\nCompressing... (this may take a while)")

    # Compress file with progress
    chunk_size = 1024 * 1024 * 100  # 100MB chunks
    bytes_processed = 0
    total_bytes = os.path.getsize(input_path)

    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb', compresslevel=compression_level) as f_out:
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                f_out.write(chunk)
                bytes_processed += len(chunk)

                # Progress update
                progress_pct = (bytes_processed / total_bytes) * 100
                gb_processed = bytes_processed / (1024 ** 3)
                print(f"  Progress: {progress_pct:.1f}% ({gb_processed:.2f} GB / {input_size_gb:.2f} GB)", end='\r')

    print()  # New line after progress

    output_size_gb = os.path.getsize(output_path) / (1024 ** 3)
    compression_ratio = input_size_gb / output_size_gb if output_size_gb > 0 else 0
    space_saved_gb = input_size_gb - output_size_gb

    print("\n" + "=" * 80)
    print("COMPRESSION COMPLETE")
    print("=" * 80)
    print(f"Original size:    {input_size_gb:.2f} GB")
    print(f"Compressed size:  {output_size_gb:.2f} GB")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Space saved:      {space_saved_gb:.2f} GB ({(space_saved_gb/input_size_gb)*100:.1f}%)")
    print("=" * 80)

    if delete_original:
        print(f"\nDeleting original file: {input_path}")
        os.remove(input_path)
        print("✓ Original file deleted")
    else:
        print(f"\nOriginal file kept: {input_path}")
        print("To delete manually: rm \"{input_path}\"")

    print("\nTo read compressed file:")
    print(f"  Terminal: zcat \"{output_path}\" | less")
    print(f"  Python:   gzip.open('{output_path}', 'rt')")


def main():
    # Hardcoded path to Seagate HDD corpus
    input_path = "/Volumes/Seagate HDD/training_data/nous_corpus.txt"
    output_path = "/Volumes/Seagate HDD/training_data/nous_corpus.txt.gz"

    print("This will compress the corpus file to save disk space.")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("\nOptions:")
    print("  1. Compress and keep original (safe)")
    print("  2. Compress and delete original (saves space)")
    print("  3. Cancel")

    choice = input("\nChoose option (1-3): ").strip()

    if choice == "1":
        compress_corpus(input_path, output_path, compression_level=6, delete_original=False)
    elif choice == "2":
        confirm = input("Are you sure you want to delete the original? (yes/no): ").strip().lower()
        if confirm == "yes":
            compress_corpus(input_path, output_path, compression_level=6, delete_original=True)
        else:
            print("Cancelled.")
    else:
        print("Cancelled.")


if __name__ == "__main__":
    main()
