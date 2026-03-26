"""
Backwards Corpus Tokenizer
Tokenizes nous_corpus.txt FROM END TO START
Saves tokenized data to .pkl and compressed archive to .txt.zst (ZSTD compression)
Truncates processed portions to free disk space during processing
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pickle
import json
import zstandard as zstd
from src.tokenizer.tiktoken_tokenizer import TikToken


def read_chunk_from_end(file_path, current_size, chunk_examples=10000):
    """
    Read a chunk of examples from the end of the file.

    Args:
        file_path: Path to text file
        current_size: Current file size in bytes
        chunk_examples: Target number of examples to read

    Returns:
        tuple: (examples_list, new_file_size, bytes_read)
    """
    # Read from end in large blocks to find example boundaries
    block_size = 1024 * 1024  # 1MB blocks
    examples = []
    bytes_read = 0

    with open(file_path, 'rb') as f:
        # Start from end
        position = current_size
        buffer = b''

        while len(examples) < chunk_examples and position > 0:
            # Move back one block
            read_size = min(block_size, position)
            position -= read_size
            f.seek(position)
            block = f.read(read_size)

            # Prepend to buffer
            buffer = block + buffer
            bytes_read += read_size

            # Try to decode and split into examples
            try:
                text = buffer.decode('utf-8')
                # Split by double newline (example separator)
                parts = text.split('\n\n')

                # Last part might be incomplete, save it
                if position == 0:
                    # We're at the beginning, use all parts
                    new_examples = [p.strip() for p in parts if p.strip()]
                else:
                    # Keep last part in buffer for next iteration
                    new_examples = [p.strip() for p in parts[1:] if p.strip()]
                    buffer = parts[0].encode('utf-8')

                # Add new examples (in reverse since we're reading backwards)
                examples = new_examples + examples

            except UnicodeDecodeError:
                # Buffer doesn't align with UTF-8, keep reading
                continue

            # Stop if we have enough examples
            if len(examples) >= chunk_examples:
                # Only use the examples we need
                examples = examples[-chunk_examples:]
                break

    # Calculate actual bytes to truncate (align to example boundary)
    # We need to find where the examples we're keeping actually end in the file
    new_size = current_size - bytes_read

    return examples, new_size, bytes_read


def tokenize_corpus_backwards(input_path, output_path, archive_path, checkpoint_frequency=10000, chunk_size=10000):
    """
    Tokenize corpus BACKWARDS and save as pickle file (SPACE-SAVING VERSION)

    Processes the corpus from END to START to enable file truncation:
    - Reads chunks from end of file
    - Tokenizes each chunk
    - Appends tokenized data to .pkl file
    - Appends compressed text to archive .txt.gz
    - Truncates processed portion from input file (frees disk space!)
    - Saves checkpoint every N examples for resume capability

    Final result: Examples in reverse order (last to first), but doesn't matter for shuffled training.

    Args:
        input_path: Path to nous_corpus.txt
        output_path: Path to save nous_corpus.pkl
        archive_path: Path to save compressed archive nous_corpus_archive.txt.gz
        checkpoint_frequency: Save checkpoint every N examples (default 10000)
        chunk_size: Number of examples per chunk (default 10000)

    Returns:
        Dictionary with statistics
    """
    checkpoint_path = output_path.replace('.pkl', '_checkpoint.json')

    print("="*80)
    print("TOKENIZING NOUS CORPUS BACKWARDS (SPACE-SAVING MODE)")
    print("="*80)
    print(f"Input: {input_path}")
    print(f"Output (tokenized): {output_path}")
    print(f"Archive (compressed): {archive_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint frequency: every {checkpoint_frequency:,} examples")
    print(f"Chunk size: {chunk_size:,} examples")
    print("="*80)
    print("\nNOTE: Processing BACKWARDS (end to start)")
    print("This allows us to truncate the file and free disk space as we go!")
    print("="*80)

    # Initialize tokenizer
    print("\nLoading TikToken tokenizer (r50k_base)...")
    tokenizer = TikToken()
    print(f"✓ Loaded tokenizer (vocab size: {tokenizer.vocab_size:,})")

    # Check input file exists
    if not os.path.exists(input_path):
        print(f"✗ Error: Input file not found at {input_path}")
        return None

    # Load checkpoint if exists
    initial_size = os.path.getsize(input_path)
    checkpoint = {
        "current_file_size": initial_size,
        "examples_processed": 0,
        "tokens_processed": 0,
        "original_file_size": initial_size
    }

    pkl_mode = "wb"
    archive_mode = "wb"

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            loaded_checkpoint = json.load(f)

        # Backward compatibility: if old checkpoint format, use current file size
        if 'current_file_size' not in loaded_checkpoint:
            print("\n✓ Old checkpoint format detected, using current file size...")
            loaded_checkpoint['current_file_size'] = os.path.getsize(input_path)
            loaded_checkpoint['original_file_size'] = loaded_checkpoint.get('original_file_size', initial_size)

        checkpoint.update(loaded_checkpoint)

        print(f"\n✓ Resuming from checkpoint:")
        print(f"  Examples processed: {checkpoint['examples_processed']:,}")
        print(f"  Tokens processed: {checkpoint['tokens_processed']:,}")
        print(f"  Current file size: {checkpoint['current_file_size']:,} bytes ({checkpoint['current_file_size']/(1024**3):.2f} GB)")
        print(f"  Original file size: {checkpoint['original_file_size']:,} bytes ({checkpoint['original_file_size']/(1024**3):.2f} GB)")
        print(f"  Progress: {(1 - checkpoint['current_file_size']/checkpoint['original_file_size'])*100:.1f}% complete")
        pkl_mode = "ab"
        archive_mode = "ab"
    else:
        print("\n✓ Starting fresh (no checkpoint found)")
        print(f"  File size: {initial_size:,} bytes ({initial_size/(1024**3):.2f} GB)")

    # Statistics
    total_tokens = checkpoint["tokens_processed"]
    total_examples = checkpoint["examples_processed"]
    current_file_size = checkpoint["current_file_size"]
    skipped = 0
    min_tokens = float('inf')
    max_tokens = 0

    print(f"\n{'='*80}")
    print("STARTING BACKWARDS TOKENIZATION...")
    print(f"{'='*80}\n")

    # Open output files
    # Create ZSTD compressor (level 3 = good balance of speed vs compression)
    cctx = zstd.ZstdCompressor(level=3)

    with open(output_path, pkl_mode) as pkl_f:
        if archive_mode == "wb":
            archive_f = open(archive_path, archive_mode)
            archive_writer = cctx.stream_writer(archive_f)
        else:  # append mode
            archive_f = open(archive_path, archive_mode)
            archive_writer = cctx.stream_writer(archive_f, closefd=False)

        with archive_writer:

            while current_file_size > 0:
                # Read chunk from end
                try:
                    examples, new_file_size, bytes_read = read_chunk_from_end(
                        input_path, current_file_size, chunk_size
                    )

                    if not examples:
                        print("No more examples to process")
                        break

                    print(f"  Read {len(examples):,} examples ({bytes_read:,} bytes) from end")

                    # Process each example in the chunk
                    for text in examples:
                        if text:
                            try:
                                # Tokenize
                                ids = tokenizer.encode(text)
                                ids.append(tokenizer.eos_token_id)

                                # Save tokenized version
                                pickle.dump(ids, pkl_f)

                                # Save compressed text version
                                archive_writer.write((text + "\n\n").encode('utf-8'))

                                # Update statistics
                                token_count = len(ids)
                                total_tokens += token_count
                                total_examples += 1
                                min_tokens = min(min_tokens, token_count)
                                max_tokens = max(max_tokens, token_count)

                            except Exception as e:
                                skipped += 1
                                if skipped <= 10:
                                    print(f"\nWarning: Skipped example due to error: {e}")

                    # Truncate the file (FREE DISK SPACE!)
                    with open(input_path, 'r+b') as f:
                        f.truncate(new_file_size)

                    current_file_size = new_file_size

                    # Save checkpoint
                    if total_examples % checkpoint_frequency < chunk_size:
                        checkpoint["current_file_size"] = current_file_size
                        checkpoint["examples_processed"] = total_examples
                        checkpoint["tokens_processed"] = total_tokens

                        with open(checkpoint_path, 'w') as chk:
                            json.dump(checkpoint, chk, indent=2)

                        # Flush to disk
                        pkl_f.flush()
                        os.fsync(pkl_f.fileno())
                        archive_writer.flush(zstd.FLUSH_FRAME)  # ZSTD flush

                        remaining_gb = current_file_size / (1024**3)
                        processed_gb = (checkpoint['original_file_size'] - current_file_size) / (1024**3)
                        progress_pct = (processed_gb / (checkpoint['original_file_size']/(1024**3))) * 100

                        print(f"  Checkpoint: {total_examples:,} examples | {total_tokens:,} tokens")
                        print(f"             Remaining: {remaining_gb:.2f} GB | Processed: {processed_gb:.2f} GB | Progress: {progress_pct:.1f}%")

                except Exception as e:
                    print(f"\n✗ Error processing chunk: {e}")
                    # Save checkpoint before failing
                    checkpoint["current_file_size"] = current_file_size
                    checkpoint["examples_processed"] = total_examples
                    checkpoint["tokens_processed"] = total_tokens
                    with open(checkpoint_path, 'w') as chk:
                        json.dump(checkpoint, chk, indent=2)
                    raise

    # Print statistics
    print(f"\n{'='*80}")
    print("TOKENIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total examples: {total_examples:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens/example: {total_tokens / total_examples:.1f}" if total_examples > 0 else "N/A")
    print(f"Skipped examples: {skipped:,}")
    print(f"{'='*80}")

    if min_tokens != float('inf'):
        print(f"\nToken Distribution:")
        print(f"  Min tokens: {min_tokens:,}")
        print(f"  Max tokens: {max_tokens:,}")

    # Print file sizes
    pkl_size_gb = os.path.getsize(output_path) / (1024 ** 3)
    archive_size_gb = os.path.getsize(archive_path) / (1024 ** 3)
    remaining_txt_size = os.path.getsize(input_path) / (1024 ** 3)

    print(f"\n{'='*80}")
    print("OUTPUT FILES")
    print(f"{'='*80}")
    print(f"Tokenized data: {output_path}")
    print(f"  Size: {pkl_size_gb:.2f} GB")
    print(f"\nCompressed archive: {archive_path}")
    print(f"  Size: {archive_size_gb:.2f} GB")
    print(f"\nOriginal text file: {input_path}")
    print(f"  Remaining size: {remaining_txt_size:.2f} GB (should be ~0 GB)")
    print(f"{'='*80}")

    # Clean up checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"\n✓ Checkpoint file removed (tokenization completed successfully)")

    # Optionally delete the empty text file
    if remaining_txt_size < 0.01:  # Less than 10MB
        print(f"\n✓ Original text file is now empty")
        print(f"  You can delete it with: rm \"{input_path}\"")

    return {
        'total_examples': total_examples,
        'total_tokens': total_tokens,
        'skipped': skipped,
        'min_tokens': min_tokens if min_tokens != float('inf') else 0,
        'max_tokens': max_tokens,
        'pkl_size_gb': pkl_size_gb,
        'archive_size_gb': archive_size_gb
    }


def main():
    """
    Main function to tokenize corpus.txt backwards
    """
    print("="*80)
    print("NOUS CORPUS BACKWARDS TOKENIZER")
    print("Using TikToken tokenizer (cl100k_base)")
    print("Processes from END to START to free disk space as we go")
    print("="*80)

    # Paths to Extreme SSD
    input_path = "/Volumes/Extreme SSD/Nous/training_data/text/corpus.txt"
    output_path = "/Volumes/Extreme SSD/Nous/training_data/text/corpus.pkl"
    archive_path = "/Volumes/Extreme SSD/Nous/training_data/text/corpus_archive.txt.zst"

    # Tokenize backwards
    stats = tokenize_corpus_backwards(
        input_path=input_path,
        output_path=output_path,
        archive_path=archive_path,
        checkpoint_frequency=100000,  # Save checkpoint every 100K examples
        chunk_size=10000
    )

    if stats is not None:
        print("\n" + "="*80)
        print("TOKENIZATION PIPELINE COMPLETE!")
        print("="*80)
        print(f"Tokenized corpus: {output_path} ({stats['pkl_size_gb']:.2f} GB)")
        print(f"Compressed archive: {archive_path} ({stats['archive_size_gb']:.2f} GB)")
        print(f"Total space used: {stats['pkl_size_gb'] + stats['archive_size_gb']:.2f} GB")
        print(f"Space saved: {359 - (stats['pkl_size_gb'] + stats['archive_size_gb']):.2f} GB")
        print(f"\nReady for training!")
        print("="*80)


if __name__ == "__main__":
    main()
