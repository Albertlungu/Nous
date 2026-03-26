import struct

import numpy as np
import tiktoken


def tokenize_corpus_to_binary(
    input_txt: str,
    output_bin: str,
    num_shards: int = 32,
    seq_length: int = 4096,
):
    """
    Tokenizes the corpus and saves as a memory-mapped binary file

    Args:
        input_txt (str): Path to corpus
        output_bin (str): Base path for shards
        num_shards (int, optional): Number of shard files to create. Defaults to 32.
        seq_length (int, optional): Sequence length. Defaults to 4096.
    """
    enc = tiktoken.get_encoding("cl100k_base")

    total_tokens_estimate = (
        170_756_572_334  # This was pulled using "src/utils/pipeline/estimate_tokens.py"
    )
    tokens_per_shard = total_tokens_estimate // num_shards

    tokens_per_shard = (
        tokens_per_shard // seq_length
    ) * seq_length  # Aligns to sequence length

    print(f"Target tokens per shard: {tokens_per_shard}")
    print(f"Shard sizeL {tokens_per_shard * 4 / (1024**3):.2f} GB")

    shard_idx = 0
    shard_file = f"{output_bin}_{shard_idx:02d}.bin"
    shard_f = open(shard_file, "wb")

    write_header(shard_f, tokens_per_shard, enc.n_vocab)

    shard_tokens = 0
    total_tokens = 0

    with open(input_txt, "r", encoding="utf-8") as f:
        buffer = ""

        while True:
            chunk = f.read(1024**2)
            if not chunk:
                break

            buffer += chunk

            while "\n\n" in buffer:
                doc, buffer = buffer.split("\n\n", 1)
                tokens = enc.encode(doc)
                tokens.append(enc.eot_token)

                for token_id in tokens:
                    shard_f.write(struct.pack(">I", token_id))
                    shard_tokens += 1
                    total_tokens += 1

                    if shard_tokens >= tokens_per_shard:
                        shard_f.close()
                        print(f"Shard {shard_idx}: {shard_tokens:,} tokens")

                        shard_idx += 1
                        shard_file = f"{output_bin}_{shard_idx:02d}.bin"
                        shard_f = open(shard_file, "wb")
                        write_header(shard_f, tokens_per_shard, enc.n_vocab)
                        shard_tokens = 0

                if total_tokens % 10_000_000 == 0:
                    print(f"    Processed {total_tokens:,} tokens ...")
        shard_f.close()
        print(f"Final shard {shard_idx}: {shard_tokens:,} tokens")
        print(f"\nTotal: {total_tokens:,} tokens across {shard_idx + 1} shards")


def write_header(f, num_tokens: int, vocab_size: int):
    """
    Writes the binary file header

    Args:
        f (_type_): file
        num_tokens (int): Number of tokens
        vocab_size (int): Vocabulary size
    """
    f.write(struct.pack(">Q", 0x4E4F555343505253))
    f.write(struct.pack(">Q", 1))
    f.write(struct.pack(">Q", num_tokens))
    f.write(struct.pack(">Q", vocab_size))
    f.write(struct.pack(">Q", 4))
    f.write(struct.pack(b"\x00" * 24))
