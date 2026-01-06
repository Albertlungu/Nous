"""
./src/tokenizer/tiktoken_tokenizer.py

Using OpenAI's TikToken library to get a powerful and fast tokenizer for better training runs.
Although you can use the custom BPE tokenizer, this is faster at the actual tokenization process
    and has better vocabulary distributions
"""

import pickle
from api.paths import get_training_data_path, get_tokenizer_path
from typing import Any

import tiktoken # pylint: disable=no-member
from tqdm import tqdm

class TikToken:
    """
    TikToken tokenizer class for ease of use.
    """
    def __init__(
            self,
            encoding="r50k_base"
            ):
        """
        Initializing TikToken class.

        Args:
            encoding (str, optional): Encoding name from TikToken. Defaults to "r50k_base".
        """
        self.enc = tiktoken.get_encoding(encoding)
        original_vocab = self.enc.max_token_value + 1
        self.vocab_size = 50304  # +1 because max_token_value is 0-indexed
        self._original_vocab_size = original_vocab
        self.eos_token_id = self.enc.eot_token
        # Use 0 as padding token ID (common convention)
        self.padding_token_id = 0

    def encode(
            self,
            text:str
            ):
        """
        Encode text to token IDs.

        Args:
            text (str): Text to encode

        Returns:
            list: List of token IDs
        """
        return self.enc.encode(text)

    # def encode_special(
    #         self,
    #         special_token:bytes
    #         ):
    #     """
    #     Method for encoding special tokens

    #     Args:
    #         special_token (bytes): special token bytes.

    #     Returns:
    #         list: list of token IDs for the special token.
    #     """

    #     if isinstance(special_token, str):
    #         special_token = special_token.encode("utf-8")
    #     return self.enc.encode_single_token(special_token.decode('utf-8'))

    def decode(
            self,
            token_ids:list
            ):
        """
        Decode token IDs to text.

        Args:
            token_ids (list): List of token IDs

        Returns:
            str: Decoded text
        """
        return self.enc.decode(token_ids)

def tokenize_data(
        input_path:str,
        output_path:str
        ):
    """
    Tokenizes training data

    Args:
        input_path (str): Path of the non-tokenized .txt dataset file.
        output_path (str): Output path to which function writes.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()

    training_texts = [doc.strip() for doc in data.split('\n\n') if doc.strip()]

    tokenizer = TikToken()
    token_ids = []

    for i in tqdm(training_texts):
        ids = tokenizer.encode(i)
        ids.append(tokenizer.eos_token_id)
        token_ids.append(ids)

    with open(output_path, "wb") as f:
        pickle.dump(token_ids, f)


def main():
    """
    Main function to test encoding and decoding
    """
    tokenizer = TikToken()

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"Padding token ID: {tokenizer.padding_token_id}")

    # Test encoding and decoding
    text = "Hello, world!"
    encoded = tokenizer.encode(text)
    print(f"\nOriginal text: {text}")
    print(f"Encoded: {encoded}")
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

def test_tokenized(path:str):
    """
    Tests the tokenized dataset to ensure that it looks as it should.

    Args:
        path (str): Path to the pickled tokenized dataset
    """
    with open(path, "rb", encoding="utf-8") as f:
        data = pickle.load(f)

    tokenizer = TikToken()

    print(tokenizer.decode(data[-1]))



if __name__ == "__main__":
    tokenize_data(get_training_data_path('alpaca.txt'), get_training_data_path('tiktoken_alpaca.pkl'))
    main()
    test_tokenized(get_training_data_path('tiktoken_alpaca.pkl'))
