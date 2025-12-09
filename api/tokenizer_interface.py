"""
Tokenizer interface
"""

import sys
import os
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tokenizer.tiktoken_tokenizer import TikToken
from src.tokenizer.tokenizer_class import BPETokenizer

class TokenizerInterface:
    def __init__(self):
        self.current_tokenizer = 'tiktoken'
        self.tiktoken_tokenizer = TikToken()
        self.tokenizer = TikToken() # TikToken by default, this one changes, the one above does not.

    def list_tokenizers(self):
        """
        List all available tokenizers
        """
        return [
            {
                "name": "tiktoken",
                "description": "OpenAI TikToken",
                "vocab_size": self.tiktoken_tokenizer.vocab_size
            },
            {
                "name": "bpe",
                "description": "Custom BPE tokenizer",
                "vocab_size": "varies"
            }
        ]

    def get_tokenizer_info(self):
        """
        Get current tokenizer info
        """
        return {
            "current": self.current_tokenizer,
            "vocab_size": self.get_vocab_size(),
            "type": type(self.tokenizer).__name__
        }

    def switch_tokenizer(self, tokenizer_name:str):
        """
        Switch tokenizer

        Args:
            tokenizer_name (str): Name of the tokenizer you are switching to (bpe or TikToken)
        """
        if tokenizer_name.lower() == 'tiktoken':
            self.tokenizer = TikToken()
            self.current_tokenizer = 'tiktoken'
        elif tokenizer_name.lower() == 'bpe':
            tokenizer_path = "artifacts/tokenizer/tokenizer_alpaca.pkl"
            if not os.path.exists(tokenizer_path):
                raise Exception("BPE tokenizer not found")
            with open(tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
                if hasattr(self.tokenizer, '_ensure_vocab'):
                    self.tokenizer._ensure_vocab()
            self.current_tokenizer = 'bpe'
        else:
            raise Exception(f"Unknown tokenizer: {tokenizer_name}")

    def get_vocab_size(self):
        """
        Get vocab size
        """
        return self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 0

    def encode(self, text:str):
        """
        Encode text

        Args:
            text (str): Text you want to encode
        """
        return self.tokenizer.encode(text)

    def decode(self, tokens:list):
        return self.tokenizer.decode(tokens)