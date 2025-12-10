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

        # BPE training progress tracking
        self.bpe_training_state = {
            'is_training': False,
            'current_merge': 0,
            'total_merges': 0,
            'is_complete': False,
            'error': None,
            'output_path': None
        }

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

    def train_bpe(self, dataset_path, vocab_size, output_path):
        """
        Train a new BPE tokenizer with progress tracking

        Args:
            dataset_path (str): Path to dataset file
            vocab_size (int): Desired vocabulary size
            output_path (str): Where to save the trained tokenizer
        """
        import threading

        # Reset progress state
        self.bpe_training_state = {
            'is_training': True,
            'current_merge': 0,
            'total_merges': vocab_size - 256,
            'is_complete': False,
            'error': None,
            'output_path': output_path
        }

        def train_worker():
            try:
                with open(dataset_path, "r", encoding='utf-8') as f:
                    dataset = f.read()

                tokenizer = BPETokenizer(vocab_size)

                encoded = dataset.encode('utf-8')
                ids = list(encoded)

                # Progress callback
                def update_progress(current, total):
                    self.bpe_training_state['current_merge'] = current
                    self.bpe_training_state['total_merges'] = total

                tokenizer.make_merges(ids, len(ids), progress_callback=update_progress)

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    pickle.dump(tokenizer, f)

                self.tokenizer = tokenizer
                self.current_tokenizer = 'bpe'

                # Mark complete
                self.bpe_training_state['is_complete'] = True
                self.bpe_training_state['is_training'] = False

            except Exception as e:
                self.bpe_training_state['error'] = str(e)
                self.bpe_training_state['is_training'] = False
                print(f"BPE training error: {e}")
                import traceback
                traceback.print_exc()

        # Start training in background thread
        training_thread = threading.Thread(target=train_worker, daemon=True)
        training_thread.start()

    def get_bpe_progress(self):
        """
        Get current BPE training progress

        Returns:
            dict: Progress state with current_merge, total_merges, is_complete, error
        """
        return self.bpe_training_state

    def load_bpe(self, tokenizer_path):
        """
        Load BPE tokenizer from custom path

        Args:
            tokenizer_path (str): Path to tokenizer .pkl file
        """
        if not os.path.exists(tokenizer_path):
            raise Exception(f"Tokenizer not found at {tokenizer_path}")

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
            if hasattr(self.tokenizer, '_ensure_vocab'):
                self.tokenizer._ensure_vocab()

        self.current_tokenizer = 'bpe'

    def set_tiktoken(self, name='r50k_base'):
        """
        Set TikToken tokenizer with a specific name

        Args:
            tokenizer_name (str): TikToken tokenizer name (e.g., 'r50k_base', 'cl100k_base')
        """
        self.tokenizer = TikToken(name)
        self.current_tokenizer = 'tiktoken'