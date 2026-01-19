"""
Model interface for PyGPT Electron App
Handles model loading, inference and generation
"""

import pickle
import jax
import jax.numpy as jnp
import sys
import os
from threading import Lock
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.train import Trainer
from src.tokenizer.tiktoken_tokenizer import TikToken
from src.tokenizer.tokenizer_class import BPETokenizer
from api.paths import get_models_path, get_tokenizer_path

class ModelInterface:
    def __init__(self):
        self.trainer = None
        self.tokenizer = None
        self.current_model_path = None
        self.lock = Lock()

    def is_loaded(self):
        """
        Check if model is loaded
        """
        return self.trainer is not None and self.tokenizer is not None

    def list_available_models(self):
        """
        List all available model checkpoints
        """
        models_dir = get_models_path()
        models = glob.glob(os.path.join(models_dir, "*.pkl"))
        return sorted([{
            "name": os.path.basename(m),
            "path": m,
            "size_mb": os.path.getsize(m) / (1024 * 1024),
            "modified": os.path.getmtime(m)
        } for m in models], key=lambda x: x['modified'], reverse=True)

    def load_model(self, model_path:str):
        """
        Load model an tokenizer

        Args:
            model_path (str): Path to model user seeks to load.
        """

        with self.lock:
            print(f"Loading model from {model_path}...")

            with open(model_path, "rb") as f:
                self.checkpoint = pickle.load(f)

            config = self.checkpoint.get('config', {})


            tokenizer_type = config.get('tokenizer', 'tiktoken')

            if tokenizer_type == 'tiktoken':
                self.tokenizer = TikToken()
            else:
                tokenizer_path = get_tokenizer_path('tokenizer_alpaca.pkl')
                if os.path.exists(tokenizer_path):
                    with open(tokenizer_path, "rb") as f:
                        self.tokenizer = pickle.load(f)
                        if hasattr(self.tokenizer, '_ensure_vocab'):
                            self.tokenizer._ensure_vocab()
                else:
                    self.tokenizer = TikToken()

            self.trainer = Trainer(
                tokenizer=self.tokenizer,
                token_ids=None,
                lr=config.get('lr', 1e-4),
                num_blocks=config.get('num_blocks', 8),
                num_heads=config.get('num_heads', 8),
                embedding_dim=config.get('embedding_dim', 512),
                max_seq_length=256,
                dropout=0.0
            )

            self.trainer.load_checkpoint(model_path)

            self.current_model_path = model_path
            print(f"Model loaded successfully!")

            return self.get_model_info()

    def unload_model(self):
        """
        Unload current model to free memory
        """
        with self.lock:
            self.trainer = None
            self.tokenizer = None
            self.current_model_path = None

    def get_model_info(self):
        """
        Get information about currently loaded model.
        """
        if not self.is_loaded():
            return None

        param_counts = self.trainer.count_parameters()

        return {
            "model_path": self.current_model_path,
            "config": {
                "num_blocks": self.trainer.num_blocks,
                "num_heads": self.trainer.num_heads,
                "embedding_dim": self.trainer.embedding_dim,
                "vocab_size": self.tokenizer.vocab_size,
                "max_seq_len": self.trainer.max_seq_length
            },
            "total_parameters": param_counts['total'],
            "parameter_breakdown": param_counts,
            "tokenizer": type(self.tokenizer).__name__,
            "vocab_size": self.tokenizer.vocab_size
        }

    def generate_stream(self, prompt:str, max_tokens=100, temperature=0.7, top_k=40):
        """
        Generate tokens one at a time (streaming).

        Args:
            prompt (str): User prompt to model
            max_tokens (int, optional): Maximum output token length. Defaults to 100.
            temperature (float, optional): Temperature value of model. Dictates how 'creative' it is. Defaults to 0.7.
            top_k (int, optional): Tok K of the model, so it is able to look at its top k next token probabilities. Defaults to 40.
        """

        if not self.is_loaded():
            raise Exception("Model not loaded")

        with self.lock:
            for token_id in self.trainer.generate_stream(
                prompt=prompt,
                max_length=max_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=1.2
            ):
                token_text = self.tokenizer.decode([token_id])
                yield token_text

    def generate(self, prompt:str, max_tokens=100, temperature=0.7, top_k=40):
        """
        Generate complete response.

        Args:
            prompt (str): User prompt to model
            max_tokens (int, optional): Maximum output token length. Defaults to 100.
            temperature (float, optional): Temperature value of model. Dictates how 'creative' it is. Defaults to 0.7.
            top_k (int, optional): Tok K of the model, so it is able to look at its top k next token probabilities. Defaults to 40.
        """
        if not self.is_loaded():
            raise Exception("Model not loaded")

        with self.lock:
            import io
            from contextlib import redirect_stdout

            output = io.StringIO()
            with redirect_stdout(output):
                _, token_ids = self.trainer.generate(
                    prompt=prompt,
                    max_length=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=1.2,
                    debug=False
                )

            response = self.tokenizer.decode(token_ids)
            return response