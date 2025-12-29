"""
Training interface
"""

import threading
import sys
import os
import pickle
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.train import Trainer
from src.tokenizer.tiktoken_tokenizer import TikToken
from api.paths import get_tokenizer_path, get_models_path

class TrainingInterface:
    def __init__(self):
        self.trainer = None
        self.training_thread = None
        self.is_training = False
        self.is_paused = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_batch = 0
        self.total_batches = 0
        self.current_loss = 0.0
        self.current_lr = 0.0
        self.stop_requested = False
        self.loss_history = []
        self.lr_history = []
        self.recent_logs = deque(maxlen=100)

    def is_active(self):
        """
        Check if training is currently active
        """
        return self.is_training

    def reset_state(self):
        """
        Force reset training state (use when training is stuck)
        """
        self.is_training = False
        self.is_paused = False
        self.stop_requested = False

    def start_training(self, config):
        """
        Start training.

        Args:
            config (dict): Dictionary containing model configurations.
        """
        if self.is_training:
            raise Exception("Training already in progress")

        token_ids_path = config.get('token_ids_path')
        dataset_path = config.get('dataset_path')

        if not token_ids_path and not dataset_path:
            raise Exception("No dataset provided. Please provide either 'token_ids_path' or 'dataset_path' in config")

        if token_ids_path and not os.path.exists(token_ids_path):
            raise Exception(f"Token IDs file not found: {token_ids_path}")

        if dataset_path and not os.path.exists(dataset_path):
            raise Exception(f"Dataset file not found: {dataset_path}")

        self.stop_requested = False
        self.is_paused = False
        self.total_epochs = config.get('epochs', 10)
        self.loss_history = []
        self.lr_history = []

        def training_loop():
            self.is_training = True
            try:
                tokenizer_type = config.get('tokenizer', 'tiktoken')
                if tokenizer_type == 'tiktoken':
                    tokenizer = TikToken()
                else:
                    tokenizer_path = get_tokenizer_path('tokenizer_alpaca.pkl')
                    with open(tokenizer_path, "rb") as f:
                        tokenizer = pickle.load(f)
                        if hasattr(tokenizer, '_ensure_vocab'):
                            tokenizer._ensure_vocab()

                token_ids_path = config.get('token_ids_path')
                dataset_path = config.get('dataset_path')

                if token_ids_path and os.path.exists(token_ids_path):
                    with open(token_ids_path, "rb") as f:
                        token_ids = pickle.load(f)
                elif dataset_path and os.path.exists(dataset_path):
                    # Check if dataset is .pkl (pre-tokenized) or .txt (needs tokenization)
                    if dataset_path.endswith('.pkl'):
                        with open(dataset_path, 'rb') as f:
                            token_ids = pickle.load(f)
                    else:
                        # .txt file - needs tokenization
                        with open(dataset_path, 'r', encoding='utf-8')  as f:
                            content = f.read()
                            training_texts = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
                            token_ids = []
                            for text in training_texts:
                                ids = tokenizer.encode(text)
                                ids.append(tokenizer.eos_token_id)
                                token_ids.append(ids)
                else:
                    raise Exception("No valid dataset given")

                self._log(f"Loaded {len(token_ids)} training_examples")

                self.trainer = Trainer(
                    tokenizer=tokenizer,
                    token_ids=token_ids,
                    lr=config.get('lr', 1.1e-3),
                    num_blocks=config.get('num_blocks', 8),
                    num_heads=config.get('num_heads', 8),
                    embedding_dim=config.get('embedding_dim', 512),
                    max_seq_length=config.get('max_seq_length', 256),
                    use_moe=config.get('use_moe', True),
                    num_experts=config.get('num_experts', 8),
                    experts_per_token=config.get('experts_per_token', 2),
                    dropout=config.get('dropout', 0.0),
                    use_lr_schedule=config.get('use_lr_schedule', True),
                    warmup_steps=config.get('warmup_steps', 500),
                    min_lr=config.get('min_lr', 5e-6)
                )

                self._log("Trainer initialized")
                self._log(f"Model has {self.trainer.count_parameters()['total']:,} parameters")

                batch_size = config.get('batch_size', 64)
                batches = self.trainer.create_batches(batch_size)
                self.total_batches = len(batches)

                for epoch in range(self.total_epochs):
                    if self.stop_requested:
                        self._log("Training stopped by user")
                        break

                    while self.is_paused and not self.stop_requested:
                        import time
                        time.sleep(0.1)

                        if self.stop_requested:
                            break

                    self.current_epoch = epoch + 1
                    epoch_loss = 0.0

                    for batch_idx, batch in enumerate(batches):
                        if self.stop_requested:
                            break

                        while self.is_paused and not self.stop_requested:
                            import time
                            time.sleep(0.1)

                        self.current_batch = batch_idx + 1

                        import jax.numpy as jnp
                        batch_jax = jnp.array(batch, dtype=jnp.int32)
                        input_tokens = batch_jax[:, :-1]
                        target_tokens = batch_jax[:, 1:]

                        loss, grads = self.trainer.compute_loss_and_grads(input_tokens, target_tokens)
                        self.trainer.update_params(grads)

                        self.current_loss = float(loss)
                        self.current_lr = float(self.trainer.optimizer.lr)
                        epoch_loss += self.current_loss

                    avg_loss = epoch_loss / len(batches)
                    self.loss_history.append(avg_loss)
                    self.lr_history.append(self.current_lr)
                    self._log(f"Epoch {self.current_epoch} complete. Avg loss: {avg_loss:.4f}, LR: {self.current_lr:.8f}")

                    if config.get('save_checkpoints', True) and (epoch + 1) % config.get('save_every', 5) == 0:
                        checkpoint_path = get_models_path(f"checkpoint_epoch{epoch + 1}.pkl")
                        self.trainer.save_checkpoint(checkpoint_path)
                        self._log(f"Checkpoint saved at {checkpoint_path}")

                if not self.stop_requested:
                    final_path = config.get('final_model_path', get_models_path('final_model.pkl'))
                    # If user supplied a relative path, prefer storing in models dir
                    if not os.path.isabs(final_path):
                        final_path = get_models_path(os.path.basename(final_path))
                    self.trainer.save_checkpoint(final_path)
                    self._log(f"Training complete. Model saved: {final_path}")

            except Exception as e:
                self._log(f"Error during training: {str(e)}")
                import traceback
                self._log(traceback.format_exc())
            finally:
                self.is_training = False
                self.is_paused = False

        self.training_thread = threading.Thread(target=training_loop, daemon=True)
        self.training_thread.start()

    def stop_training(self):
        """
        Stop training
        """
        self.stop_requested = True
        if self.training_thread:
            self.training_thread.join(timeout=10)

    def pause_training(self):
        """
        Pause training
        """
        self.is_paused = True

    def resume_training(self):
        self.is_paused = False

    def get_status(self):
        """
        Get current training status
        """
        return {
            "is_training": self.is_training,
            "is_paused": self.is_paused,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "current_loss": float(self.current_loss),
            "current_lr": float(self.current_lr),
            "progress": (self.current_epoch / self.total_epochs * 100) if self.total_epochs > 0 else 0,
            "batch_progress": (self.current_batch / self.total_batches * 100) if self.total_batches > 0 else 0
        }

    def get_history(self):
        """
        Get training history
        """
        return {
            "losses": self.loss_history,
            "learning_rates": self.lr_history,
            "epochs": list(range(1, len(self.loss_history) + 1))
        }

    def get_recent_logs(self, count=50):
        """
        Get recent logs

        Args:
            count (int, optional): How many logs to return. Defaults to 50.
        """
        return list(self.recent_logs)[-count:]

    def _log(self, message:str):
        """
        Add message to logs

        Args:
            message (str): Message that you want to add.
        """
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.recent_logs.append(log_entry)
        print(log_entry)