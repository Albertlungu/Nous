"""
Path configuration for νοῦς (nous)
Handles data paths for both development and production environments
"""
import os

# Get data path from environment variable (set by Electron)
# In development: uses project root
# In production: uses ~/Library/Application Support/nous/
DATA_PATH = os.getenv('NOUS_DATA_PATH', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Helper functions to get proper paths
def get_training_data_path(filename=''):
    """Get path to training_data directory or specific file"""
    path = os.path.join(DATA_PATH, 'training_data')
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename) if filename else path

def get_models_path(filename=''):
    """Get path to models directory or specific file"""
    path = os.path.join(DATA_PATH, 'artifacts', 'models')
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename) if filename else path

def get_tokenizer_path(filename=''):
    """Get path to tokenizer directory or specific file"""
    path = os.path.join(DATA_PATH, 'artifacts', 'tokenizer')
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename) if filename else path
