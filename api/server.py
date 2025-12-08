"""
Flask API server for PyGPT Electron App
Gets REST endpoints for model inference, training, and more.
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import sys
import os
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.model_interface import ModelInterface
from api.training_interface import TrainingInterface
from api.dataset_interface import DatasetInterface
from api.tokenizer_interface import TokenizerInterface

app = Flask(__name__)
CORS(app)

model_interface = ModelInterface()
training_interface = TrainingInterface()
dataset_interface = DatasetInterface()
tokenizer_interface = TokenizerInterface()

@app.route('/api/health', methods=["GET"])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "model_loaded": model_interface.is_loaded(),
        "training_active": training_interface.is_active()
    })

@app.route('/api/health', methods=["GET"])
def get_status():
    """
    Get overall system status
    """
    return jsonify({
        "model": {
            "loaded": model_interface.is_loaded(),
            "training_active": training_interface.is_active()
        }
    })