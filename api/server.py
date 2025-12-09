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
from src.tokenizer.tokenizer_class import BPETokenizer

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

model_interface = ModelInterface()
training_interface = TrainingInterface()
dataset_interface = DatasetInterface()
tokenizer_interface = TokenizerInterface()

#--------------------------------- Health and Status -------------------------------------

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


#----------------------------------- Model Management ------------------------------------

@app.route('/api/models/list', methods=["GET"])
def list_models():
    """
    List available model checkpoints
    """
    models = model_interface.list_available_models()
    return jsonify({"models": models})

@app.route('/api/models/load', methods=["POST"])
def load_model():
    """
    Load a model checkpoint
    """
    data = request.json
    model_path = data.get('model_path')

    try:
        info = model_interface.load_model(model_path)
        return jsonify({
            "success": True,
            "message": "Model loaded",
            "model_info": info
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/models/info', methods=['GET'])
def get_model_info():
    """
    Get information about loaded model
    """
    if not model_interface.is_loaded():
        return jsonify({"error": "No model loaded"}), 400

    info = model_interface.get_model_info()
    return jsonify(info)

@app.route('/api/models/unload', methods=['POST'])
def unload_model():
    """
    Unload current model
    """
    model_interface.unload_model()
    return jsonify({"success": True, "message": "Model unloaded"})


#----------------------------------- Chat & Gen ------------------------------------
@app.route('/api/chat/generate', methods=['POST'])
def generate_response():
    """
    Generate response
    """
    data = request.json
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 40)

    if not model_interface.is_loaded():
        return jsonify({"error": "No model loaded"}), 400

    def generate():
        try:
            for token in model_interface.generate_stream(prompt, max_tokens, temperature, top_k):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

#----------------------------------- Training ------------------------------------
@app.route('/api/training/start', methods=['POST'])
def start_training():
    """
    Start training with specified configurations
    """
    data = request.json
    config = data.get('config', {})

    try:
        training_interface.start_training(config)
        return jsonify({"sucess": True, "message": "Training started"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """
    Stop current training
    """
    training_interface.stop_training()
    return jsonify({"success": True, "message": "Training stopped"})

@app.route('/api/training/pause', methods=['POST'])
def pause_training():
    """
    Pause current training
    """
    training_interface.pause_training()
    return jsonify({"success": True, "message": "Training paused"})

@app.route('/api/training/resume', methods=['POST'])
def resume_training():
    """
    Resume paused training
    """
    training_interface.resume_training()
    return jsonify({"success": True, "message": "Training resumed"})

@app.route('/api/training/status', methods=['GET'])
def training_status():
    """
    Get current training status
    """
    status = training_interface.get_status()
    return jsonify(status)

@app.route('/api/training/history', methods=['GET'])
def training_history():
    """
    Get training history
    """
    history = training_interface.get_history()
    return jsonify(history)

#----------------------------------- Datasets ------------------------------------
@app.route('/api/datasets/list', methods=['GET'])
def list_datasets():
    """
    List available datasets
    """
    datasets = dataset_interface.list_datasets()
    return jsonify({"datasets": datasets})

@app.route('/api/datasets/info/<dataset_name>', methods=['GET'])
def get_dataset_info(dataset_name:str):
    """
    Get information about a specific dataset.

    Args:
        dataset_name (str): Name of the target dataset.
    """
    info = dataset_interface.get_dataset_info(dataset_name)
    if info:
        return jsonify(info)
    return jsonify({"error": "Dataset not found"}), 404

@app.route('/api/datasets/combine', methods=['POST'])
def combine_datasets():
    """
    Combine multiple datasets
    """
    data = request.json
    dataset_names = data.get('datasets', [])
    output_name = data.get('output_name', 'combined')

    try:
        result = dataset_interface.combine_datasets(dataset_names, output_name)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/datasets/create', methods=['POST'])
def create_dataset(hf_path:str, inst_label:str, output_label:str, path:str, ds_len=0, dataset_branch='train', streaming=True, input_label=None):
    """
    Create dataset.

    Args:
        hf_path (str): HuggingFace dataset library. e.g., tatsu-lab/alpaca
            inst_label (str): Label that shows the instruction given to the model.
            output_label (str): Label that shows the output from the dataset.
            path (str): Output path of dataset.
            ds_len (int, optional): Dataset length, in examples. Defaults to 0 (all examples).
            dataset_branch (str, optional): Branch of the dataset (e.g., 'train'). Defaults to 'train'.
            streaming (bool, optional): Don't load the whole dataset into memory for large datasets. Defaults to true.
            input_label (str, optional): Label that shows the input given to the model. Defaults to None.
    """
    dataset_interface.create_dataset(hf_path, inst_label, output_label, path, ds_len, dataset_branch, streaming, input_label)

#----------------------------------- Tokenizer ------------------------------------
@app.route('/api/tokenizers/list', methods=['GET'])
def list_tokenizers():
    """
    List available tokenizers
    """
    tokenizers = tokenizer_interface.list_tokenizers()
    return jsonify({"tokenizers": tokenizers})

@app.route('/api/tokenizers/current', methods=['GET'])
def get_current_tokenizer():
    """
    Get current tokenizer info
    """
    info = tokenizer_interface.get_tokenizer_info()
    return jsonify(info)

@app.route('/api/tokenizers/switch', methods=['POST'])
def switch_tokenizer():
    """
    Switch to a different tokenizer
    """
    data = request.json
    tokenizer_name = data.get('tokenizer')

    try:
        tokenizer_interface.switch_tokenizer(tokenizer_name)
        return jsonify({"success": True, "message": f"Switched to {tokenizer_name}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

#----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    print("Starting PyGPT API Server...")
    print("Server running at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)