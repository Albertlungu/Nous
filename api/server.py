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
import random
import pickle

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
    instruction = data.get('prompt', '')
    context = data.get('context', '')
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.7)
    top_k = data.get('top_k', 40)

    if not model_interface.is_loaded():
        return jsonify({"error": "No model loaded"}), 400

    if context:
        formatted_prompt = f"Instruction {instruction}\nInput: {context}\nOutput:"
    else:
        formatted_prompt = f"Instruction: {instruction}\nInput:\nOutput:"

    def generate():
        try:
            for token in model_interface.generate_stream(formatted_prompt, max_tokens, temperature, top_k):
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
        return jsonify({"success": True, "message": "Training started"})
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

@app.route('/api/tokenizers/train-bpe', methods=['POST'])
def train_bpe_tokenizer():
    """
    Trains a new BPE tokenizer.
    """
    data = request.json
    dataset_path = data.get('dataset_path')
    vocab_size = data.get('vocab_size', 5000)
    output_path = data.get('output_path', 'artifacts/tokenizer/custom_bpe.pkl')

    try:
        tokenizer_interface.train_bpe(dataset_path, vocab_size, output_path)
        return jsonify({"success": True, "message": "BPE Tokenizer trained successfully", "path": output_path})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/tokenizers/load-bpe', methods=['POST'])
def load_bpe_tokenizer():
    """
    Loads BPE tokenizer from custom path
    """
    data = request.json
    tokenizer_path = data.get('tokenizer_path')

    try:
        tokenizer_interface.load_bpe(tokenizer_path)
        return jsonify({"success": True, "messsage":f"Loaded BPE tokenizer from {tokenizer_path}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/tokenizers/set-tiktoken', methods=['POST'])
def set_tiktoken():
    """
    Set TikToken tokenizer with specific name
    """
    data = request.json
    name = data.get('tokenizer_name', 'r50k_base')

    try:
        tokenizer_interface.set_tiktoken(name)
        return jsonify({"success": True, "message":f"Set tokenizer to TikToken's {name}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/tokenizers/bpe-progress', methods=['GET'])
def bpe_progress():
    """
    Get current BPE training progress
    """
    progress = tokenizer_interface.get_bpe_progress()
    return jsonify(progress)


#----------------------------------- Dataset Management ------------------------------------

@app.route('/api/datasets/list', methods=['GET'])
def list_datasets():
    """List all available datasets"""
    try:
        datasets = []
        training_data_dir = 'training_data'

        if not os.path.exists(training_data_dir):
            return jsonify({'datasets': []})

        for filename in os.listdir(training_data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(training_data_dir, filename)

                # Count examples (separated by double newlines)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    examples = len([ex for ex in content.split('\n\n') if ex.strip()])

                # Get file size
                size_bytes = os.path.getsize(filepath)
                size_mb = f"{size_bytes / (1024 * 1024):.1f}MB"

                datasets.append({
                    'name': filename,
                    'path': filepath,
                    'examples': examples,
                    'size': size_mb
                })

        return jsonify({'datasets': datasets})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/create', methods=['POST'])
def create_dataset():
    """Create a new dataset from preset/HuggingFace/local files"""
    try:
        from datasets import load_dataset
        import re

        data = request.json
        source_type = data.get('source_type')
        format_template = data.get('format_template', 'Instruction: {instruction}\nInput: {input}\nOutput: {output}\n')
        output_filename = data.get('output_filename')

        if not output_filename:
            return jsonify({'error': 'Output filename is required'}), 400

        output_path = os.path.join('training_data', output_filename)
        os.makedirs('training_data', exist_ok=True)

        all_examples = []

        # Helper function to format examples
        def format_example(example, template):
            field_pattern = r'\{(\w+)\}'
            fields = re.findall(field_pattern, template)
            formatted_text = template
            for field in fields:
                value = example.get(field, '')
                if isinstance(value, dict):
                    value = value.get('value', '')
                formatted_text = formatted_text.replace(f'{{{field}}}', str(value))
            return formatted_text.strip()

        if source_type == 'preset':
            presets = data.get('presets', [])
            preset_map = {
                'alpaca': ('tatsu-lab/alpaca', 'train'),
                'wizardlm': ('WizardLM/WizardLM_evol_instruct_V2_196k', 'train'),
                'flan': ('Muennighoff/flan', 'train'),
                'gpt-teacher': ('teknium/GPTeacher-General-Instruct', 'train'),
                'dolly': ('databricks/databricks-dolly-15k', 'train'),
                'trivia-qa': ('mandarjoshi/trivia_qa', 'train')
            }

            for preset in presets:
                if preset in preset_map:
                    ds_path, split = preset_map[preset]
                    if preset == 'trivia-qa':
                        ds = load_dataset(ds_path, "rc")[split]
                    else:
                        ds = load_dataset(ds_path)[split]

                    for example in ds:
                        all_examples.append(format_example(example, format_template))

        elif source_type == 'huggingface':
            hf_path = data.get('hf_path')
            hf_split = data.get('hf_split', 'train')
            max_examples = data.get('max_examples', 0)

            ds = load_dataset(hf_path)[hf_split]
            if max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))

            for example in ds:
                all_examples.append(format_example(example, format_template))

        elif source_type == 'file':
            file_paths = data.get('file_paths', [])
            for file_path in file_paths:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    all_examples.extend([ex.strip() for ex in content.split('\n\n') if ex.strip()])

        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in all_examples:
                f.write(example + '\n\n')

        return jsonify({
            'success': True,
            'output_path': output_path,
            'examples': len(all_examples)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/combine', methods=['POST'])
def combine_datasets():
    """Combine multiple datasets into one"""
    try:
        data = request.json
        dataset_paths = data.get('dataset_paths', [])
        output_filename = data.get('output_filename')
        shuffle = data.get('shuffle', False)

        if len(dataset_paths) < 2:
            return jsonify({'error': 'At least 2 datasets required'}), 400

        if not output_filename:
            return jsonify({'error': 'Output filename is required'}), 400

        output_path = os.path.join('training_data', output_filename)
        all_examples = []

        # Read all datasets
        for path in dataset_paths:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                examples = [ex.strip() for ex in content.split('\n\n') if ex.strip()]
                all_examples.extend(examples)

        # Shuffle if requested
        if shuffle:
            random.shuffle(all_examples)

        # Write combined dataset
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in all_examples:
                f.write(example + '\n\n')

        return jsonify({
            'success': True,
            'output_path': output_path,
            'total_examples': len(all_examples)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/tokenize', methods=['POST'])
def tokenize_dataset():
    """Tokenize a dataset using BPE or TikToken"""
    try:
        import tiktoken

        data = request.json
        dataset_path = data.get('dataset_path')
        output_filename = data.get('output_filename')
        tokenizer_type = data.get('tokenizer_type')

        if not all([dataset_path, output_filename, tokenizer_type]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Read dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Tokenize based on type
        if tokenizer_type == 'tiktoken':
            tokenizer_name = data.get('tokenizer_name', 'gpt2')
            encoding = tiktoken.get_encoding(tokenizer_name)
            token_ids = encoding.encode(content)

        elif tokenizer_type == 'bpe':
            tokenizer_path = data.get('tokenizer_path')
            if not tokenizer_path:
                return jsonify({'error': 'BPE tokenizer path required'}), 400

            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)

            token_ids = tokenizer.encode(content)

        else:
            return jsonify({'error': 'Invalid tokenizer type'}), 400

        # Save tokenized data
        output_path = os.path.join('training_data', output_filename)
        with open(output_path, 'wb') as f:
            pickle.dump(token_ids, f)

        return jsonify({
            'success': True,
            'output_path': output_path,
            'num_tokens': len(token_ids)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/preview', methods=['POST'])
def preview_dataset():
    """Preview first few examples from a dataset"""
    try:
        data = request.json
        dataset_path = data.get('dataset_path')
        num_examples = data.get('num_examples', 3)

        if not dataset_path:
            return jsonify({'error': 'Dataset path is required'}), 400

        with open(dataset_path, 'r', encoding='utf-8') as f:
            content = f.read()

        examples = [ex.strip() for ex in content.split('\n\n') if ex.strip()]
        preview_examples = examples[:num_examples]

        return jsonify({'examples': preview_examples})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/remove', methods=['POST'])
def remove_dataset():
    """Delete a dataset file"""
    try:
        data = request.json
        dataset_path = data.get('dataset_path')

        if not dataset_path:
            return jsonify({'error': 'Dataset path is required'}), 400

        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset not found'}), 404

        os.remove(dataset_path)

        return jsonify({
            'success': True,
            'message': 'Dataset removed successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


#----------------------------------- Main ------------------------------------
if __name__ == '__main__':
    print("Starting PyGPT API Server...")
    print("Server running at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)