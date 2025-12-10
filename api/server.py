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
        # If training is stuck (flag is True but no thread is running), reset it
        if training_interface.is_training and (not training_interface.training_thread or not training_interface.training_thread.is_alive()):
            training_interface.reset_state()

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
def list_datasets_v2():
    """List all available datasets"""
    try:
        datasets = []
        training_data_dir = 'training_data'

        if not os.path.exists(training_data_dir):
            return jsonify({'datasets': []})

        for filename in os.listdir(training_data_dir):
            if filename.endswith('.txt') or filename.endswith('.pkl'):
                filepath = os.path.join(training_data_dir, filename)

                # Get file size
                size_bytes = os.path.getsize(filepath)
                size_mb = f"{size_bytes / (1024 * 1024):.1f}MB"

                # Count examples
                examples = 0
                if filename.endswith('.txt'):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            examples = len([ex for ex in content.split('\n\n') if ex.strip()])
                    except:
                        examples = 0
                elif filename.endswith('.pkl'):
                    try:
                        with open(filepath, 'rb') as f:
                            data = pickle.load(f)
                            if isinstance(data, list):
                                examples = len(data)
                            else:
                                examples = 0
                    except:
                        examples = 0

                datasets.append({
                    'name': filename,
                    'path': filepath,
                    'examples': examples if examples > 0 else 'Unknown',
                    'size': size_mb
                })

        return jsonify({'datasets': datasets})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/create', methods=['POST'])
def create_dataset_v2():
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


@app.route('/api/datasets/import-files', methods=['POST'])
def import_files():
    """Import .txt or .pkl files directly to training_data/"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')

        if not files:
            return jsonify({'error': 'No files provided'}), 400

        training_data_dir = 'training_data'
        os.makedirs(training_data_dir, exist_ok=True)

        imported_files = []
        for file in files:
            if not file.filename:
                continue

            # Only accept .txt and .pkl files
            if not (file.filename.endswith('.txt') or file.filename.endswith('.pkl')):
                return jsonify({'error': f'Invalid file type: {file.filename}. Only .txt and .pkl files are allowed.'}), 400

            # Save file to training_data/
            output_path = os.path.join(training_data_dir, file.filename)
            file.save(output_path)
            imported_files.append(output_path)

        return jsonify({
            'success': True,
            'imported_files': imported_files,
            'count': len(imported_files)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/datasets/combine', methods=['POST'])
def combine_datasets_v2():
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

        # Check if dataset is .pkl (tokenized) or .txt (raw text)
        if dataset_path.endswith('.pkl'):
            with open(dataset_path, 'rb') as f:
                token_ids = pickle.load(f)

            # For .pkl files, show token ID arrays
            preview_examples = []
            for i, ids in enumerate(token_ids[:num_examples]):
                if isinstance(ids, list):
                    preview_examples.append(f"Token IDs (length {len(ids)}): {ids[:50]}{'...' if len(ids) > 50 else ''}")
                else:
                    preview_examples.append(f"Token IDs: {str(ids)[:200]}")
        else:
            # .txt file - show raw text
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