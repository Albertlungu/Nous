"""
Dataset interface for PyGPT electron
"""

import os
import glob
import sys


from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import (
    save_alpaca, save_wizardlm, save_dolly, save_flan, save_gpt_teacher
)

class DatasetInterface:
    def __init__(self):
        self.datasets_dir = "training_data"

    def list_datasets(self):
        """
        List available datasets
        """
        if not os.path.exists(self.datasets_dir):
            return []

        datasets = []
        for file in glob.glob(f"{self.datasets_dir}/*.txt"):
            name = os.path.basename(file)
            size = os.path.getsize(file)

            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    examples = len([ex for ex in content.split('\n\n') if ex.strip()])
            except:
                examples = 0
            datasets.append({
                "name": name,
                "path": file,
                "size_mb": round(size / (1024 * 1024), 2),
                "examples": examples
            })

        return sorted(datasets, key=lambda x: x['name'])

    def get_dataset_info(self, dataset_name):
        """
        Get detailed info about a dataset

        Args:
            dataset_name (str): name of the target dataset
        """
        path = os.path.join(self.datasets_dir, dataset_name)
        if not os.path.exists(path):
            return None

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        examples = [ex.strip() for ex in content.split('\n\n') if ex.strip()]
        total_chars = sum(len(ex) for ex in examples)

        return {
            "name": dataset_name,
            "path": path,
            "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
            "examples": len(examples),
            "avg_characters": round(total_chars / len(examples)) if examples else 0,
            "preview": examples[0][:500] if examples else ""
        }

    def combine_datasets(self, dataset_names, output_name):
        """
        Combine multiple datasets

        Args:
            dataset_names (str): Name of the input files.
            output_name (str): Name of the output file.
        """
        output_path = os.path.join(self.datasets_dir, f"{output_name}.txt")

        total_examples = 0
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for name in dataset_names:
                path = os.path.join(self.datasets_dir, name)
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as in_file:
                        content = in_file.read()
                        out_file.write(content)
                        if not content.endswith('\n\n'):
                            out_file.write('\n\n')

                    examples = len([ex for ex in content.split('\n\n') if ex.strip()])
                    total_examples += examples

        return {
            "output_path": output_path,
            "total_examples": total_examples,
            "size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2)
        }

    def download_dataset(self, dataset_name):
        """
        Download dataset using existing functions.

        Args:
            dataset_name (str): Name of the output file to download dataset to.
        """

        output_path = os.path.join(self.datasets_dir, f"{dataset_name}.txt")

        dataset_map = {
            'alpaca': save_alpaca,
            'wizardlm': save_wizardlm,
            'flan': save_flan,
            'gpt_teacher': save_gpt_teacher
        }

        if dataset_name.lower() not in dataset_map:
            raise Exception(f"Unknown dataset: {dataset_name}.")

        download_func = dataset_map[dataset_name.lower()]
        download_func(output_path)

        return {
            "name": dataset_name,
            "path": output_path,
            "size_mb": round(os.path.getsize(output_path) / (1024 * 1024), 2)
        }

    def create_dataset(self, hf_path:str, inst_label:str, output_label:str, path:str, ds_len=0, dataset_branch='train', streaming=True, input_label=None):
        """
        Create a new dataset, not preexisting, from HuggingFace

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

        print(f"Loading {ds_len} examples from {hf_path} dataset from HuggingFace...")
        ds = load_dataset(hf_path, streaming=streaming)
        ds = ds[dataset_branch]
        ds = ds.take(ds_len) if ds_len > 0 else ds

        print(f"Writing to {path}...")
        with open(path, 'w', encoding='utf-8') as f:
            for idx, ex in enumerate(ds):
                instruction = ex.get(inst_label, '')
                inpt = ex.get(input_label, '') if input_label else ''
                output = ex.get(output_label, '')

                text = (
                    f"Instruction: {instruction}\n",
                    f"Input: {inpt}\n",
                    f"Output: {output}"
                )

                f.write(text + '\n\n')

                if (idx + 1) % 1000 == 0:
                    print(f"Processed {idx + 1} examples")
        print(f"Successfuly saved {idx+1} examples to {path}")