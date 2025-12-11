"""
data_loader.py

This file is used to load any datasets the user desires.
It already includes the Dolly 15k dataset, Alpaca, and TriviaQA datasets from HuggingFace.
All datasets are saved as .txt files, and must be tokenized before training using either the main.py
    save_token_ids() function or the tokenize_data() function inside of
    src/tokenizer/tiktoken_tokenizer.py
"""

from datasets import load_dataset # pylint: disable=no-member
from api.paths import get_training_data_path


def save_dolly(path:str) -> None:
    """
    Load the Databricks Dolly-15k dataset and save it in a formatted text file.

    Args:
        path (str): Output file path for the formatted dataset
    """
    print("Loading Databricks Dolly-15k dataset...")
    dataset_len = 0
    ds = load_dataset("databricks/databricks-dolly-15k")
    ds = ds['train']
    ds = ds.select(range(dataset_len)) if dataset_len != 0 else ds
    print(f"Loaded {len(ds)} examples")

    print(f"Writing to {path}...")
    with open(path, 'w', encoding='utf-8') as f:
        for idx, ex in enumerate(ds):
            # Create formatted entry
            text = (
                f"Instruction: {ex['instruction']}\n"
                f"Input: {ex['context']}\n"
                f"Output: {ex['response']}\n"
                )
            f.write(text + "\n\n")

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(ds)} examples")

    print(f"Successfully saved {len(ds)} examples to {path}")

def save_alpaca(path:str) -> None:
    """
    Load the Alpaca dataset from HuggingFace
    Dataset size: 51,974 examples (20MB, 284,280 lines)

    Args:
        path (str): Output file path for the formatted dataset
    """
    print("Loading Alpaca dataset...")
    ds_len = 0
    ds = load_dataset("tatsu-lab/alpaca")
    ds = ds['train']
    ds = ds.select(range(ds_len)) if ds_len > 0 else ds
    print(f"Loaded {len(ds)} examples")

    print(f"Writing to {path}...")
    with open(path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            text = (
                f"Instruction: {ex['instruction']}\n"
                f"Input: {ex['input']}\n"
                f"Output: {ex['output']}\n"
            )
            f.write(text + "\n\n")

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(ds)} examples")

    print(f"Successfully saved {len(ds)} examples to {path}")

def save_wizardlm(path:str) -> None:
    """
    Load the WizardLM dataset from HuggingFace
    Dataset size: 70,004 examples (126MB, 1,537,373 lines)

    Args:
        path (str): Output file path for the formatted dataset
    """
    print("Loading WizardLM dataset...")
    ds_len = 0
    ds = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k")
    ds = ds['train']
    ds = ds.select(range(ds_len)) if ds_len > 0 else ds
    print(f"Loaded {len(ds)} examples")

    print(f"Writing to {path}...")
    with open(path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            # WizardLM typically has 'instruction' and 'output' fields
            instruction = ex.get('instruction', ex.get('conversations', [{}])[0].get('value', ''))
            output = ex.get('output', ex.get('conversations', [{}])[-1].get('value', ''))

            text = (
                f"Instruction: {instruction}\n"
                f"Input: \n"
                f"Output: {output}\n"
            )
            f.write(text + "\n\n")

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(ds)} examples")

    print(f"Successfully saved {len(ds)} examples to {path}")

def save_flan(path:str) -> None:
    """
    Load the FLAN 50K dataset from HuggingFace using streaming
    Dataset size: 50,000 examples (87MB, 1,962,003 lines)

    Args:
        path (str): Output file path for the formatted dataset
    """
    print("Loading FLAN 50K dataset with streaming...")
    ds_len = 0
    ds = load_dataset("Muennighoff/flan", streaming=True)
    ds = ds['train']

    if ds_len > 0:
        ds = ds.take(ds_len)

    print(f"Writing to {path}...")
    with open(path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            # FLAN typically has 'inputs' and 'targets' fields
            instruction = ex.get('inputs', '')
            output = ex.get('targets', '')

            text = (
                f"Instruction: {instruction}\n"
                f"Input: \n"
                f"Output: {output}\n"
            )
            f.write(text + "\n\n")

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1} examples")

    print(f"Successfully saved {idx + 1} examples to {path}")

def save_gpt_teacher(path:str) -> None:
    """
    Load the GPT Teacher dataset from HuggingFace
    Dataset size: 89,260 examples (55MB, 534,010 lines)

    Args:
        path (str): Output file path for the formatted dataset
    """
    print("Loading GPT Teacher dataset...")
    ds_len = 0
    ds = load_dataset("teknium/GPTeacher-General-Instruct")
    ds = ds['train']
    ds = ds.select(range(ds_len)) if ds_len > 0 else ds
    print(f"Loaded {len(ds)} examples")

    print(f"Writing to {path}...")
    with open(path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            instruction = ex.get('instruction', '')
            input_text = ex.get('input', '')
            output = ex.get('response', '')

            text = (
                f"Instruction: {instruction}\n"
                f"Input: {input_text}\n"
                f"Output: {output}\n"
            )
            f.write(text + "\n\n")

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(ds)} examples")

    print(f"Successfully saved {len(ds)} examples to {path}")

def save_trivia_qa(path:str) -> None:
    """
    Saves mandarjoshi/trivia_qa dataset from HuggingFace using Datasets module

    Args:
        path (str): Output path for the trivia_qa dataset .txt file
    """
    ds_len = 20000
    ds = load_dataset("mandarjoshi/trivia_qa", "rc")
    ds = ds['train']
    ds = ds.select(range(ds_len)) if ds_len > 0 else ds

    with open(path, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(ds):
            # Check if 'answer' is a dict with 'value' and 'aliases'
            answer_info = ex['answer']
            if isinstance(answer_info, dict):
                main_answer = answer_info.get('value', '')
                aliases = answer_info.get('aliases', [])
                aliases_str = ', '.join(aliases) if aliases else 'N/A'
            else:
                main_answer = answer_info
                aliases_str = 'N/A'

            text = f"Question:\n{ex['question']}\nAnswer:\n{main_answer}\nAliases:\n{aliases_str}\n"
            f.write(text + "\n")

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(ds)} examples")

    print(f"Successfully saved {len(ds)} examples to {path}")

def save_general_instruct(path:str, ds_len:int) -> None:
    """
    Save general instruction dataset from HuggingFace's Teknium/GPTeacher-General-Instruct dataset

    Args:
        path (str): Path to save dataset to
        ds_len (int): Length (# examples) of dataset
    """
    skipped_count = 0
    saved_count = 0
    ds = load_dataset("teknium/GPTeacher-General-Instruct")['train']
    ds = ds.select(range(ds_len)) if ds_len > 0 else ds

    with open(path, "w", encoding='utf-8') as f:
        for idx, ex in enumerate(ds):
            response = ex['response'].strip()

            if not response or response.lower in ['<nooutput>', '<no output>', 'none', '']:
                skipped_count += 1
                continue

            text = f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nOutput: {response}\n"
            saved_count += 1
            f.write(text + "\n\n")

            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx+1}/{ds_len} examples")

    print(f"Successfully saved {saved_count} examples to {path}")
    print(f"Skipped {skipped_count} examples")


def load_text_file(path:str) -> list:
    """
    Load a text file for training.

    Args:
        path (str): Path to the text file

    Returns:
        list: List of text strings
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to separate examples
    examples = [ex.strip() for ex in content.split('\n\n') if ex.strip()]
    return examples


def combine_all_datasets(output_path:str, alpaca_path:str = None, wizardlm_path:str = None,
                         flan_path:str = None, gpt_teacher_path:str = None) -> None:
    """
    Combine all datasets into a single file.

    This function can either:
    1. Load existing dataset files and combine them (if paths are provided)
    2. Download and save each dataset, then combine them (if paths are None)

    Total combined size: 261,238 examples
    - Alpaca: 51,974 examples
    - WizardLM: 70,004 examples
    - FLAN 50K: 50,000 examples
    - GPT Teacher: 89,260 examples

    Args:
        output_path (str): Path to save the combined dataset
        alpaca_path (str, optional): Path to alpaca dataset file. If None, will download.
        wizardlm_path (str, optional): Path to wizardlm dataset file. If None, will download.
        flan_path (str, optional): Path to flan dataset file. If None, will download.
        gpt_teacher_path (str, optional): Path to gpt_teacher dataset file. If None, will download.
    """
    import os

    temp_dir = get_training_data_path('temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Download datasets if paths not provided
    if alpaca_path is None:
        alpaca_path = os.path.join(temp_dir, "alpaca.txt")
        print("\nDownloading Alpaca dataset...")
        save_alpaca(alpaca_path)

    if wizardlm_path is None:
        wizardlm_path = os.path.join(temp_dir, "wizardlm.txt")
        print("\nDownloading WizardLM dataset...")
        save_wizardlm(wizardlm_path)

    if flan_path is None:
        flan_path = os.path.join(temp_dir, "flan.txt")
        print("\nDownloading FLAN dataset...")
        save_flan(flan_path)

    if gpt_teacher_path is None:
        gpt_teacher_path = os.path.join(temp_dir, "gpt_teacher.txt")
        print("\nDownloading GPT Teacher dataset...")
        save_gpt_teacher(gpt_teacher_path)

    # Combine all datasets
    print(f"\nCombining all datasets into {output_path}...")
    total_examples = 0

    with open(output_path, 'w', encoding='utf-8') as out_file:
        # Combine each dataset
        for dataset_name, dataset_path in [
            ("Alpaca", alpaca_path),
            ("WizardLM", wizardlm_path),
            ("FLAN", flan_path),
            ("GPT Teacher", gpt_teacher_path)
        ]:
            if os.path.exists(dataset_path):
                print(f"Adding {dataset_name} dataset...")
                with open(dataset_path, 'r', encoding='utf-8') as in_file:
                    content = in_file.read()
                    out_file.write(content)
                    if not content.endswith('\n\n'):
                        out_file.write('\n\n')

                # Count examples
                examples = [ex for ex in content.split('\n\n') if ex.strip()]
                print(f"  Added {len(examples)} examples from {dataset_name}")
                total_examples += len(examples)
            else:
                print(f"Warning: {dataset_path} not found, skipping {dataset_name}")

    print(f"\nSuccessfully combined {total_examples} total examples into {output_path}")


if __name__ == "__main__":
    # save_dolly(get_training_data_path('pygpt_training_corpus.txt'))
    # save_general_knowledge(get_training_data_path('general_knowledge.txt'))
    save_alpaca(get_training_data_path('alpaca.txt'))
    # save_trivia_qa(get_training_data_path('trivia.txt'))
