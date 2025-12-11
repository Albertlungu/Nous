# Save as check_merges.py and run on your GPU machine
import pickle
from api.paths import get_tokenizer_path, get_training_data_path

with open(get_tokenizer_path('tokenizer_alpaca.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

print(tokenizer.encode("Hello World"))

prompts = [
        "Instruction: Construct an analogy based on the given two words. \nInput: Air, Water. \nOutput:",
        "Instruction: What is the major contribution of the philosopher Immanuel Kant?\nInput: \nOutput:",
        "Instruction: Create a list of 20 vocabulary words related to marine animals.\nInput: \nOutput:",
        "Instruction: Write a description of the Golden Gate Bridge.\nInput: \nOutput:",
        "Instruction: List three best practices for starting a conversation.\nInput: \nOutput:",
        "Instruction: Explain the importance of NASA's current mission to Mars.\nInput: \nOutput:"
    ]

tokenizer.vocab[tokenizer.eos_token_id] = b"<EOS>"
tokenizer.vocab[0] = b"<PAD>"

print(tokenizer.encode(prompts))
print(tokenizer.eos_token_id)
print(tokenizer.decode([32000]))
print(tokenizer.decode([0]))


with open(get_training_data_path('alpaca_tokenized.pkl'), 'rb') as f:
    data = pickle.load(f)

print(data[-1])
print(tokenizer.decode(data[0]))
