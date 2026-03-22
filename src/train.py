from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
import torch
import json

# Model & Preprocessor Loading
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8b-Base", dtype = 'float32', device_map = 'mps')
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8b-Base")
if tokenizer.pad_token is None: # Just In Case
    tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = 'left' # Left Padding for transformers

# Load Dataset
dataset = []
with open('./dataset.jsonl') as f:
    for line in f:
        if line.strip():
            dataset.append(json.loads(line))

## Loading input_ids
input_ids_raw = []
for i in range(len(dataset)):
    input_ids_raw.append(f"# OCR\n{dataset[i]['ocr_result']}\n# OUTPUT\n{json.dumps(dataset[i]['label'])}<|endoftext|>")

## Dataset Tokenize
tokenized_inputs = tokenizer(input_ids_raw, padding = True, return_tensors = 'pt')
input_ids_tokenized, attention_mask = [tokenized_inputs[key] for key in ("input_ids", "attention_mask")]
print(input_ids_tokenized)

labels = input_ids_tokenized.clone()

## Find pattern
pattern = tokenizer(["# OUTPUT\n"], return_tensors = 'pt')["input_ids"][0]
print(pattern)
unfolded_ids = labels.unfold(1,pattern.size(-1),1)
print(unfolded_ids)

for item_idx in range(len(unfolded_ids)): # Iterating the sentences
    pass



# Training Sector

