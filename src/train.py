import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from dataset import Dataset
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
    for batch_idx in range(len(unfolded_ids[item_idx])): # Iterating batches
        if (unfolded_ids[item_idx][batch_idx] == pattern).all():
            labels[item_idx][0:batch_idx+len(pattern)] = -100
            # Pattern matched, and i is the index.

## Dataset is in three vars: input_ids_tokenized, attention_mask, labels

## Form dataset acceptable for transformers trainer
class ReceiptDataset(Dataset):
    def __init__(self, input_ids_tokenized, attention_mask, labels):
        self.input_ids = input_ids_tokenized
        self.attention_mask = attention_mask
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }

## Train/Test Split
TRAIN_RATIO = 0.8
n = len(input_ids_tokenized)
indices = torch.randperm(n)
split = int(n * TRAIN_RATIO)
train_idx, test_idx = indices[:split], indices[split:]

train_dataset = ReceiptDataset(input_ids_tokenized[train_idx], attention_mask[train_idx], labels[train_idx])
test_dataset  = ReceiptDataset(input_ids_tokenized[test_idx],  attention_mask[test_idx],  labels[test_idx] ) # PyTorch advanced indexing: tensors accept a list of indices directly



