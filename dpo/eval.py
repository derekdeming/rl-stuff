import json 
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
import numpy as np 
from torch.nn.utils.rnn import pad_sequence

model_name = "Qwen/Qwen3-4B-Thinking-2507-dpo"
local_dir = "./dpo_models/" + model_name
data_dir = "safe-pair-data/"
max_length = 1024
output_file = f"safe_rlhf_{model_name}.json"

tokenizer = AutoTokenizer.from_pretrained(local_dir, use_fast=False)
dataset = load_from_disk(data_dir)["test"]
print(f"loaded {len(dataset)} examples")

model = AutoModelForCausalLM.from_pretrained(local_dir)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


