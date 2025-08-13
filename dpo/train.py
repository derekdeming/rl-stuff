import os 
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

import math 
from torch.utils.data import DataLoader, Dataset 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

torch.autograd.set_detect_anomaly(True)

'''
This is a simple implementation of DPO from scratch.

first we need to define the model and the dataset. we will use some open source model 
and load a train split of the SAFE-RLHF 

this is going to be able to run on multiple GPUs 

'''

hf_dir = "Qwen/Qwen3-4B-Thinking-2507"
tokenizer = AutoTokenizer.from_pretrained(hf_dir, use_fast= False) 
ds = load_dataset('Mingyin0312/safe-pair-data')["train"]


def get_rank(): return int(os.environ.get("HF_RANK", 0))

def get_world_size(): return int(os.environ.get("WORLD_SIZE", 1))

def get_local_rank(): return int(os.environ.get("LOCAL_RANK", 0))

def get_local_world_size(): return int(os.environ.get("HF_LOCAL_WORLD_SIZE"))

def get_device(): return torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")

primary_process = get_rank() == 0

'''
for the pad token and id we're going to use eos_token and eos_token_id

going to use the eos token and eos token id to pad the sequences 


'''

def setup_tokenizer(tokenizer_obj, eos_token, eos_token_id):
    tokenizer_obj.pad_token = eos_token
    tokenizer_obj.pad_token_id = eos_token_id
    return tokenizer_obj

eos_token, eos_token_id = tokenizer.eos_token, tokenizer.eos_token_id
tokenizer = setup_tokenizer(tokenizer, eos_token, eos_token_id)

model = AutoModelForCausalLM.from_pretrained(hf_dir, device_map="auto", torch_dtype=torch.bfloat16)
model.to(get_device())


# need to load frozen model for DPO regularization 
ref_model = AutoModelForCausalLM.from_pretrained(hf_dir, device_map="auto", torch_dtype=torch.bfloat16)
ref_model.to(get_device())
ref_model.eval()

for param in ref_model.parameters():
    param.requires_grad = False

# wrap the training model with a distributed data parallel
model = DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank())


class PreferenceDataset(Dataset):
    def __init__(self, hf_dataset): 
        self.data = hf_dataset
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["prompt"], item["chosen"], item["rejected"]

train_dataset = PreferenceDataset(ds)

