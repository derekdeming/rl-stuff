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

l_dir = "qwen3.2-14b-instruct"
tokenizer = AutoTokenizer.from_pretrained(l_dir, use_fast= false) 
datase = load_from_disk("safe_pair_data/")["train"]


def get_rank(): return int(os.environ.get("L_RANK", 0))

def get_world_size(): return int(os.environ.get("WORLD_SIZE", 1))

def get_local_rank(): return int(os.environ.get("LOCAL_RANK", 0))

def get_local_world_size(): return int(os.environ.get("LOCAL_WORLD_SIZE", 1))

def get_device(): return torch.device(f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu")


