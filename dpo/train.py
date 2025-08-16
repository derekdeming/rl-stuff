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


"""
need to collate the data into a batch 
batch of chosen and rejected
"""

train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

def collate_fn(batch):
    input_ids, labels_list = [], []
    for prompt, chosen_resp, rejected_resp in batch:
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(chosen_resp, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(rejected_resp, add_special_tokens=False)["input_ids"]
        input_ids += [torch.tensor(p_ids + c_ids, dtype = torch.long), torch.tensor(p_ids + r_ids, dtype = torch.long)]
        labels_list += [torch.tensor([-100]*len(p_ids) + c_ids, dtype = torch.long), torch.tensor([-100]*len(p_ids) + r_ids, dtype = torch.long)]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id)
    labels_tensor = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    assert input_ids.shape == attention_mask.shape and attention_mask.shape == labels_tensor.shape
    return input_ids.to(get_local_rank()), attention_mask.to(get_local_rank()), labels_tensor.to(get_local_rank())


def def_lr(it, max_steps, warmup_steps = None, max_lr=2e-6, min_lr=2e-7):
    warmup_steps = int(0.01*max_steps)
    if it < warmup_steps: # linear warmup 
        return max_lr * (it+1) / warmup_steps # return max learning rate
    if it > max_steps: # return min learning rate
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0 (this describes the cosine decay)
    return min_lr + coeff * (max_lr - min_lr)
    
# setup the dataloader for distributed training 
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=get_world_size(),
    rank=get_rank(),
    shuffle=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    sampler=train_sampler,
    collate_fn=collate_fn,
    # num_workers=4,    # adjust if needed
    # pin_memory=True
)

max_steps = len(train_loader)
if primary_process: 
    print(f"the epoch will run for {max_steps} steps")

# optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
torch.set_float32_matmul_precision('high')

# training loop for DPO fine-tuning 
epochs = 3
beta = 0.1
if primary_process:
    log_dir = "dpo_models"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

gradient_accumulation_steps = 4
optimizer.zero_grad(set_to_none=True)

for epoch in range(number_epochs):
    train_sampler.set_epoch(epoch)
    model.train()
    for steps, (input_ids, attention_mask, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{number_epochs}")):
        is_grad_accum_step = (steps + 1) % gradient_accumulation_steps == 0
        model.require_backward_grad_sync = is_grad_accum_step
        loss_ = 0.0 # purely for logging
        #forward current model 
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits # .float() [B*2, T, V] which is [batch_size, seq_len, vocab_size]

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise RuntimeError(f"NaN or Inf in logits at step: {steps}")

        # forward reference model (frozen model -- no grads)
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                ref_outputs = ref_model(input_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits # .float()

        logits = logits[..., :-1, :].contiguous() # [2B, T-1, V]
        ref_logits = ref_logits[..., :-1, :].contiguous() # [2B, T-1, V]
        labels = labels[..., :1].contiguous() # [2B, T-1]

        # early check 2: ref_logits 
        if torch.isnan(ref_logits).any() or torch.isinf(ref_logits).any():
            raise RuntimeError(f"NaN or Inf in ref_logits at step: {steps}")

        # compute the per-token NLL (negative log likelihood)
        V = logits.size(-1)
        loss_t = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction='none').view(logits.size(0), logits.size(1)) # [2B, T-1]
        ref_loss_t = F.cross_entropy(ref_logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction='none').view(ref_logits.size(0), ref_logits.size(1)) # [2B, T-1]

        # check 3 : per-token losses 
        if torch.isnan(loss_t).any() or torch.isinf(loss_t).any():
            raise RuntimeError(f"NaN or Inf in loss_t at step: {steps}")

        if torch.isnan(ref_loss_t).any() or torch.isinf(ref_loss_t).any():
            raise RuntimeError(f"NaN or Inf in ref_loss_t at step: {steps}")

        # sum to get sequence NLL 
        nll_seq = loss_t.sum(dim=-1) # [2B]
        ref_nll_seq = ref_loss_t.sum(dim=-1) # [2B]

        # check 4 : sequence NLL 
        if torch.isnan(nll_seq).any() or torch.isinf(nll_seq).any():
            raise RuntimeError(f"NaN or Inf in nll_seq at step: {steps}")
        if torch.isnan(ref_nll_seq).any() or torch.isinf(ref_nll_seq).any():
            raise RuntimeError(f"NaN or Inf in ref_nll_seq at step: {steps}")

        # split chosen and rejected 
        nll_c, nll_r = nll_seq[0::2], nll_seq[1::2]
        ref_nll_c, ref_nll_r = ref_nll_seq[0::2], ref_nll_seq[1::2]

        #dpo loss
        diff_thetas = (nll_r - nll_c) # .unsqueeze(-1) # [B]
        diff_ref = ref_nll_r - ref_nll_c # [B]
        inner = beta*(diff_thetas - diff_ref)

        # check inner
        if torch.isnan(inner).any() or torch.isinf(inner).any():
            raise RuntimeError(f"NaN or Inf in inner at step: {steps}")

        # dpo loss 
        dpo_loss = -F.logsigmoid(inner).mean()
        dpo_loss = dpo_loss/gradient_accumulation_steps
        loss_ += dpo_loss.detach()


