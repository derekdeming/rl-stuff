import json 
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 
import numpy as np 
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from datasets import load_from_disk

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


# compute log-likelihood func 
def compute_log_likelihood(prompt:str, response:str):
    input_ids, label_ids = [], []
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    input_ids += [torch.tensor(p_ids + r_ids, dtype=torch.long)]
    label_ids += [torch.tensor([-100]*len(p_ids) + r_ids, dtype=torch.long)] # building labels -- mask prompt with -100, keep response tokens 
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id)
    labels_tensor = pad_sequence(label_ids, batch_first=True, padding_value=-100)
    assert input_ids.shape == attention_mask.shape and attention_mask.shape == labels_tensor.shape

    with torch.no_grad():
        outputs = model(input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels_tensor.to(device))
        mean_nll = outputs.loss.item()
        return np.exp(mean_nll)


# streaming eval & collects LLs 
results = []
pbar = tqdm(dataset, total=len(dataset), desc="Evaluating SAFE-RLHF", unit= "ex")
pbar.set_postfix({'acc': '0.00%', 'correct': '0'})
correct = 0
total = 0

for ex in pbar: 
    prompt = ex["prompt"]
    chosen = ex["chosen"]
    rejected = ex["rejected"]

    # compute LLs 
    ll_chosen = compute_log_likelihood(prompt, chosen)
    ll_rejected = compute_log_likelihood(prompt, rejected)

    results.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "ll_chosen": ll_chosen,
        "ll_rejected": ll_rejected
    })

    if ll_chosen > ll_rejected:
        correct += 1
    total += 1

    # compute acc 
    acc = correct / total
    print(f"acc: {acc:.4%}, correct: {correct}/{total}")
    pbar.set_postfix({'acc': f"{acc:.4%}", 'correct': f"{correct}/{total}"})
    pbar.close()
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"saved {len(results)} log-likelihoods pairs to {output_file}")
    break