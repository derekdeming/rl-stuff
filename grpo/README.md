# GRPO from scratch 

Implementations of **Group Relative Policy Optimization (GRPO)** in PyTorch, built from first principles—no reliance on libraries like [TRL](https://github.com/huggingface/trl) or [VERL](https://github.com/volcengine/verl). 

- GRPO paper: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

## Motivation

this is for learning purposes and to make it less of a black box

i want to demystify the training stack—masking, KL control, scheduling, and evaluation—so you can see exactly how these methods work end to end.

Although libs like TRL, VERL, Puffer are great, i want to understand the internals of these methods and how they work.

## 

---

## Backbone (2025-09-27)

The initial build now targets the Qwen3-Coder family, following the open-source release in [Qwen3-Coder (2025)](https://arxiv.org/abs/2505.09388). Call `src/grpo/models.load_hf_policy_and_reference` to fetch a policy/reference/tokenizer trio backed by `Qwen/Qwen3-Coder-7B-Instruct` from Hugging Face. This requires `torch` (2.4+ recommended with CUDA 12 support) and `transformers` 4.43 or newer with `trust_remote_code=True`. The loader zero-initializes the scalar value head matched to the model hidden size and configures left padding plus EOS-as-PAD so rollouts can stream responses without manual token fiddling. When crafting prompts, rely on `tokenizer.apply_chat_template` from the upstream release to stay aligned with Qwen’s conversation format.
