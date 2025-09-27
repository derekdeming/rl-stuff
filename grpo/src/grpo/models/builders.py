"""Helpers for instantiating Hugging Face backbones."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .policy import PolicyWithValue, ReferenceModel

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - handled lazily
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

Device = Union[str, torch.device]


def _ensure_transformers() -> None:
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError(
            "transformers must be installed to load Hugging Face backbones"
        ) from _IMPORT_ERROR


def _init_value_head(hidden_size: int, dtype: torch.dtype, device: Device) -> nn.Module:
    head = nn.Linear(hidden_size, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head.to(device=device, dtype=dtype)


def load_hf_policy_and_reference(
    model_name: str = "Qwen/Qwen3-Coder-7B-Instruct",
    *,
    device: Optional[Device] = None,
    torch_dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    model_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[PolicyWithValue, ReferenceModel, "AutoTokenizer"]:
    """Instantiate policy/reference models plus tokenizer from Hugging Face."""
    _ensure_transformers()
    preload_args: Dict[str, object] = {"output_hidden_states": True}
    if torch_dtype is not None:
        preload_args["torch_dtype"] = torch_dtype
    if cache_dir is not None:
        preload_args["cache_dir"] = cache_dir
    if trust_remote_code:
        preload_args["trust_remote_code"] = True
    if model_kwargs:
        preload_args.update(model_kwargs)

    policy_backbone = AutoModelForCausalLM.from_pretrained(model_name, **preload_args)
    reference_backbone = AutoModelForCausalLM.from_pretrained(model_name, **preload_args)

    policy_backbone.config.output_hidden_states = True
    reference_backbone.config.output_hidden_states = True

    if device is not None:
        policy_backbone.to(device)
        reference_backbone.to(device)

    dtype = next(policy_backbone.parameters()).dtype
    current_device = next(policy_backbone.parameters()).device
    value_head = _init_value_head(policy_backbone.config.hidden_size, dtype, current_device)
    policy = PolicyWithValue(policy_backbone, value_head)

    reference = ReferenceModel(reference_backbone)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    policy.backbone.config.pad_token_id = tokenizer.pad_token_id
    reference.backbone.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(policy.backbone, "generation_config"):
        policy.backbone.generation_config.pad_token_id = tokenizer.pad_token_id
        policy.backbone.generation_config.eos_token_id = tokenizer.eos_token_id

    return policy, reference, tokenizer


__all__ = ["load_hf_policy_and_reference"]
