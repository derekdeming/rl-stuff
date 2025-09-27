"""Model wrappers for policy, value, and reference networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ForwardOutput:
    """Container for policy forward pass outputs."""

    logits: Tensor
    values: Tensor
    hidden_states: Optional[Tensor] = None


class PolicyWithValue(nn.Module):
    """Wrapper around an autoregressive language model with a learned value head."""

    def __init__(self, backbone: nn.Module, value_head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = value_head

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> ForwardOutput:
        """Return policy logits and value predictions for given inputs."""
        kwargs: Dict[str, Any] = {"input_ids": input_ids, "return_dict": True}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        kwargs.setdefault("output_hidden_states", True)
        outputs = self.backbone(**kwargs)
        last_hidden = getattr(outputs, "hidden_states", None)
        if last_hidden:
            features = last_hidden[-1]
        else:
            features = outputs.last_hidden_state
        values = self.value_head(features).squeeze(-1)
        return ForwardOutput(logits=outputs.logits, values=values, hidden_states=features)

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        **generate_kwargs: Any,
    ) -> Any:
        """Sample sequences from the policy backbone."""
        if attention_mask is not None:
            generate_kwargs.setdefault("attention_mask", attention_mask)
        return self.backbone.generate(input_ids=input_ids, **generate_kwargs)


class ReferenceModel(nn.Module):
    """Frozen copy of the pre-trained model used for KL penalty computation."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Return logits of the frozen reference model."""
        kwargs: Dict[str, Any] = {"input_ids": input_ids, "return_dict": True}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        outputs = self.backbone(**kwargs)
        return outputs.logits


__all__ = [
    "ForwardOutput",
    "PolicyWithValue",
    "ReferenceModel",
]
