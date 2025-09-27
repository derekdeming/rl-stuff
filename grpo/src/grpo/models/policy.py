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
        # TODO: need to integrate actual transformer outputs here.
        raise NotImplementedError("Implement forward pass by integrating with backbone outputs")

    @torch.no_grad()
    def generate(self, *args: Any, **kwargs: Any) -> Dict[str, Tensor]:
        """Sample sequences from the policy; to be overridden per backbone."""
        raise NotImplementedError("Provide sampling implementation for the chosen backbone")


class ReferenceModel(nn.Module):
    """Frozen copy of the pre-trained model used for KL penalty computation."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Return logits of the frozen reference model."""
        # TODO: need to implement reference forward pass to obtain logits
        raise NotImplementedError("Implement reference forward pass to obtain logits")


__all__ = [
    "ForwardOutput",
    "PolicyWithValue",
    "ReferenceModel",
]
