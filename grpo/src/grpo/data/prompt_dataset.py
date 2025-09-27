from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from torch.utils.data import Dataset

@dataclass
class PromptSample:
    """Single prompt entry used for rollout prompts."""
    text: str
    metadata: Optional[dict] = None


class PromptDataset(Dataset[PromptSample]):
    """In-memory dataset of prompts for GRPO rollouts."""

    def __init__(self, prompts: Iterable[PromptSample]) -> None:
        self._prompts: List[PromptSample] = list(prompts)

    @classmethod
    def from_text_file(cls, path: Path) -> "PromptDataset":
        """Load prompts from a newline separated text file."""
        with path.open("r", encoding="utf-8") as handle:
            prompts = [PromptSample(text=line.strip()) for line in handle if line.strip()]
        return cls(prompts)

    def __len__(self) -> int:
        return len(self._prompts)

    def __getitem__(self, index: int) -> PromptSample:
        return self._prompts[index]

__all__ = [
    "PromptSample",
    "PromptDataset",
]
