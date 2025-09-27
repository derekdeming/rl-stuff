"""Configuration dataclasses for GRPO experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence


@dataclass
class OptimizerConfig:
    """Hyperparameters for the policy optimizer."""

    learning_rate: float = 3e-5
    betas: Sequence[float] = (0.9, 0.95)
    weight_decay: float = 0.01
    eps: float = 1e-8
    max_grad_norm: float = 1.0


@dataclass
class SchedulerConfig:
    """Learning rate scheduling parameters."""

    warmup_steps: int = 100
    total_steps: Optional[int] = None
    min_lr_ratio: float = 0.05


@dataclass
class RolloutConfig:
    """Controls how prompts and responses are generated for GRPO."""

    prompt_batch_size: int = 8
    response_per_prompt: int = 4
    max_prompt_length: int = 256
    max_response_length: int = 256
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: Optional[int] = None


@dataclass
class KLControlConfig:
    """Settings for KL penalty between policy and reference models."""

    target_kl: float = 1.0
    adaptive_kl: bool = True
    kl_coef: float = 0.02
    kl_horizon: int = 1000


@dataclass
class TrainingConfig:
    """Top level configuration for GRPO training runs."""

    seed: int = 42
    total_iterations: int = 1000
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    kl: KLControlConfig = field(default_factory=KLControlConfig)
    log_interval: int = 10
    save_interval: int = 100
    device: str = "cuda"

    def as_dict(self) -> dict:
        """Return a python dict suitable for logging or serialization."""

        def serialize(obj: object) -> object:
            if hasattr(obj, "__dict__"):
                return {
                    key: serialize(value)
                    for key, value in obj.__dict__.items()
                }
            if isinstance(obj, (list, tuple)):
                return type(obj)(serialize(value) for value in obj)
            return obj

        return serialize(self)


__all__ = [
    "OptimizerConfig",
    "SchedulerConfig",
    "RolloutConfig",
    "KLControlConfig",
    "TrainingConfig",
]
