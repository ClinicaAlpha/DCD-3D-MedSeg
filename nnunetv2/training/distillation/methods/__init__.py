"""Method registry for DCD distillation strategies."""
from __future__ import annotations

from typing import Dict, Type

import torch

from .base import DistillationMethod
from .dcd import DetailConsistentDistillation


class NoDistillation(DistillationMethod):
    """Baseline placeholder that returns zero loss."""

    def forward(self, student_features, teacher_features, target, **kwargs):
        device = target.device if torch.is_tensor(target) else torch.device("cpu")
        return torch.tensor(0.0, device=device), {}

    def get_required_features(self) -> Dict[str, str]:
        return {}


METHOD_REGISTRY: Dict[str, Type[DistillationMethod]] = {
    "dcd": DetailConsistentDistillation,
    "none": NoDistillation,
}


def build_method(name: str, **config) -> DistillationMethod:
    key = name.lower()
    if key not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY))
        raise ValueError(f"Unknown distillation method '{name}'. Available: {available}")
    return METHOD_REGISTRY[key](**config)


__all__ = [
    "DistillationMethod",
    "DetailConsistentDistillation",
    "NoDistillation",
    "METHOD_REGISTRY",
    "build_method",
]
