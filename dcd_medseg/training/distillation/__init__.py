"""DCD distillation package."""
from .config import DistillationConfig, get_baseline_config
from .distiller import DistillationTrainer
from .ema import ExponentialMovingAverage
from .methods import (
    METHOD_REGISTRY,
    DetailConsistentDistillation,
    build_method,
)

__all__ = [
    "DistillationTrainer",
    "DistillationConfig",
    "DetailConsistentDistillation",
    "build_method",
    "METHOD_REGISTRY",
    "get_baseline_config",
    "ExponentialMovingAverage",
]
