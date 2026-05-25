"""Configuration system for DCD distillation training."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import os
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DistillationConfig:
    """
    Unified configuration for distillation training.
    """

    # ==================== Teacher Model ====================
    teacher_checkpoint: Optional[str] = None
    teacher_plans: Optional[str] = None
    freeze_teacher: bool = True

    # ==================== Configuration ====================
    configuration: str = "3d_fullres"

    # ==================== Student Model ====================
    reduction_factor: int = 1
    student_plans: Optional[str] = None
    student_architecture: Optional[str] = None
    student_features_per_stage: Optional[List[int]] = None

    # ==================== Distillation Strategy ====================
    strategy: str = "dcd"
    strategy_config: Dict[str, Any] = field(default_factory=dict)

    # ==================== Loss Weights ====================
    kd_weight: float = 0.5
    kd_schedule: str = "constant"
    kd_warmup_epochs: int = 0
    kd_warmup_start_epoch: int = 0

    # ==================== Training Configuration ====================
    num_epochs: int = 1000
    num_iterations_per_epoch: Optional[int] = None
    batch_size: Optional[int] = None
    initial_lr: float = 1e-2
    weight_decay: float = 3e-5

    # ==================== Validation & Checkpointing ====================
    val_interval: int = 1
    save_interval: int = 100
    early_stopping_patience: int = 40
    eval_with_best: bool = False

    # ==================== Visualization-only Mode ====================
    visualize_only: bool = False
    visualize_only_batches: int = 1

    # ==================== Logging ====================
    log_interval: int = 10
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None

    # ==================== Experiment Naming ====================
    experiment_tag: Optional[str] = None
    experiment_tag_mode: str = "append"

    # ==================== Advanced Options ====================
    mixed_precision: bool = True
    compile_model: Optional[bool] = None
    num_workers: int = 12
    seed: Optional[int] = None
    use_ema: Optional[bool] = None
    ema_decay: float = 0.999

    # ==================== I/O Methods ====================

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DistillationConfig":
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config_dict = _expand_paths(config_dict)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        config_dict = asdict(self)
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
        return self

    def __repr__(self) -> str:
        lines = ["DistillationConfig:"]
        for key, value in asdict(self).items():
            if isinstance(value, dict) and value:
                lines.append(f"  {key}:")
                for k, v in value.items():
                    lines.append(f"    {k}: {v}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def _expand_paths(value):
    """Expand environment variables and ~ in YAML string values."""
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_paths(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_paths(v) for k, v in value.items()}
    return value


def get_baseline_config(**kwargs) -> DistillationConfig:
    config = DistillationConfig(strategy="none", kd_weight=0.0)
    config.update(**kwargs)
    return config


__all__ = [
    "DistillationConfig",
    "get_baseline_config",
]
