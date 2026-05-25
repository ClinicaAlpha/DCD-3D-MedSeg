#!/usr/bin/env python3
"""Train a DCD student model from a YAML distillation config."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.distillation import DistillationConfig, DistillationTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Detail Consistent Distillation with nnU-Net.")
    parser.add_argument("--config", required=True, help="Path to a DCD YAML config.")
    parser.add_argument("--dataset", required=True, help="nnU-Net dataset name, for example Dataset226_BraTS2024-BraTS-GLI.")
    parser.add_argument("--fold", required=True, type=int, help="Training fold.")
    parser.add_argument("--configuration", default="3d_fullres", help="nnU-Net configuration.")
    parser.add_argument("--resume_from", help="Optional checkpoint path to resume from.")
    parser.add_argument("--device", default="cuda", help="Torch device, default: cuda.")
    parser.add_argument("--skip_eval", action="store_true", help="Skip validation after training.")
    parser.add_argument("--export_validation_probabilities", action="store_true")
    parser.add_argument("--verify_features", action="store_true")
    return parser.parse_args()


def load_dataset_info(dataset_name: str, configuration: str, plans_path_override: str | None = None):
    preprocessed_dir = join(nnUNet_preprocessed, dataset_name)

    dataset_json_path = join(preprocessed_dir, "dataset.json")
    if not Path(dataset_json_path).exists():
        raise FileNotFoundError(f"Dataset JSON not found: {dataset_json_path}")
    with open(dataset_json_path, "r") as f:
        dataset_json = json.load(f)

    plans_path = plans_path_override or join(preprocessed_dir, "nnUNetPlans.json")
    if not Path(plans_path).exists():
        raise FileNotFoundError(f"Plans not found: {plans_path}")
    with open(plans_path, "r") as f:
        plans = json.load(f)

    print(f"Loaded dataset: {dataset_json.get('name', dataset_name)}")
    print(f"Configuration: {configuration}")
    return dataset_json, plans


def main() -> None:
    args = parse_args()

    config = DistillationConfig.from_yaml(args.config)
    if config.strategy not in ("dcd", "none"):
        raise ValueError("This public DCD release supports only 'dcd' and 'none'.")

    plans_override = config.student_plans or config.teacher_plans
    dataset_json, plans = load_dataset_info(args.dataset, args.configuration, plans_override)

    print("\nDCD configuration")
    print("=" * 70)
    print(config)
    print("=" * 70)

    trainer = DistillationTrainer(
        plans=plans,
        configuration=args.configuration,
        fold=args.fold,
        dataset_json=dataset_json,
        distillation_config=config,
        device=torch.device(args.device),
    )
    trainer.initialize()

    if args.resume_from:
        checkpoint = Path(args.resume_from)
        if not checkpoint.exists():
            raise FileNotFoundError(f"--resume_from path not found: {checkpoint}")
        trainer.load_checkpoint(str(checkpoint))

    if args.verify_features:
        from nnunetv2.training.distillation.utils import verify_feature_extraction

        success = verify_feature_extraction(
            model=trainer.network,
            input_shape=(1, trainer.num_input_channels, *trainer.configuration_manager.patch_size),
            required_features=trainer.distill_strategy.get_required_features(),
            device=trainer.device,
        )
        if not success:
            raise RuntimeError("Feature extraction verification failed.")

    trainer.run_training()

    if args.skip_eval or config.visualize_only:
        return

    if config.eval_with_best:
        best_checkpoint = Path(trainer.output_folder) / "checkpoint_best.pth"
        if best_checkpoint.exists():
            trainer.load_checkpoint(str(best_checkpoint))
        else:
            print(f"Requested best-checkpoint validation but not found: {best_checkpoint}", file=sys.stderr)

    trainer.perform_actual_validation(save_probabilities=args.export_validation_probabilities)


if __name__ == "__main__":
    main()
