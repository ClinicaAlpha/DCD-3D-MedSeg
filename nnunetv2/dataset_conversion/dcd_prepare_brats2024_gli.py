"""Create nnU-Net Dataset226 metadata for BraTS 2024 GLI."""
from __future__ import annotations

import argparse
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


MODALITIES = ("t1n", "t1c", "t2w", "t2f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BraTS 2024 GLI metadata for nnU-Net.")
    parser.add_argument("--brats_dir", required=True, help="Extracted BraTS 2024 GLI training_data directory.")
    parser.add_argument("--dataset_id", type=int, default=226)
    parser.add_argument("--dataset_name", default="BraTS2024-BraTS-GLI")
    return parser.parse_args()


def case_record(case_dir: Path) -> dict:
    case_id = case_dir.name
    label = case_dir / f"{case_id}-seg.nii.gz"
    images = [case_dir / f"{case_id}-{mod}.nii.gz" for mod in MODALITIES]

    missing = [str(p) for p in [label, *images] if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files for {case_id}: {missing}")

    return {"label": str(label), "images": [str(p) for p in images]}


def main() -> None:
    args = parse_args()
    brats_dir = Path(args.brats_dir).expanduser().resolve()
    if not brats_dir.is_dir():
        raise FileNotFoundError(f"BraTS directory not found: {brats_dir}")
    if nnUNet_raw is None:
        raise RuntimeError("Set nnUNet_raw before running dataset conversion.")

    dataset_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_dir = join(nnUNet_raw, dataset_name)
    maybe_mkdir_p(dataset_dir)

    dataset = {}
    for case_dir in sorted(p for p in brats_dir.iterdir() if p.is_dir()):
        dataset[case_dir.name] = case_record(case_dir)

    if not dataset:
        raise RuntimeError(f"No case folders found in {brats_dir}")

    generate_dataset_json(
        dataset_dir,
        {0: "T1N", 1: "T1C", 2: "T2W", 3: "T2F"},
        {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3, "RC": 4},
        num_training_cases=len(dataset),
        file_ending=".nii.gz",
        regions_class_order=None,
        dataset_name=dataset_name,
        reference="https://www.synapse.org/Synapse:syn53708249",
        license="See the BraTS 2024 data access terms on Synapse.",
        dataset=dataset,
        description="BraTS 2024 GLI linked-file dataset for DCD experiments.",
    )
    print(f"Wrote nnU-Net dataset metadata to {dataset_dir}")


if __name__ == "__main__":
    main()
