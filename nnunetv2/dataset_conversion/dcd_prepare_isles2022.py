"""Create nnU-Net Dataset229 metadata for prepared ISLES 2022 data."""
from __future__ import annotations

import argparse
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ISLES 2022 metadata for nnU-Net.")
    parser.add_argument("--prepared_dir", required=True, help="Output directory from dcd_prepare_isles2022_to_flair.")
    parser.add_argument("--dataset_id", type=int, default=229)
    parser.add_argument("--dataset_name", default="ISLES2022")
    return parser.parse_args()


def find_one(folder: Path, patterns: list[str]) -> Path:
    matches = []
    for pattern in patterns:
        matches.extend(folder.glob(pattern))
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one match in {folder} for {patterns}, found {len(matches)}")
    return matches[0]


def case_record(prepared_dir: Path, case_dir: Path) -> dict:
    sessions = [p for p in case_dir.iterdir() if p.is_dir() and p.name.startswith("ses-")]
    if len(sessions) != 1:
        raise RuntimeError(f"Expected one session under {case_dir}, found {[p.name for p in sessions]}")
    session_dir = sessions[0]
    case = case_dir.name
    session = session_dir.name

    flair = find_one(session_dir / "anat", ["*FLAIR*.nii", "*FLAIR*.nii.gz"])
    dwi = find_one(session_dir / "dwi", ["*dwi*.nii", "*dwi*.nii.gz"])
    adc = find_one(session_dir / "dwi", ["*adc*.nii", "*adc*.nii.gz"])
    mask = find_one(prepared_dir / "derivatives" / case / session, ["*_msk.nii", "*_msk.nii.gz"])

    return {"label": str(mask), "images": [str(flair), str(dwi), str(adc)]}


def main() -> None:
    args = parse_args()
    prepared_dir = Path(args.prepared_dir).expanduser().resolve()
    if not prepared_dir.is_dir():
        raise FileNotFoundError(f"Prepared ISLES directory not found: {prepared_dir}")
    if nnUNet_raw is None:
        raise RuntimeError("Set nnUNet_raw before running dataset conversion.")

    dataset_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_dir = join(nnUNet_raw, dataset_name)
    maybe_mkdir_p(dataset_dir)

    dataset = {}
    for case_dir in sorted(p for p in prepared_dir.iterdir() if p.is_dir() and p.name.startswith("sub-strokecase")):
        dataset[case_dir.name] = case_record(prepared_dir, case_dir)

    if not dataset:
        raise RuntimeError(f"No ISLES subject folders found in {prepared_dir}")

    generate_dataset_json(
        dataset_dir,
        {0: "FLAIR", 1: "DWI", 2: "ADC"},
        {"background": 0, "lesion": 1},
        num_training_cases=len(dataset),
        file_ending=".nii.gz",
        regions_class_order=None,
        dataset_name=dataset_name,
        reference="https://doi.org/10.5281/zenodo.7153326",
        license="CC BY 4.0",
        dataset=dataset,
        description="ISLES 2022 stroke lesion segmentation prepared on the FLAIR grid.",
    )
    print(f"Wrote nnU-Net dataset metadata to {dataset_dir}")


if __name__ == "__main__":
    main()
