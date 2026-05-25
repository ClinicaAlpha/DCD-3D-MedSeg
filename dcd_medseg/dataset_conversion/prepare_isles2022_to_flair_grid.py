"""Resample ISLES 2022 image channels and masks onto the FLAIR grid."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ISLES 2022 data on a common FLAIR grid.")
    parser.add_argument("--isles_dir", required=True, help="Extracted ISLES 2022 BIDS root.")
    parser.add_argument("--output_dir", required=True, help="Output root for FLAIR-grid data.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def find_one(folder: Path, patterns: Iterable[str]) -> Path:
    matches = []
    for pattern in patterns:
        matches.extend(folder.glob(pattern))
    matches = sorted(set(matches))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one match in {folder} for {list(patterns)}, found {len(matches)}")
    return matches[0]


def write_image(image: sitk.Image, path: Path, overwrite: bool) -> None:
    if path.is_file() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path))


def resample_to_reference(moving: sitk.Image, reference: sitk.Image, interpolator: int) -> sitk.Image:
    return sitk.Resample(
        moving,
        reference,
        sitk.Transform(),
        interpolator,
        0.0,
        moving.GetPixelID(),
    )


def process_case(case_dir: Path, derivatives_dir: Path, output_root: Path, overwrite: bool) -> None:
    sessions = [p for p in case_dir.iterdir() if p.is_dir() and p.name.startswith("ses-")]
    if len(sessions) != 1:
        raise RuntimeError(f"Expected one session under {case_dir}, found {[p.name for p in sessions]}")
    session_dir = sessions[0]
    case = case_dir.name
    session = session_dir.name

    flair_path = find_one(session_dir / "anat", ["*FLAIR*.nii", "*FLAIR*.nii.gz"])
    dwi_path = find_one(session_dir / "dwi", ["*dwi*.nii", "*dwi*.nii.gz"])
    adc_path = find_one(session_dir / "dwi", ["*adc*.nii", "*adc*.nii.gz"])
    mask_path = find_one(derivatives_dir / case / session, ["*_msk.nii", "*_msk.nii.gz"])

    flair = sitk.ReadImage(str(flair_path))
    dwi = resample_to_reference(sitk.ReadImage(str(dwi_path)), flair, sitk.sitkLinear)
    adc = resample_to_reference(sitk.ReadImage(str(adc_path)), flair, sitk.sitkLinear)
    mask = resample_to_reference(sitk.ReadImage(str(mask_path)), flair, sitk.sitkNearestNeighbor)

    out_case = output_root / case / session
    write_image(flair, out_case / "anat" / f"{case}_{session}_FLAIR.nii.gz", overwrite)
    write_image(dwi, out_case / "dwi" / f"{case}_{session}_dwi.nii.gz", overwrite)
    write_image(adc, out_case / "dwi" / f"{case}_{session}_adc.nii.gz", overwrite)
    write_image(mask, output_root / "derivatives" / case / session / f"{case}_{session}_msk.nii.gz", overwrite)


def main() -> None:
    args = parse_args()
    isles_dir = Path(args.isles_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    derivatives_dir = isles_dir / "derivatives"

    if not isles_dir.is_dir():
        raise FileNotFoundError(f"ISLES directory not found: {isles_dir}")
    if not derivatives_dir.is_dir():
        raise FileNotFoundError(f"ISLES derivatives directory not found: {derivatives_dir}")

    cases = sorted(p for p in isles_dir.iterdir() if p.is_dir() and p.name.startswith("sub-strokecase"))
    if not cases:
        raise RuntimeError(f"No ISLES subject folders found in {isles_dir}")

    failures = []
    for case_dir in cases:
        try:
            process_case(case_dir, derivatives_dir, output_dir, args.overwrite)
            print(f"OK {case_dir.name}")
        except Exception as exc:
            failures.append((case_dir.name, str(exc)))

    if failures:
        print(os.linesep + "Failures:")
        for case, reason in failures:
            print(f"- {case}: {reason}")
        raise SystemExit(1)

    print(f"Prepared FLAIR-grid ISLES data in {output_dir}")


if __name__ == "__main__":
    main()
