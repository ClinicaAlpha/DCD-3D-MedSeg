# DCD-3D-MedSeg

Detail Consistent Distillation (DCD) for efficient 3D medical image segmentation.

This repository accompanies our MICCAI 2026 accepted paper, "Detail Consistent
Stage-Wise Distillation for Efficient 3D MRI Segmentation."

This repository is an nnU-Net v2.6.2 fork with the public DCD implementation
used in our BraTS 2024 GLI and ISLES 2022 experiments.


## Results

| Dataset | Method | mDice (%) | HD95 (mm) | NSD (%) |
| --- | --- | ---: | ---: | ---: |
| BraTS 2024 GLI | DCD | 68.51 | 6.25 | 81.34 |
| ISLES 2022 | DCD | 73.95 | 12.95 | 83.47 |

The released YAML files reproduce the method settings used for these rows.

## Installation

Create and activate a Python environment with CUDA-compatible PyTorch, then run:

```bash
git clone https://github.com/ClinicaAlpha/DCD-3D-MedSeg.git
cd DCD-3D-MedSeg
pip install -e .
```

Set nnU-Net paths before preprocessing or training:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

## Data Preparation

Download data from the official sources and follow their access terms.

- BraTS 2024 GLI: https://www.synapse.org/Synapse:syn53708249
- ISLES 2022: https://doi.org/10.5281/zenodo.7153326

Prepare BraTS 2024 GLI:

```bash
dcd_prepare_brats2024_gli \
  --brats_dir /path/to/BraTS2024_GLI/training_data

nnUNetv2_plan_and_preprocess \
  -d 226 \
  -c 3d_fullres \
  -pl ResEncUNetPlanner \
  --verify_dataset_integrity
```

Prepare ISLES 2022:

```bash
dcd_prepare_isles2022_to_flair \
  --isles_dir /path/to/ISLES-2022 \
  --output_dir /path/to/ISLES-2022_resampled_to_FLAIR

dcd_prepare_isles2022 \
  --prepared_dir /path/to/ISLES-2022_resampled_to_FLAIR

nnUNetv2_plan_and_preprocess \
  -d 229 \
  -c 3d_fullres \
  -pl ResEncUNetPlanner \
  --verify_dataset_integrity
```

## Training

Train a full teacher with standard nnU-Net first. The DCD configs expect the
teacher checkpoints and plans under `$nnUNet_results`.

BraTS 2024 GLI teacher example:

```bash
nnUNetv2_train 226 3d_fullres 3 -tr nnUNetTrainer -p nnUNetResEncUNetPlans
```

ISLES 2022 teacher example:

```bash
nnUNetv2_train 229 3d_fullres 0 -tr nnUNetTrainer -p nnUNetResEncUNetPlans
```

Train DCD students:

```bash
dcd_train \
  --config configs/dcd_brats2024_gli.yaml \
  --dataset Dataset226_BraTS2024-BraTS-GLI \
  --fold 3 \
  --configuration 3d_fullres

dcd_train \
  --config configs/dcd_isles2022.yaml \
  --dataset Dataset229_ISLES2022 \
  --fold 0 \
  --configuration 3d_fullres
```

If your teacher checkpoint or plans are stored under a different result folder,
edit `teacher_checkpoint` and `teacher_plans` in the YAML before launching DCD
training.

The two public configs use `strategy: dcd`, `wavelet: db4`,
`levels: 3`, `band: [6, 5, 4, 3, 2, 1]`, `layer_indices: all`, and a
student channel reduction factor of 4.

## License

This project is released under the Apache License 2.0 because it is distributed
as an nnU-Net fork. See `NOTICE` for attribution and modification notes.
