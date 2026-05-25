# DCD Distillation Module

This package contains the public Detail Consistent Distillation implementation
used by DCD-3D-MedSeg.

Supported public strategies:

- `dcd`: stage-wise 3D wavelet feature distillation.
- `none`: baseline placeholder for disabling distillation.

The recommended entry point is:

```bash
dcd_train --config configs/dcd_brats2024_gli.yaml \
  --dataset Dataset226_BraTS2024-BraTS-GLI \
  --fold 3 \
  --configuration 3d_fullres
```

See the repository root `README.md` for installation, data preparation, and
experiment commands.
