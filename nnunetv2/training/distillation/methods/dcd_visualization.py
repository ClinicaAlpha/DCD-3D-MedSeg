"""
Visualization helpers for DCD distillation methods.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


class DCDFeatureVisualizer:
    """
    Saves 2D comparisons (teacher before/after IDWT) from 3D feature maps.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        output_dir: Optional[str] = None,
        every: int = 200,
        max_images: int = 50,
        stages: Union[str, List[int], Tuple[int, ...]] = "all",
        levels: Union[str, List[int], Tuple[int, ...]] = "all",
        channel: Union[str, int] = "mean",
        slice_index: Union[str, int] = "mid",
        num_levels: int = 1,
        resize_to_input: bool = False,
        resize_mode: str = "bilinear",
        resize_dwt: bool = False,
        contrast_gain: float = 1.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_dir = output_dir
        self.every = max(1, int(every))
        self.max_images = max(0, int(max_images))
        self.channel = channel
        self.slice_index = slice_index
        self.num_levels = int(num_levels)
        self.resize_to_input = bool(resize_to_input)
        self.resize_mode = str(resize_mode)
        self.resize_dwt = bool(resize_dwt)
        self.contrast_gain = max(0.1, float(contrast_gain))

        if self.num_levels < 1:
            raise ValueError("num_levels must be >= 1")

        if self.enabled:
            self.stages = self._resolve_stages(stages)
            self.levels = self._resolve_levels(levels, self.num_levels)
        else:
            self.stages = []
            self.levels = []

        self._step = 0
        self._saved = 0

    @staticmethod
    def _resolve_levels(
        levels: Union[str, List[int], Tuple[int, ...]],
        num_levels: int,
    ) -> List[int]:
        if isinstance(levels, (list, tuple)):
            indices = [int(i) for i in levels]
        else:
            lowered = str(levels).lower()
            if lowered in ("all", "*"):
                indices = list(range(num_levels))
            else:
                indices = [int(lowered)]

        cleaned: List[int] = []
        for idx in indices:
            if idx < 0:
                idx = num_levels + idx
            if idx < 0 or idx >= num_levels:
                raise ValueError(f"Level index {idx} out of bounds for {num_levels} levels")
            cleaned.append(idx)
        return sorted(set(cleaned))

    @staticmethod
    def _resolve_stages(stages: Union[str, List[int], Tuple[int, ...]]) -> List[int]:
        if isinstance(stages, (list, tuple)):
            indices = [int(i) for i in stages]
        else:
            lowered = str(stages).lower()
            if lowered in ("all", "*"):
                return []
            indices = [int(lowered)]

        cleaned: List[int] = []
        for idx in indices:
            if idx < 0:
                raise ValueError("Negative stage indices are not supported for visualization.")
            cleaned.append(idx)
        return sorted(set(cleaned))

    def start_step(self) -> None:
        self._step += 1

    def maybe_save_teacher_triplet(
        self,
        teacher_before: torch.Tensor,
        teacher_dwt: torch.Tensor,
        teacher_after: torch.Tensor,
        student_before: Optional[torch.Tensor] = None,
        student_dwt: Optional[torch.Tensor] = None,
        student_after: Optional[torch.Tensor] = None,
        input_data: Optional[torch.Tensor] = None,
        *,
        level_idx: int,
        stage_idx: Optional[int],
    ) -> None:
        if not self.enabled:
            return
        if self._saved >= self.max_images:
            return
        if self.every > 1 and (self._step % self.every) != 0:
            return
        if level_idx not in self.levels:
            return
        if stage_idx is not None and self.stages and stage_idx not in self.stages:
            return

        output_root = self.output_dir or os.path.join(os.getcwd(), "dcd_wavelet_samples")
        stage_tag = f"stage{stage_idx}" if stage_idx is not None else "stageX"
        out_dir = os.path.join(output_root, stage_tag, f"level{level_idx}")
        os.makedirs(out_dir, exist_ok=True)

        image_input = None
        target_hw = None
        if input_data is not None:
            image_input = self._to_image_feature(input_data).detach().cpu()
            if self.resize_to_input:
                target_hw = (int(image_input.shape[-2]), int(image_input.shape[-1]))

        image_before = self._to_image_feature(teacher_before).detach().cpu()
        image_dwt_lll = self._to_image_dwt_subband(teacher_dwt, subband_idx=0).detach().cpu()
        image_dwt_hhh = self._to_image_dwt_subband(teacher_dwt, subband_idx=7).detach().cpu()
        image_after = self._to_image_feature(teacher_after).detach().cpu()
        image_student_before = None
        image_student_dwt_hhh = None
        image_student_after = None
        if student_before is not None:
            image_student_before = self._to_image_feature(student_before).detach().cpu()
        if student_dwt is not None:
            image_student_dwt_hhh = self._to_image_dwt_subband(student_dwt, subband_idx=7).detach().cpu()
        if student_after is not None:
            image_student_after = self._to_image_feature(student_after).detach().cpu()
        if target_hw is not None:
            image_before = self._resize_2d(image_before, target_hw)
            image_after = self._resize_2d(image_after, target_hw)
            if image_student_before is not None:
                image_student_before = self._resize_2d(image_student_before, target_hw)
            if image_student_after is not None:
                image_student_after = self._resize_2d(image_student_after, target_hw)
            if self.resize_dwt:
                image_dwt_lll = self._resize_2d(image_dwt_lll, target_hw)
                image_dwt_hhh = self._resize_2d(image_dwt_hhh, target_hw)
                if image_student_dwt_hhh is not None:
                    image_student_dwt_hhh = self._resize_2d(image_student_dwt_hhh, target_hw)

        before_norm = self._apply_contrast(self._normalize(image_before))
        dwt_lll_norm = self._apply_contrast(self._normalize(image_dwt_lll))
        dwt_hhh_norm = self._apply_contrast(self._normalize(image_dwt_hhh))
        after_norm = self._apply_contrast(self._normalize(image_after))

        self._write_array(before_norm, os.path.join(out_dir, f"teacher_before_step{self._step:06d}.png"))
        self._write_array(dwt_lll_norm, os.path.join(out_dir, f"teacher_dwt_LLL_step{self._step:06d}.png"))
        self._write_array(dwt_hhh_norm, os.path.join(out_dir, f"teacher_dwt_HHH_step{self._step:06d}.png"))
        self._write_array(after_norm, os.path.join(out_dir, f"teacher_after_step{self._step:06d}.png"))
        if image_student_before is not None:
            student_before_norm = self._apply_contrast(self._normalize(image_student_before))
            self._write_array(student_before_norm, os.path.join(out_dir, f"student_before_step{self._step:06d}.png"))
        if image_student_dwt_hhh is not None:
            student_dwt_hhh_norm = self._apply_contrast(self._normalize(image_student_dwt_hhh))
            self._write_array(student_dwt_hhh_norm, os.path.join(out_dir, f"student_dwt_HHH_step{self._step:06d}.png"))
        if image_student_after is not None:
            student_after_norm = self._apply_contrast(self._normalize(image_student_after))
            self._write_array(student_after_norm, os.path.join(out_dir, f"student_after_step{self._step:06d}.png"))
        if image_input is not None:
            input_norm = self._apply_contrast(self._normalize(image_input))
            input_path = os.path.join(output_root, f"input_step{self._step:06d}.png")
            if not os.path.exists(input_path):
                self._write_array(input_norm, input_path)
        self._saved += 1

    def _to_image_feature(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() == 4:
            feat = feat.unsqueeze(0)
        if feat.dim() != 5:
            raise ValueError("Expected feature map shape (B, C, D, H, W) or (C, D, H, W)")

        b, c, d, _, _ = feat.shape
        b = min(b, 1)
        if isinstance(self.channel, str) and self.channel == "mean":
            chan = feat[:b].mean(dim=1)  # (B, D, H, W)
        else:
            channel_idx = int(self.channel)
            channel_idx = max(0, min(channel_idx, c - 1))
            chan = feat[:b, channel_idx, ...]

        if isinstance(self.slice_index, str) and self.slice_index == "mid":
            z = d // 2
        else:
            z = int(self.slice_index)
            z = max(0, min(z, d - 1))
        return chan[0, z, ...]  # (H, W)

    def _to_image_dwt_subband(self, coeffs: torch.Tensor, subband_idx: int) -> torch.Tensor:
        if coeffs.dim() != 6:
            raise ValueError("Expected DWT coeff shape (B, C, 8, D, H, W)")
        if subband_idx < 0 or subband_idx > 7:
            raise ValueError("subband_idx must be in [0, 7]")

        _, c, _, _, _, _ = coeffs.shape
        if isinstance(self.channel, str) and self.channel == "mean":
            chan = coeffs[:1].mean(dim=1)  # (1, 8, D, H, W)
            dwt_vol = chan[0, subband_idx, ...].abs()  # (D, H, W)
        else:
            channel_idx = int(self.channel)
            channel_idx = max(0, min(channel_idx, c - 1))
            dwt_vol = coeffs[0, channel_idx, subband_idx, ...].abs()  # (D, H, W)
        d = dwt_vol.shape[0]
        if isinstance(self.slice_index, str) and self.slice_index == "mid":
            z = d // 2
        else:
            z = int(self.slice_index)
            z = max(0, min(z, d - 1))
        return dwt_vol[z, ...]  # (H, W)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        x_min = float(x.min())
        x_max = float(x.max())
        if x_max - x_min < 1e-6:
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min)

    def _apply_contrast(self, x: torch.Tensor) -> torch.Tensor:
        if abs(self.contrast_gain - 1.0) < 1e-6:
            return x
        return torch.clamp((x - 0.5) * self.contrast_gain + 0.5, 0.0, 1.0)

    def _resize_2d(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2] == target_hw[0] and x.shape[-1] == target_hw[1]:
            return x
        xx = x.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
        if self.resize_mode in ("linear", "bilinear", "bicubic", "trilinear"):
            yy = F.interpolate(xx, size=target_hw, mode=self.resize_mode, align_corners=False)
        else:
            yy = F.interpolate(xx, size=target_hw, mode=self.resize_mode)
        return yy[0, 0].to(dtype=x.dtype)

    @staticmethod
    def _center_crop_2d(x: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        h, w = x.shape[-2], x.shape[-1]
        if h == target_h and w == target_w:
            return x
        top = max((h - target_h) // 2, 0)
        left = max((w - target_w) // 2, 0)
        return x[top:top + target_h, left:left + target_w]

    @classmethod
    def _match_spatial_size_three(
        cls, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_h = min(a.shape[-2], b.shape[-2], c.shape[-2])
        target_w = min(a.shape[-1], b.shape[-1], c.shape[-1])
        if target_h <= 0 or target_w <= 0:
            raise ValueError(
                f"Invalid 2D feature size for visualization: "
                f"a={tuple(a.shape)}, b={tuple(b.shape)}, c={tuple(c.shape)}"
            )
        return (
            cls._center_crop_2d(a, target_h, target_w),
            cls._center_crop_2d(b, target_h, target_w),
            cls._center_crop_2d(c, target_h, target_w),
        )

    @classmethod
    def _match_spatial_size_four(
        cls, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_h = min(a.shape[-2], b.shape[-2], c.shape[-2], d.shape[-2])
        target_w = min(a.shape[-1], b.shape[-1], c.shape[-1], d.shape[-1])
        if target_h <= 0 or target_w <= 0:
            raise ValueError(
                f"Invalid 2D feature size for visualization: "
                f"a={tuple(a.shape)}, b={tuple(b.shape)}, c={tuple(c.shape)}, d={tuple(d.shape)}"
            )
        return (
            cls._center_crop_2d(a, target_h, target_w),
            cls._center_crop_2d(b, target_h, target_w),
            cls._center_crop_2d(c, target_h, target_w),
            cls._center_crop_2d(d, target_h, target_w),
        )

    @staticmethod
    def _write_array(image_2d: torch.Tensor, output_path: str) -> None:
        try:
            import numpy as np
        except Exception:
            return

        array_u8 = (image_2d.numpy() * 255.0).clip(0, 255).astype("uint8")
        try:
            import imageio.v2 as imageio

            imageio.imwrite(output_path, array_u8)
            return
        except Exception:
            pass

        try:
            from PIL import Image

            Image.fromarray(array_u8).save(output_path)
        except Exception:
            np.save(output_path.replace(".png", ".npy"), array_u8)
