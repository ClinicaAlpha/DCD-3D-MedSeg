"""
Wavelet-based feature distillation (3D), Detail Consistent Distillation (DCD).

The stage losses are averaged across the selected encoder stages.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import DistillationMethod
from .dcd_visualization import DCDFeatureVisualizer


def _build_filter_bank_3d(wavelet: str, *, mode: str = "dec") -> torch.Tensor:
    wave = pywt.Wavelet(wavelet)
    if mode == "dec":
        lo = torch.tensor(wave.dec_lo, dtype=torch.float32)
        hi = torch.tensor(wave.dec_hi, dtype=torch.float32)
    elif mode == "rec":
        lo = torch.tensor(wave.rec_lo, dtype=torch.float32)
        hi = torch.tensor(wave.rec_hi, dtype=torch.float32)
    else:
        raise ValueError(f"Unknown filter mode '{mode}'. Use 'dec' or 'rec'.")

    kernels = []
    for fz in (lo, hi):
        for fy in (lo, hi):
            for fx in (lo, hi):
                kernel = fz[:, None, None] * fy[None, :, None] * fx[None, None, :]
                kernels.append(kernel)

    filters = torch.stack(kernels, dim=0).unsqueeze(1)  # (8, 1, K, K, K)
    return filters


class StageWiseDCDLoss(nn.Module):
    """
    Single-stage wavelet MSE distillation for 3D feature maps.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        wavelet: str = "haar",
        levels: int = 1,
        band: Union[str, List[int], Tuple[int, ...]] = "all",
        visualize: bool = False,
        visualize_dir: Optional[str] = None,
        visualize_every: int = 200,
        visualize_max: int = 50,
        visualize_stages: Union[str, List[int], Tuple[int, ...]] = "all",
        visualize_levels: Union[str, List[int], Tuple[int, ...]] = "all",
        visualize_channel: Union[str, int] = "mean",
        visualize_slice: Union[str, int] = "mid",
        visualize_resize_to_input: bool = False,
        visualize_resize_mode: str = "bilinear",
        visualize_resize_dwt: bool = False,
        visualize_contrast_gain: float = 1.0,
    ):
        super().__init__()

        if levels < 1:
            raise ValueError("levels must be >= 1")

        if student_channels != teacher_channels:
            self.align = nn.Conv3d(
                student_channels, teacher_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.align = None

        dec_filters = _build_filter_bank_3d(wavelet, mode="dec")
        rec_filters = _build_filter_bank_3d(wavelet, mode="rec")
        self.register_buffer("dec_filters", dec_filters)
        self.register_buffer("rec_filters", rec_filters)
        self.pad = dec_filters.shape[-1] - 1
        self.levels = int(levels)
        self.subband_indices = self._resolve_subbands(band)
        self.visualizer = DCDFeatureVisualizer(
            enabled=bool(visualize),
            output_dir=visualize_dir,
            every=int(visualize_every),
            max_images=int(visualize_max),
            stages=visualize_stages,
            levels=visualize_levels,
            channel=visualize_channel,
            slice_index=visualize_slice,
            num_levels=self.levels,
            resize_to_input=bool(visualize_resize_to_input),
            resize_mode=str(visualize_resize_mode),
            resize_dwt=bool(visualize_resize_dwt),
            contrast_gain=float(visualize_contrast_gain),
        )

    @staticmethod
    def _resolve_subbands(band: Union[str, List[int], Tuple[int, ...]]) -> List[int]:
        if isinstance(band, (list, tuple)):
            indices = [int(i) for i in band]
        else:
            lowered = str(band).lower()
            if lowered in ("all", "*"):
                indices = list(range(8))
            elif lowered in ("low", "ll", "lll"):
                indices = [0]
            elif lowered in ("mid", "mf", "middle"):
                indices = [1, 2, 4]
            elif lowered in ("high", "hf"):
                indices = list(range(1, 8))
            else:
                raise ValueError(f"Unknown band selection '{band}'. Use all/low/high or a list.")

        cleaned: List[int] = []
        for idx in indices:
            if idx < 0:
                idx = 8 + idx
            if idx < 0 or idx > 7:
                raise ValueError(f"Subband index {idx} out of range [0..7]")
            cleaned.append(idx)
        return sorted(set(cleaned))

    def _dwt3_single(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 5:
            raise ValueError("Wavelet distillation expects 3D features: (B, C, D, H, W).")

        b, c, d, h, w = features.shape
        filters = self.dec_filters.to(dtype=features.dtype, device=features.device)
        flat = features.view(b * c, 1, d, h, w)
        coeffs = F.conv3d(flat, filters, stride=2, padding=self.pad)
        coeffs = coeffs.view(b, c, 8, coeffs.shape[-3], coeffs.shape[-2], coeffs.shape[-1])
        return coeffs

    def _idwt3_single(self, coeffs: torch.Tensor, out_shape: Tuple[int, int, int]) -> torch.Tensor:
        if coeffs.dim() != 6:
            raise ValueError("Wavelet coefficients must be (B, C, 8, D, H, W).")
        b, c, _, d, h, w = coeffs.shape
        filters = self.rec_filters.to(dtype=coeffs.dtype, device=coeffs.device)
        flat = coeffs.view(b * c, 8, d, h, w)
        recon = F.conv_transpose3d(flat, filters, stride=2, padding=self.pad)
        recon = recon.view(b, c, recon.shape[-3], recon.shape[-2], recon.shape[-1])
        out_d, out_h, out_w = out_shape
        return recon[..., :out_d, :out_h, :out_w]

    def _dwt3_multilevel(
        self, features: torch.Tensor
    ) -> List[Tuple[torch.Tensor, Tuple[int, int, int], torch.Tensor]]:
        coeffs_per_level: List[Tuple[torch.Tensor, Tuple[int, int, int], torch.Tensor]] = []
        current = features
        for _ in range(self.levels):
            current_shape = (current.shape[-3], current.shape[-2], current.shape[-1])
            coeffs = self._dwt3_single(current)
            coeffs_per_level.append((coeffs, current_shape, current))
            current = coeffs[:, :, 0, ...]
        return coeffs_per_level

    def forward(
        self,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        stage_idx: Optional[int] = None,
        input_data: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del target  # unused
        if self.align is not None:
            student_feat = self.align(student_feat)

        student_levels = self._dwt3_multilevel(student_feat)
        teacher_levels = self._dwt3_multilevel(teacher_feat)

        self.visualizer.start_step()
        losses: List[torch.Tensor] = []
        teacher_after_level0: Optional[torch.Tensor] = None
        student_after_level0: Optional[torch.Tensor] = None
        deepest_level_idx = max(0, len(teacher_levels) - 1)
        for level_idx, ((s_coeffs, s_shape, s_in), (t_coeffs, t_shape, t_in)) in enumerate(
            zip(student_levels, teacher_levels)
        ):
            if s_shape != t_shape:
                raise ValueError("Student/teacher feature shapes must match at each level.")
            if self.subband_indices != list(range(8)):
                mask = torch.zeros_like(s_coeffs)
                mask[:, :, self.subband_indices, ...] = 1
                s_coeffs = s_coeffs * mask
                t_coeffs = t_coeffs * mask
            s_recon = self._idwt3_single(s_coeffs, s_shape)
            t_recon = self._idwt3_single(t_coeffs, t_shape)
            if level_idx == 0:
                teacher_after_level0 = t_recon
                student_after_level0 = s_recon
            with torch.no_grad():
                if level_idx == deepest_level_idx:
                    self.visualizer.maybe_save_teacher_triplet(
                        teacher_feat,
                        t_coeffs,
                        teacher_after_level0 if teacher_after_level0 is not None else t_recon,
                        student_before=student_feat,
                        student_dwt=s_coeffs,
                        student_after=student_after_level0 if student_after_level0 is not None else s_recon,
                        input_data=input_data,
                        level_idx=level_idx,
                        stage_idx=stage_idx,
                    )
            losses.append(F.mse_loss(s_recon, t_recon))

        loss = torch.stack(losses).mean()
        return loss, {"wavelet": loss.detach()}


class DetailConsistentDistillation(DistillationMethod, nn.Module):
    """
    Stage-wise wavelet MSE distillation wrapper (mean across stages).
    """

    supports_stagewise: bool = True

    def __init__(self, **config):
        DistillationMethod.__init__(self, **config)
        nn.Module.__init__(self)

        required = ["student_channels", "teacher_channels"]
        for key in required:
            if key not in config:
                raise ValueError(f"DetailConsistentDistillation requires '{key}' in config")

        student_channels = config["student_channels"]
        teacher_channels = config["teacher_channels"]

        student_channels = (
            list(student_channels)
            if isinstance(student_channels, (list, tuple))
            else [student_channels]
        )
        teacher_channels = (
            list(teacher_channels)
            if isinstance(teacher_channels, (list, tuple))
            else [teacher_channels]
        )

        if len(student_channels) != len(teacher_channels):
            raise ValueError("student_channels and teacher_channels must have the same length")

        layer_indices = self._parse_indices(config.get("layer_indices"), len(student_channels))
        self.layer_indices: List[int] = layer_indices

        base_kwargs = dict(
            wavelet=config.get("wavelet", "haar"),
            levels=int(config.get("levels", 1)),
            band=config.get("band", "all"),
            visualize=bool(config.get("visualize", config.get("sample", False))),
            visualize_dir=config.get("visualize_dir", config.get("sample_dir")),
            visualize_every=int(config.get("visualize_every", config.get("sample_every", 200))),
            visualize_max=int(config.get("visualize_max", config.get("sample_max", 50))),
            visualize_stages=config.get("visualize_stages", config.get("sample_stages", "all")),
            visualize_levels=config.get("visualize_levels", config.get("sample_levels", "all")),
            visualize_channel=config.get("visualize_channel", config.get("sample_channel", "mean")),
            visualize_slice=config.get("visualize_slice", config.get("sample_slice", "mid")),
            visualize_resize_to_input=bool(config.get("visualize_resize_to_input", False)),
            visualize_resize_mode=str(config.get("visualize_resize_mode", "bilinear")),
            visualize_resize_dwt=bool(config.get("visualize_resize_dwt", False)),
            visualize_contrast_gain=float(config.get("visualize_contrast_gain", 1.0)),
        )

        self.distill_modules = nn.ModuleDict()
        for idx in self.layer_indices:
            module = StageWiseDCDLoss(
                student_channels=student_channels[idx],
                teacher_channels=teacher_channels[idx],
                **base_kwargs,
            )
            self.distill_modules[str(idx)] = module

    @staticmethod
    def _parse_indices(
        raw_indices: Optional[Union[int, str, List[int], Tuple[int, ...]]], num_stages: int
    ) -> List[int]:
        if raw_indices is None:
            indices: List[int] = [num_stages - 1]
        elif isinstance(raw_indices, (int,)):
            indices = [raw_indices]
        elif isinstance(raw_indices, str):
            lowered = raw_indices.lower()
            if lowered in ("all", "*"):
                indices = list(range(num_stages))
            else:
                indices = [int(raw_indices)]
        else:
            indices = [int(i) for i in raw_indices]

        processed: List[int] = []
        for idx in indices:
            if idx < 0:
                idx = num_stages + idx
            if idx < 0 or idx >= num_stages:
                raise ValueError(f"layer index {idx} out of bounds for {num_stages} stages")
            processed.append(idx)
        return sorted(set(processed))

    def get_stage_indices(self) -> List[int]:
        return list(self.layer_indices)

    def compute_stage_loss(
        self,
        stage_idx: int,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
        target: torch.Tensor,
        student_output: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del target, student_output  # unused on purpose
        input_data = kwargs.get("input_data")
        if str(stage_idx) not in self.distill_modules:
            raise ValueError(f"Stage {stage_idx} not configured for DetailConsistentDistillation")

        module = self.distill_modules[str(stage_idx)]
        stage_loss, stage_loss_dict = module(
            student_feat,
            teacher_feat,
            None,
            stage_idx=stage_idx,
            input_data=input_data,
        )
        return stage_loss, stage_loss_dict

    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        target: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        student_output = kwargs.get("student_output")
        input_data = kwargs.get("input_data")
        if not self.layer_indices:
            raise ValueError("No encoder stages selected for DetailConsistentDistillation")

        total_loss: Optional[torch.Tensor] = None
        loss_dict: Dict[str, float] = {}
        stage_count = len(self.layer_indices)
        stage_scale = 1.0 / stage_count if stage_count > 0 else 1.0

        for idx in self.layer_indices:
            key = f"stage{idx}"
            student_feat = student_features.get(key)
            teacher_feat = teacher_features.get(key)

            if student_feat is None or teacher_feat is None:
                raise ValueError(
                    f"DetailConsistentDistillation requires '{key}' features. "
                    f"Student keys: {list(student_features.keys())}, "
                    f"Teacher keys: {list(teacher_features.keys())}"
                )

            stage_loss, stage_loss_dict = self.compute_stage_loss(
                idx,
                student_feat,
                teacher_feat,
                target,
                student_output=student_output,
                input_data=input_data,
            )
            stage_loss = stage_loss * stage_scale
            if total_loss is None:
                total_loss = stage_loss
            else:
                total_loss = total_loss + stage_loss

            for name, value in stage_loss_dict.items():
                value_item = value.item() if isinstance(value, torch.Tensor) else float(value)
                loss_dict[f"stage{idx}_{name}"] = value_item * stage_scale

        if total_loss is None:
            total_loss = torch.zeros((), device=next(iter(student_features.values())).device)
        return total_loss, loss_dict

    def get_required_features(self) -> Dict[str, str]:
        return {f"stage{idx}": f"encoder.stages[{idx}]" for idx in self.layer_indices}

    def to(self, device: torch.device) -> "DetailConsistentDistillation":
        self.distill_modules = self.distill_modules.to(device)
        return self


__all__ = ["StageWiseDCDLoss", "DetailConsistentDistillation"]
