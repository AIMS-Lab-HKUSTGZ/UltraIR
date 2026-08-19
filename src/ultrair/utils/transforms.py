"""Spectral transforms shared by downstream training and pretraining."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _parts(sample: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    return sample["signal"], sample["label"], {
        key: value for key, value in sample.items() if key not in {"signal", "label"}
    }


class AddNoise:
    def __init__(
        self,
        p: float = 0.2,
        target_snr_db=(2, 10),
        mean_noise: float = 0.0,
        legacy_semantics: bool = False,
    ) -> None:
        self.p, self.snr, self.mean_noise = float(p), target_snr_db, float(mean_noise)
        self.legacy_semantics = bool(legacy_semantics)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        signal, label, extras = _parts(sample)
        if np.random.random() < self.p:
            if self.legacy_semantics:
                snr = np.random.randint(int(self.snr[0]), int(self.snr[1]))
            else:
                snr = np.random.uniform(float(self.snr[0]), float(self.snr[1]))
            power = float(np.mean(np.square(signal)))
            if not self.legacy_semantics:
                power = max(power, 1e-12)
            noise_power = 10 ** ((10 * np.log10(power) - snr) / 10)
            noise_shape = (1, signal.shape[-1]) if self.legacy_semantics else signal.shape
            signal = signal + np.random.normal(
                self.mean_noise, np.sqrt(noise_power), noise_shape
            )
        return {"signal": signal, "label": label, **extras}


class ShiftLR:
    def __init__(self, p: float = 0.2, shift_p=(0.01, 0.05)) -> None:
        self.p, self.shift_p = float(p), shift_p

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        signal, label, extras = _parts(sample)
        if np.random.random() < self.p:
            amount = int(np.random.uniform(*self.shift_p) * signal.shape[-1])
            if amount > 0:
                shifted = np.zeros_like(signal)
                if np.random.random() > 0.5:
                    shifted[..., amount:] = signal[..., :-amount]
                else:
                    shifted[..., :-amount] = signal[..., amount:]
                signal = shifted
        return {"signal": signal, "label": label, **extras}


class ShiftUD:
    def __init__(
        self, p: float = 0.2, shift_p=(0.01, 0.05), legacy_semantics: bool = False
    ) -> None:
        self.p, self.shift_p = float(p), shift_p
        self.legacy_semantics = bool(legacy_semantics)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        signal, label, extras = _parts(sample)
        if np.random.random() < self.p:
            peak = np.max(signal) if self.legacy_semantics else np.max(np.abs(signal))
            offset = float(peak) * np.random.uniform(*self.shift_p)
            signal = signal + (offset if np.random.random() > 0.5 else -offset)
        return {"signal": signal, "label": label, **extras}


class Resizer:
    def __init__(self, signal_size: int = 1024, legacy_semantics: bool = False) -> None:
        self.signal_size = int(signal_size)
        self.legacy_semantics = bool(legacy_semantics)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        signal, label, extras = _parts(sample)
        tensor = torch.as_tensor(np.asarray(signal, dtype=np.float32))
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"Expected [L] or [C, L] signal, got {tuple(tensor.shape)}")
        if self.legacy_semantics:
            try:
                import cv2
            except ImportError:
                tensor = F.interpolate(
                    tensor.unsqueeze(0).unsqueeze(2),
                    size=(1, self.signal_size),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).squeeze(1)
            else:
                rows = tensor.numpy()
                resized = [
                    cv2.resize(
                        row[None, :],
                        (self.signal_size, 1),
                        interpolation=cv2.INTER_CUBIC,
                    ).reshape(-1)
                    for row in rows
                ]
                tensor = torch.from_numpy(np.stack(resized).astype(np.float32, copy=False))
        else:
            tensor = F.interpolate(
                tensor.unsqueeze(0), size=self.signal_size, mode="linear", align_corners=False
            ).squeeze(0)
        return {"signal": tensor, "label": label, **extras}


def apply_batched_torch_transforms(
    signals: torch.Tensor,
    transform_cfg: dict | None,
    return_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    if not transform_cfg:
        return (signals, None) if return_mask else signals
    if signals.ndim != 2:
        raise ValueError(f"Expected [B, L] signals, got {tuple(signals.shape)}")
    output = signals.clone()
    mask = None
    noise = transform_cfg.get("add_noise", {}) or {}
    if noise and float(noise.get("p", 0.0)) > 0:
        apply = torch.rand(output.shape[0], 1, device=output.device) < float(noise["p"])
        snr_range = noise.get("target_snr_db", [15, 30])
        snr = torch.empty(output.shape[0], 1, device=output.device).uniform_(
            float(snr_range[0]), float(snr_range[1])
        )
        power = output.square().mean(1, keepdim=True).clamp_min(1e-12)
        noise_std = torch.sqrt(power / torch.pow(10.0, snr / 10.0))
        output = output + apply * torch.randn_like(output) * noise_std
    shift = transform_cfg.get("shift_ud", {}) or {}
    if shift and float(shift.get("p", 0.0)) > 0:
        apply = torch.rand(output.shape[0], 1, device=output.device) < float(shift["p"])
        bounds = shift.get("shift_p", [0.003, 0.04])
        fraction = torch.empty(output.shape[0], 1, device=output.device).uniform_(
            float(bounds[0]), float(bounds[1])
        )
        direction = torch.where(torch.rand_like(fraction) > 0.5, 1.0, -1.0)
        output = output + apply * direction * fraction * output.abs().amax(1, keepdim=True)
    mask_cfg = transform_cfg.get("mask_zeros", {}) or {}
    if mask_cfg and float(mask_cfg.get("p", 0.0)) > 0:
        mask = torch.zeros_like(output, dtype=torch.bool)
        count_range = mask_cfg.get("num_blocks", [3, 7])
        width_range = mask_cfg.get("block_points", [8, 80])
        for row in range(output.shape[0]):
            if torch.rand((), device=output.device) >= float(mask_cfg["p"]):
                continue
            blocks = int(torch.randint(int(count_range[0]), int(count_range[1]) + 1, ()).item())
            for _ in range(blocks):
                width = int(torch.randint(int(width_range[0]), int(width_range[1]) + 1, ()).item())
                width = min(width, output.shape[1])
                start = int(torch.randint(0, output.shape[1] - width + 1, ()).item())
                mask[row, start : start + width] = True
        output = output.masked_fill(mask, 0.0)
    return (output, mask) if return_mask else output
