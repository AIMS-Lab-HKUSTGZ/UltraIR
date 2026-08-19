"""GPU-friendly spectral augmentation used by multitask pretraining."""

from __future__ import annotations

from typing import Any

import torch


def augment_signals(
    signals: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the final pretraining augmentations and return masked positions."""
    if signals.ndim != 2:
        raise ValueError(f"Expected signals [B, L], got {tuple(signals.shape)}")

    output = signals.clone()
    batch_size, signal_len = output.shape
    device, dtype = output.device, output.dtype
    masked_positions = torch.zeros_like(output, dtype=torch.bool)

    noise_cfg = config.get("add_noise")
    if noise_cfg and float(noise_cfg.get("p", 0.2)) > 0:
        apply = torch.rand(batch_size, 1, device=device) < float(noise_cfg.get("p", 0.2))
        snr_range = noise_cfg.get("target_snr_db", [2, 10])
        low, high = int(snr_range[0]), int(snr_range[1])
        if high <= low:
            raise ValueError("augment.add_noise.target_snr_db must have high > low")
        snr_db = torch.randint(low, high, (batch_size, 1), device=device).to(dtype)
        signal_power = output.square().mean(dim=1, keepdim=True)
        noise_power = signal_power / torch.pow(output.new_tensor(10.0), snr_db / 10.0)
        noise = torch.randn_like(output) * noise_power.clamp_min(0).sqrt()
        output = torch.where(apply, output + noise, output)

    shift_lr_cfg = config.get("shift_lr")
    if shift_lr_cfg and float(shift_lr_cfg.get("p", 0.2)) > 0:
        apply = torch.rand(batch_size, 1, device=device) < float(shift_lr_cfg.get("p", 0.2))
        shift_range = shift_lr_cfg.get("shift_p", [0.01, 0.05])
        fractions = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(
            float(shift_range[0]), float(shift_range[1])
        )
        shifts = (fractions * signal_len).long().clamp(0, max(0, signal_len - 1))
        positions = torch.arange(signal_len, device=device).view(1, signal_len)
        shift_right = torch.rand(batch_size, 1, device=device) > 0.5
        sources = torch.where(shift_right, positions - shifts, positions + shifts)
        valid = torch.where(
            shift_right,
            positions >= shifts,
            positions < signal_len - shifts,
        )
        shifted = output.gather(1, sources.clamp(0, max(0, signal_len - 1)))
        shifted = shifted.masked_fill(~valid, 0.0)
        output = torch.where(apply, shifted, output)

    shift_ud_cfg = config.get("shift_ud")
    if shift_ud_cfg and float(shift_ud_cfg.get("p", 0.2)) > 0:
        apply = (torch.rand(batch_size, 1, device=device) < float(shift_ud_cfg.get("p", 0.2))).to(dtype)
        shift_range = shift_ud_cfg.get("shift_p", [0.01, 0.05])
        fractions = torch.empty(batch_size, 1, device=device, dtype=dtype).uniform_(
            float(shift_range[0]), float(shift_range[1])
        )
        direction = torch.where(
            torch.rand(batch_size, 1, device=device) > 0.5,
            output.new_tensor(1.0),
            output.new_tensor(-1.0),
        )
        output = output + apply * direction * output.max(dim=1, keepdim=True).values * fractions

    mask_cfg = config.get("mask_zeros")
    if mask_cfg and float(mask_cfg.get("p", 0.2)) > 0:
        apply = torch.rand(batch_size, device=device) < float(mask_cfg.get("p", 0.2))
        num_blocks = mask_cfg.get("num_blocks", [3, 8])
        min_blocks, max_blocks = max(0, int(num_blocks[0])), max(0, int(num_blocks[1]))
        max_blocks = max(min_blocks, max_blocks)
        block_counts = torch.randint(min_blocks, max_blocks + 1, (batch_size,), device=device)
        block_counts = torch.where(apply, block_counts, torch.zeros_like(block_counts))

        wn_range = mask_cfg.get("wavenumber_range", [400, 4000])
        wn_span = max(abs(float(wn_range[1]) - float(wn_range[0])), 1.0)
        points_per_cm = float(max(signal_len - 1, 1)) / wn_span
        positions = torch.arange(signal_len, device=device).view(1, signal_len)
        normal = mask_cfg.get("block_cm", [20, 200])
        small = mask_cfg.get("small_block_cm", [10, 50])
        large = mask_cfg.get("large_block_cm", [100, 300])
        small_prob = float(mask_cfg.get("small_block_prob", 0.6))
        large_prob = float(mask_cfg.get("large_block_prob", 0.2))

        for block_index in range(max_blocks):
            active = block_index < block_counts
            if not bool(active.any()):
                continue
            selector = torch.rand(batch_size, device=device)
            lows = torch.full((batch_size,), float(normal[0]), device=device, dtype=dtype)
            highs = torch.full((batch_size,), float(normal[1]), device=device, dtype=dtype)
            use_small = selector < small_prob
            use_large = selector >= 1.0 - large_prob
            lows = torch.where(use_small, lows.new_tensor(float(small[0])), lows)
            highs = torch.where(use_small, highs.new_tensor(float(small[1])), highs)
            lows = torch.where(use_large, lows.new_tensor(float(large[0])), lows)
            highs = torch.where(use_large, highs.new_tensor(float(large[1])), highs)
            lengths_cm = lows + torch.rand(batch_size, device=device, dtype=dtype) * (highs - lows)
            lengths = (lengths_cm * points_per_cm).round().long().clamp(1, signal_len)
            max_start = (signal_len - lengths).clamp_min(0)
            starts = (torch.rand(batch_size, device=device) * (max_start + 1)).floor().long()
            block_mask = (positions >= starts[:, None]) & (positions < (starts + lengths)[:, None])
            masked_positions |= block_mask & active[:, None]

        output = output.masked_fill(masked_positions, 0.0)

    return output, masked_positions
