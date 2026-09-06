"""Auditable PyTorch fallback for QuaRot's fast-hadamard-transform API.

This preserves the exact normalized Walsh-Hadamard map used by fake quantization
without compiling a CUDA extension. It is slower and is not a latency result.
"""

from __future__ import annotations

import torch


def hadamard_transform(x: torch.Tensor, scale=1.0) -> torch.Tensor:
    n = x.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"Hadamard dimension must be a positive power of two, got {n}")
    y = x.contiguous()
    width = 1
    while width < n:
        blocks = y.reshape(*y.shape[:-1], -1, 2, width)
        left = blocks[..., 0, :]
        right = blocks[..., 1, :]
        y = torch.stack((left + right, left - right), dim=-2).reshape_as(y)
        width *= 2
    factor = torch.as_tensor(scale, dtype=y.dtype, device=y.device)
    return y * factor
