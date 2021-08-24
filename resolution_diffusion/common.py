import operator
import os
import itertools as it
from functools import reduce
from typing import Any

import torch


def first_unique_filename(x):
    return next(
        x
        for x in (
            x + suffix for suffix in it.chain([""], (f".{i}" for i in it.count(1)))
        )
        if not os.path.exists(x)
    )


@torch.jit.script
def view_center(x: torch.Tensor, target_shape: list[int]) -> torch.Tensor:
    for (i, a), b in zip(list(enumerate(x.shape))[::-1], target_shape[::-1]):
        x = x.transpose(0, i)
        x = x[(a - b) // 2 : (a - b) // 2 + b]
        x = x.transpose(0, i)
    return x


@torch.jit.script
def center_align(
    xs: list[torch.Tensor],
    target_shape: list[int],
    pad_value: float = 0.0,
) -> torch.Tensor:
    ys = []
    for x in xs:
        y = torch.full(target_shape, pad_value).to(x)
        if pad_value is not None:
            y.fill_(pad_value)
        ysliced = view_center(y, x.shape)
        ysliced[...] = x
        ys.append(y)
    return torch.stack(ys)


@torch.jit.script
def interpolate(
    x: torch.Tensor,
    scale_factor: float,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
):
    transform = (1 / scale_factor) * torch.eye(3, device=x.device)[:2]
    grid = torch.nn.functional.affine_grid(
        transform.expand(x.shape[0], -1, -1),
        size=x.shape,
        align_corners=False,
    )
    return torch.nn.functional.grid_sample(
        x, grid, mode=mode, padding_mode=padding_mode, align_corners=False
    )


@torch.jit.script
def interpolate_samples(
    x: torch.Tensor,
    scale_factors: list[float],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    return torch.stack(
        [interpolate(x, k, mode=mode, padding_mode=padding_mode) for k in scale_factors]
    )
