import os
import itertools as it

import numpy as np
import torch


def first_unique_filename(x):
    return next(
        x
        for x in (
            x + suffix for suffix in it.chain([""], (f".{i}" for i in it.count(1)))
        )
        if not os.path.exists(x)
    )


def add_noise(img: torch.Tensor, scale: float = 0.05) -> torch.Tensor:
    return ((1 - 2 * scale) * img + 0.05 * torch.randn_like(img)).clamp(-1.0, 1.0)


def compute_scale_factors(dataset: torch.utils.data.Dataset, incremental_scale: float):
    incremental_scale = incremental_scale
    img_shape = dataset[0][0].shape[1:]
    data_dim = np.max(img_shape)
    num_steps = int(np.ceil(np.log(data_dim) / np.log(incremental_scale)))
    scale_factors = torch.tensor(
        [1 / incremental_scale ** k for k in range(0, num_steps)]
    )
    return scale_factors


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
def interpolate2d(
    x: torch.Tensor,
    scale_factors: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    assert x.ndim == 4
    assert scale_factors.ndim == 2

    x, _ = torch.broadcast_tensors(x[:, None], scale_factors[..., None, None, None])
    _, scale_factors = torch.broadcast_tensors(x[:, :, 0, 0, 0], scale_factors)
    transform = (1 / scale_factors[..., None, None]) * torch.eye(3, device=x.device)[:2]

    x_ = x.flatten(0, 1)
    transform_ = transform.flatten(0, 1)
    grid = torch.nn.functional.affine_grid(
        transform_, size=x_.shape, align_corners=False
    )
    x_ = torch.nn.functional.grid_sample(
        x_, grid, mode=mode, padding_mode=padding_mode, align_corners=False
    )

    return x_.reshape_as(x)
