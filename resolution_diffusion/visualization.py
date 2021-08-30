from typing import Optional

import numpy as np
import torch
from moviepy.editor import VideoClip
from scipy.ndimage.morphology import binary_fill_holes

from .common import interpolate2d, compute_scale_factors


def sampling(
    model,
    incremental_scale: float,
    dataset: torch.utils.data.Dataset,
    device: torch.device = None,
    num_samples: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale_factors = compute_scale_factors(dataset, incremental_scale)
    scale_factors = scale_factors.flip(0)
    samples = torch.stack(
        [dataset[i][0] for i in np.random.choice(len(dataset), size=(num_samples,))]
    )
    samples_interp = interpolate2d(
        samples, scale_factors[0][None, None], padding_mode="zeros"
    )
    samples_mask = interpolate2d(
        torch.ones_like(samples),
        scale_factors.unsqueeze(0),
        padding_mode="zeros",
    )

    samples = [samples_interp.reshape(-1, *samples_interp.shape[-3:])]
    for mask in samples_mask[:, 1:].transpose(0, 1):
        with torch.no_grad():
            x = model(samples[-1].to(device)).sample().cpu()
        x = x.clamp(-1.0, 1.0)
        x = x * mask
        samples.append(x)
    samples = torch.stack(samples)

    return samples, scale_factors


def super_resolution(
    model,
    incremental_scale: float,
    dataset: torch.utils.data.Dataset,
    device: torch.device = None,
    num_samples: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_factors = compute_scale_factors(dataset, incremental_scale)
    data = torch.stack(
        [dataset[i][0] for i in np.random.choice(len(dataset), size=(num_samples,))]
    )

    starting_rank = int(np.ceil(np.log(4) / np.log(incremental_scale)))
    scale_factors = scale_factors[:starting_rank].flip(0)

    sample0 = interpolate2d(data, scale_factors[:1, None], padding_mode="zeros")
    sample0 = sample0.squeeze(1)
    sample_masks = interpolate2d(
        torch.ones_like(data),
        scale_factors[1:].unsqueeze(0),
        padding_mode="zeros",
    )

    samples = [sample0]
    for mask in sample_masks.transpose(0, 1):
        with torch.no_grad():
            x = model(samples[-1].to(device)).sample().cpu()
        x = x.clamp(-1.0, 1.0)
        x = x * mask
        samples.append(x)
    samples = torch.stack(samples)

    return samples, scale_factors, data


def make_animation_frames(
    samples: torch.Tensor, scale_factors: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if scale_factors is None:
        border = torch.as_tensor(binary_fill_holes(samples.cpu().numpy() != 0))
        border_pixels = border.sum((2, 3, 4)).float().mean(1)
        scale_factors = border_pixels / np.prod(samples.shape[2:])
    frames = interpolate2d(
        samples.flatten(1, 2), 1 / scale_factors.unsqueeze(1), mode="nearest"
    )
    frames = frames.squeeze(1).reshape_as(samples)
    return frames


def make_animation_clip(
    frames: torch.Tensor,
    animation_duration: int = 5,
    retain_first_frame_for: int = 1,
    retain_last_frame_for: int = 1,
) -> VideoClip:
    if frames.shape[0] == 1:
        frames = frames.repeat_interleave(3, dim=1)
    frames = (frames + 1.0) / 2 * 255
    frames = frames.permute(0, 2, 3, 1)
    frames = frames.cpu().numpy().astype(np.uint8)

    def make_frame(t):
        t = frames.shape[0] * (t - retain_first_frame_for) / animation_duration
        if t < 0:
            return frames[0]
        if t > frames.shape[0] - 1:
            return frames[-1]
        t1 = int(np.floor(t))
        t2 = int(np.ceil(t))
        w = np.sin(np.pi / 2 * (t - t1)) ** 2
        return (1 - w) * frames[t1] + w * frames[t2]

    return VideoClip(
        make_frame,
        duration=animation_duration + retain_first_frame_for + retain_last_frame_for,
    )
