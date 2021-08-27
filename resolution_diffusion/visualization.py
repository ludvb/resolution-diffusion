import numpy as np
import torch

from .common import interpolate2d, compute_scale_factors
from .model import Model


def sampling(
    model,
    incremental_scale: float,
    dataset: torch.utils.data.Dataset,
    device: torch.device = None,
    num_samples: int = 32,
):
    scale_factors = compute_scale_factors(dataset, incremental_scale)
    samples = torch.stack(
        [dataset[i][0] for i in np.random.choice(len(dataset), size=(num_samples,))]
    )
    samples_interp = interpolate2d(
        samples, scale_factors[-1][None, None], padding_mode="zeros"
    )
    samples_mask = interpolate2d(
        torch.ones_like(samples),
        scale_factors.unsqueeze(0),
        padding_mode="zeros",
    )

    samples = [samples_interp[:, -1:].reshape(-1, *samples_interp.shape[-3:])]
    for mask in samples_mask[:, :-1].transpose(0, 1).flip(0):
        with torch.no_grad():
            x = model(samples[-1].to(device)).sample().cpu()
        x = x.clamp(-1.0, 1.0)
        x = x * mask
        samples.append(x)
    samples = torch.stack(samples)
    samples = (samples + 1.0) / 2
    samples = samples[
        torch.linspace(0, samples.shape[0] - 1, 10).round().long().unique()
    ]

    return samples


def super_resolution(
    model,
    incremental_scale: float,
    dataset: torch.utils.data.Dataset,
    device: torch.device = None,
    num_samples: int = 32,
):
    scale_factors = compute_scale_factors(dataset, incremental_scale)
    data = torch.stack(
        [dataset[i][0] for i in np.random.choice(len(dataset), size=(num_samples,))]
    )

    starting_rank = int(np.ceil(np.log(4) / np.log(incremental_scale)))
    starting_scale = scale_factors[starting_rank]

    sample0 = interpolate2d(data, starting_scale[None, None], padding_mode="zeros")
    sample0 = sample0.squeeze(1)
    sample_masks = interpolate2d(
        torch.ones_like(data),
        scale_factors[:starting_rank].unsqueeze(0),
        padding_mode="zeros",
    )

    samples = [sample0]
    for mask in sample_masks.transpose(0, 1).flip(0):
        with torch.no_grad():
            x = model(samples[-1].to(device)).sample().cpu()
        x = x.clamp(-1.0, 1.0)
        x = x * mask
        samples.append(x)
    samples = torch.stack(samples)
    samples = (samples + 1.0) / 2
    samples = samples[
        torch.linspace(0, samples.shape[0] - 1, 10).round().long().unique()
    ]
    samples = torch.cat([samples, data.unsqueeze(0)])

    return samples
