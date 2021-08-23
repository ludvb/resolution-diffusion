import os
import itertools as it
from datetime import datetime
from numbers import Number
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as image_transforms
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm import tqdm


dataset = MNIST(
    "./data",
    train=True,
    download=True,
    transform=image_transforms.Compose(
        [
            image_transforms.ToTensor(),
            image_transforms.Normalize(
                mean=torch.tensor([0.5]), std=torch.tensor([0.5])
            ),
        ]
    ),
)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    drop_last=True,
    num_workers=len(os.sched_getaffinity(0)),
    persistent_workers=True,
    shuffle=True,
)

incremental_scale = 1.25
data_dim = np.max(next(iter(dataloader))[0].shape[2:])
num_steps = int(np.ceil(np.log(data_dim / 2) / np.log(incremental_scale))) + 1
scale_factors = [1 / incremental_scale ** k for k in range(0, num_steps)]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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
def interpolate_samples(
    x: torch.Tensor,
    scale_factors: list[float],
    pad_value: float = 0.0,
) -> torch.Tensor:
    return center_align(
        [
            torch.nn.functional.interpolate(
                x,
                scale_factor=[k, k],
                mode="bilinear",
                align_corners=False,
                recompute_scale_factor=False,
            )
            for k in scale_factors
        ],
        target_shape=list(x.shape),
        pad_value=pad_value,
    )


class Model(torch.jit.ScriptModule):
    def __init__(self, incremental_scale: float):
        super().__init__()
        self.incremental_scale = incremental_scale
        self._forward = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 8, 3, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 8, 3, padding=1),
            torch.nn.LeakyReLU(0.1),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 2, 3, padding=1),
        )

    def forward(
        self, x: torch.Tensor, keep_dims: bool = True
    ) -> torch.distributions.Distribution:
        y = torch.nn.functional.interpolate(
            x,
            scale_factor=[self.incremental_scale, self.incremental_scale],
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        if keep_dims:
            y = view_center(y, x.shape)
        y = self._forward(y)
        mu = y[:, : y.shape[1] // 2]
        sd = y[:, y.shape[1] // 2 :]
        return Normal(torch.tanh(mu), 1e-2 + torch.nn.functional.softplus(sd))


def main():
    model = Model(incremental_scale=incremental_scale)
    model = model.to(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-4)

    summary_writer = SummaryWriter(
        f"/tmp/resolution-diffusion/{datetime.now().isoformat()}"
    )

    epdf = torch.stack(
        [dataset[i][0] for i in np.random.choice(len(dataset), size=1024)]
    )
    epdf_interp = interpolate_samples(epdf, scale_factors, pad_value=-1.0)
    epdf_masks = interpolate_samples(
        torch.ones_like(epdf),
        scale_factors,
    )
    epdf_masks = epdf_masks.bool()
    assert (
        len(np.unique(epdf[0], axis=0)) < 64
    ), "Lowest resolution is not densely sampled"

    global_step = 0
    for epoch in it.count(1):
        model.train(False)
        sample_idxs = np.random.choice(epdf.size(1), size=8)
        samples = [epdf_interp[-1, sample_idxs]]
        for mask in epdf_masks[:-1, sample_idxs].flip(0):
            with torch.no_grad():
                x = model(samples[-1].to(device), keep_dims=True).sample().cpu()
            x = x * mask
            samples.append(x)
        samples = center_align(
            samples, target_shape=np.max([x.shape for x in samples], axis=0)
        )
        samples = samples.transpose(-2, -1)
        summary_writer.add_image(
            "samples",
            make_grid(samples.reshape(-1, *samples.shape[2:]), nrow=samples.shape[1]),
            global_step=global_step,
        )

        model.train(True)
        losses = []
        progress = tqdm(dataloader)
        for x, _ in progress:
            x = x.to(device=device)
            data_masks = torch.ones_like(x)

            x = interpolate_samples(x, scale_factors)
            data_masks = interpolate_samples(data_masks, scale_factors)

            y = model(x[1:].reshape(-1, *x.shape[2:]), keep_dims=True)
            lp = y.log_prob(x[:-1].reshape(-1, *x.shape[2:]))
            lp = lp * data_masks[:-1].reshape(-1, *data_masks.shape[2:])
            loss = -lp.sum((1, 2, 3)).mean()
            # loss = -lp.mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            progress.set_description(f"LOSS = {np.mean(losses):.2e}")
            summary_writer.add_scalar("loss", loss, global_step=global_step)

            global_step += 1

        print(
            "  //  ".join(
                [
                    f"EPOCH {epoch:d}",
                    f"LOSS = {np.mean(losses):.3e}",
                ]
            )
        )


if __name__ == "__main__":
    main()
