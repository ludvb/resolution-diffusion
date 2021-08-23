import argparse
import os
import itertools as it
import logging
import sys
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as image_transforms
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def first_unique_filename(x):
    return next(
        x
        for x in (x + suffix for suffix in it.chain("", (f"{i}" for i in it.count(1))))
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


class Model(torch.jit.ScriptModule):
    def __init__(self, incremental_scale: float, num_latent_features: int = 24):
        super().__init__()
        self.incremental_scale = incremental_scale
        self._forward_shared = torch.nn.Sequential(
            torch.nn.Conv2d(1, num_latent_features, 3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(num_latent_features),
            torch.nn.Conv2d(num_latent_features, num_latent_features, 3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(num_latent_features),
            torch.nn.Conv2d(num_latent_features, num_latent_features, 3, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(num_latent_features),
        )
        self._forward_mu = torch.nn.Sequential(
            torch.nn.Conv2d(num_latent_features, num_latent_features, 3, padding=1),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(num_latent_features),
            torch.nn.Conv2d(num_latent_features, 1, 3, padding=1),
        )
        torch.nn.init.constant_(
            self._forward_mu[-1].bias,
            val=0.0,
        )
        torch.nn.init.normal_(
            self._forward_mu[-1].weight,
            mean=0.0,
            std=1e-3 / np.sqrt(num_latent_features),
        )
        self._forward_sd = torch.nn.Sequential(
            torch.nn.Conv2d(num_latent_features, num_latent_features, 3, padding=1),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(num_latent_features),
            torch.nn.Conv2d(num_latent_features, 1, 3, padding=1),
            torch.nn.Softplus(),
        )
        torch.nn.init.constant_(
            self._forward_sd[-2].bias,
            val=np.log(1e-1),
        )
        torch.nn.init.normal_(
            self._forward_sd[-2].weight,
            mean=0.0,
            std=1e-3 / np.sqrt(num_latent_features),
        )

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        x = interpolate(x, self.incremental_scale)
        shared = self._forward_shared(x)
        mu = x + self._forward_mu(shared)
        sd = self._forward_sd(shared)
        return Normal(mu, 1e-4 + sd)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--checkpoint", type=str)
    argparser.add_argument("--incremental-scale", type=float, default=1.25)
    argparser.add_argument("--dataset", type=str, default="MNIST")
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument(
        "--save-path",
        type=str,
        default=f"./resolution-diffusion-{datetime.now().isoformat()}",
    )
    options = argparser.parse_args()

    os.makedirs(os.path.join(options.save_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(options.save_path, "tb"), exist_ok=True)
    logging.basicConfig(
        filename=first_unique_filename(os.path.join(options.save_path, "log")),
        level=logging.DEBUG,
        format="[%(asctime)s]  (%(levelname)s)  %(message)s",
        encoding="utf-8",
    )

    logging.info(" ".join(sys.argv))

    if options.dataset.lower() == "mnist":
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
    else:
        raise RuntimeError(f'Unknown dataset "{options.dataset}"')

    dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        drop_last=True,
        num_workers=len(os.sched_getaffinity(0)),
        persistent_workers=True,
        shuffle=True,
    )

    incremental_scale = options.incremental_scale
    data_dim = np.max(next(iter(dataloader))[0].shape[2:])
    num_steps = int(np.ceil(np.log(data_dim) / np.log(incremental_scale)))
    scale_factors = [1 / incremental_scale ** k for k in range(0, num_steps)]

    model = Model(incremental_scale=incremental_scale)
    model = model.to(device=device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    if options.checkpoint is not None:
        logging.info("Loading checkpoint: %s", options.checkpoint)
        checkpoint = torch.load(options.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])

    summary_writer = SummaryWriter(os.path.join(options.save_path, "tb"))

    viz_samples = torch.stack(
        [dataset[i][0] for i in np.random.choice(len(dataset), size=1024)]
    )
    epdf_interp = interpolate_samples(viz_samples, scale_factors, padding_mode="border")
    epdf_masks = interpolate_samples(
        torch.ones_like(viz_samples), scale_factors, padding_mode="zeros"
    )
    epdf_masks = epdf_masks.bool()
    assert (
        len(np.unique(epdf_interp[-1], axis=0)) < 64
    ), "Lowest resolution is not densely sampled"

    global_step = 0
    for epoch in it.count(1):
        torch.save(
            {"model": model.state_dict(), "optim": optim.state_dict()},
            first_unique_filename(
                os.path.join(options.save_path, "checkpoints", f"epoch-{epoch:04d}.pkl")
            ),
        )

        ## Visualization
        model.eval()
        rng_state = torch.get_rng_state()
        np.random.seed(0)
        torch.manual_seed(0)

        sample_idxs = np.random.choice(epdf_interp.size(1), size=8)

        # Sampling
        samples = [epdf_interp[-1, sample_idxs]]
        for mask in epdf_masks[:-1, sample_idxs].flip(0):
            with torch.no_grad():
                x = model(samples[-1].to(device)).sample().cpu()
            x = x * mask
            samples.append(x)
        samples = torch.stack(samples)
        samples = ((samples + 1) / 2).clamp(0.0, 1.0)
        summary_writer.add_image(
            "samples/generative",
            make_grid(
                samples.transpose(0, 1).reshape(-1, *samples.shape[2:]),
                nrow=samples.shape[0],
            ),
            global_step=global_step,
        )

        # Super resolution
        samples = [
            torch.nn.functional.pad(
                viz_samples[sample_idxs],
                [x // 2 + 1 for x in viz_samples.shape[-2:][::-1] for _ in range(2)],
                mode="constant",
                value=-1,
            )
        ]
        cur_scale_factor = 1.0
        while cur_scale_factor < 2.0:
            cur_scale_factor *= incremental_scale
            with torch.no_grad():
                x = model(samples[-1].to(device)).sample().cpu()
            samples.append(x)
        samples = torch.stack(samples)
        samples = ((samples + 1) / 2).clamp(0.0, 1.0)
        summary_writer.add_image(
            "samples/super-resolution",
            make_grid(
                samples.transpose(0, 1).reshape(-1, *samples.shape[2:]),
                nrow=samples.shape[0],
            ),
            global_step=global_step,
        )

        ## Trainig loop
        model.train()
        torch.set_rng_state(rng_state)
        losses = []
        progress = tqdm(dataloader)
        for x, _ in progress:
            x = x.to(device=device)
            data_masks = torch.ones_like(x)

            x = interpolate_samples(x, scale_factors, padding_mode="border")
            data_masks = interpolate_samples(
                data_masks, scale_factors, padding_mode="zeros"
            )

            y = model(x[1:].reshape(-1, *x.shape[2:]))
            lp = y.log_prob(x[:-1].reshape(-1, *x.shape[2:]))
            lp = lp * data_masks[:-1].reshape(-1, *data_masks.shape[2:])
            loss = -lp.sum((1, 2, 3)).mean(0)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            progress.set_description(f"LOSS = {np.mean(losses):.2e}")
            summary_writer.add_scalar("loss", loss, global_step=global_step)

            global_step += 1

        logging.info(
            "  //  ".join(
                [
                    f"EPOCH {epoch:d}",
                    f"LOSS = {np.mean(losses):.3e}",
                ]
            )
        )


if __name__ == "__main__":
    main()
