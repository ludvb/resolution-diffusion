import argparse
import os
import itertools as it
import logging
import sys
from datetime import datetime
from subprocess import run

import numpy as np
import torch
import torchvision.transforms as image_transforms
from torch.distributions import Normal
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm import tqdm

from .common import first_unique_filename, interpolate, interpolate_samples
from .model import Model

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--checkpoint", type=str)
    argparser.add_argument("--pixel-weights", type=bool)
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

    logging.info(
        "Commit: {}".format(
            run(
                ["git", "describe", "--always", "--long", "--dirty"],
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )
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
    epdf_interp = interpolate_samples(viz_samples, scale_factors, padding_mode="zeros")
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
            x = x.clamp(-1.0, 1.0)
            x[~mask] = 0.0
            samples.append(x)
        samples = torch.stack(samples)
        samples = (samples + 1.0) / 2
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
                [(x + 1) // 2 for x in viz_samples.shape[-2:][::-1] for _ in range(2)],
                mode="constant",
                value=0.0,
            )
        ]
        mask = torch.nn.functional.pad(
            torch.ones_like(viz_samples[sample_idxs]),
            [(x + 1) // 2 for x in viz_samples.shape[-2:][::-1] for _ in range(2)],
            mode="constant",
            value=0.0,
        )
        cur_scale_factor = 1.0
        while cur_scale_factor < 2.0:
            cur_scale_factor *= incremental_scale
            mask = interpolate(mask, scale_factor=incremental_scale)
            with torch.no_grad():
                x = model(samples[-1].to(device)).sample().cpu()
            x = x.clamp(-1.0, 1.0)
            x[~mask.bool()] = 0.0
            samples.append(x)
        samples = torch.stack(samples)
        samples = (samples + 1.0) / 2
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

            x = interpolate_samples(x, scale_factors, padding_mode="zeros")
            data_masks = interpolate_samples(
                data_masks, scale_factors[:-1], padding_mode="zeros"
            )
            if options.pixel_weights:
                pixel_weights = data_masks / data_masks.sum((2, 3, 4), keepdim=True)
            else:
                pixel_weights = data_masks

            y = model(x[1:].reshape(-1, *x.shape[2:]))
            lp = y.log_prob(x[:-1].reshape(-1, *x.shape[2:])).reshape_as(pixel_weights)
            loss = -(lp * pixel_weights).sum((1, 2, 3, 4)).mean(0)

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
