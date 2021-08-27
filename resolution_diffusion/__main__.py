import argparse
import os
import itertools as it
import logging
import subprocess
import sys
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as image_transforms
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10, MNIST
from torchvision.utils import make_grid
from tqdm import tqdm

from . import visualization as viz
from .common import (
    add_noise,
    first_unique_filename,
    compute_scale_factors,
    interpolate2d,
)
from .evaluation import inception_score
from .model import Model

world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--checkpoint", type=str)
    argparser.add_argument("--learning-rate", type=float, default=1e-4)
    argparser.add_argument("--grad-clip", type=float)
    argparser.add_argument("--incremental-scale", type=float, default=1.25)
    argparser.add_argument("--dataset", type=str, default="MNIST")
    argparser.add_argument("--add-dataset-noise", type=bool)
    argparser.add_argument("--batch-size", type=int, default=32)
    argparser.add_argument("--features", type=int, default=32)
    argparser.add_argument("--num-levels", type=int, default=2)
    argparser.add_argument("--attention", nargs="*", type=int, default=[1, 2])
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--viz-interval", type=int, default=1000)
    argparser.add_argument("--checkpoint-interval", type=int, default=1000)
    argparser.add_argument("--mp-host", type=str, default="localhost")
    argparser.add_argument("--mp-port", type=str, default="12355")
    argparser.add_argument(
        "--save-path",
        type=str,
        default=f"./resolution-diffusion-{datetime.now().isoformat()}",
    )
    options = argparser.parse_args()

    os.makedirs(os.path.join(options.save_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(options.save_path, "tb"), exist_ok=True)

    os.environ["MASTER_ADDR"] = options.mp_host
    os.environ["MASTER_PORT"] = options.mp_port
    if world_size == 1:
        run_training(0, options)
    else:
        torch.multiprocessing.spawn(run_training, nprocs=world_size, args=(options,))


def run_training(rank, options):
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logging.basicConfig(
            filename=os.path.join(options.save_path, "log"),
            level=logging.DEBUG,
            format="[%(asctime)s]  (%(levelname)s)  %(message)s",
            encoding="utf-8",
        )
    else:
        logging.basicConfig(
            filename=os.path.join(options.save_path, "log"),
            level=logging.WARNING,
            format="[%(asctime)s]  $(process)d  (%(levelname)s)  %(message)s",
            encoding="utf-8",
        )

    logging.info(
        "Commit: {}".format(
            subprocess.run(
                ["git", "describe", "--always", "--long", "--dirty"],
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )
    )
    logging.info(" ".join(sys.argv))
    logging.info("Options parsed as: " + str(options))

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
                + (
                    [
                        add_noise,
                        # ^ Add isotropic Gaussian noise to make the data manifold
                        #   slightly smoother. MNIST images have many extreme values
                        #   (0 or 1) that otherwise may make training more difficult.
                    ]
                    if options.add_dataset_noise
                    else []
                )
            ),
        )
    elif options.dataset.lower() == "cifar10":
        dataset = CIFAR10(
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
                + ([add_noise] if options.add_dataset_noise else [])
            ),
        )
    else:
        raise RuntimeError(f'Unknown dataset "{options.dataset}"')

    dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        num_workers=len(os.sched_getaffinity(0)),
        persistent_workers=True,
        sampler=torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        ),
    )

    incremental_scale = options.incremental_scale
    img_channels = dataset[0][0].shape[0]
    scale_factors = compute_scale_factors(dataset, incremental_scale)

    torch.manual_seed(options.seed)
    model = Model(
        incremental_scale=incremental_scale,
        img_channels=img_channels,
        num_latent_features=options.features,
        num_levels=options.num_levels,
        attention_levels=options.attention,
    )
    model = torch.nn.parallel.DistributedDataParallel(
        model.to(device=device), device_ids=[device], output_device=device
    )

    optim = ZeroRedundancyOptimizer(
        model.parameters(), torch.optim.Adam, lr=options.learning_rate
    )

    if options.checkpoint is not None:
        logging.info("Loading checkpoint: %s", options.checkpoint)
        checkpoint = torch.load(options.checkpoint)
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])

    summary_writer = (
        SummaryWriter(os.path.join(options.save_path, "tb")) if rank == 0 else None
    )

    def generate_batches():
        for epoch in it.count(1):
            for batch_number, (x, _) in enumerate(dataloader):
                yield epoch, batch_number, x

    def step(x, global_step):
        model.train()

        x = x.to(device=device)
        data_masks = torch.ones_like(x)

        scale_factors_idxs = torch.randint(
            1, scale_factors.shape[0], size=(x.shape[0],)
        )
        x_src = interpolate2d(
            x,
            scale_factors[scale_factors_idxs].unsqueeze(1).to(x),
            padding_mode="zeros",
        ).squeeze(1)
        x_trg = interpolate2d(
            x,
            scale_factors[scale_factors_idxs - 1].unsqueeze(1).to(x),
            padding_mode="zeros",
        ).squeeze(1)
        data_masks = interpolate2d(
            data_masks,
            scale_factors[scale_factors_idxs - 1].unsqueeze(1).to(x),
            padding_mode="zeros",
        ).squeeze(1)

        y = model(x_src)
        lp = y.log_prob(x_trg).reshape_as(data_masks)
        loss = -(lp * data_masks).sum((1, 2, 3)).mean(0)

        optim.zero_grad()
        loss.backward()
        if options.grad_clip:
            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), options.grad_clip
            )
        optim.step()

        torch.distributed.all_reduce(loss)

        if summary_writer:
            summary_writer.add_scalar("loss", loss, global_step=global_step)

        return loss.item()

    progress = tqdm(total=len(dataloader)) if rank == 0 else None
    losses = []
    for global_step, (epoch, batch_number, x) in enumerate(generate_batches()):
        if global_step % options.checkpoint_interval == 0:
            optim.consolidate_state_dict()
            if rank == 0:
                torch.save(
                    {"model": model.state_dict(), "optim": optim.state_dict()},
                    first_unique_filename(
                        os.path.join(
                            options.save_path,
                            "checkpoints",
                            f"step-{global_step:06d}.pkl",
                        )
                    ),
                )

        if global_step % options.viz_interval == 0:
            model.eval()
            torch_rng_state = torch.get_rng_state()
            np_rng_state = np.random.get_state()
            np.random.seed(0)
            torch.manual_seed(0)

            samples = viz.sampling(model, incremental_scale, dataset, device)
            if summary_writer:
                summary_writer.add_image(
                    "samples/generative",
                    make_grid(
                        samples.transpose(0, 1).reshape(-1, *samples.shape[2:]),
                        nrow=samples.shape[0],
                    ),
                    global_step=global_step,
                )
                summary_writer.add_scalar(
                    "samples/generative/inception-score",
                    inception_score(samples[-1], dataset),
                    global_step=global_step,
                )

            superres = viz.super_resolution(model, incremental_scale, dataset, device)
            if summary_writer:
                summary_writer.add_image(
                    "samples/super-resolution",
                    make_grid(
                        superres.transpose(0, 1).reshape(-1, *superres.shape[2:]),
                        nrow=superres.shape[0],
                    ),
                    global_step=global_step,
                )

            torch.set_rng_state(torch_rng_state)
            np.random.set_state(np_rng_state)

        loss = step(x, global_step=global_step)

        if batch_number == 0:
            losses.clear()
        losses.append(loss)

        if progress:
            if batch_number == 0:
                print("")
                progress.reset()
            progress.update()
            progress.set_description(f"EPOCH {epoch:04d}  LOSS = {np.mean(losses):.2e}")


if __name__ == "__main__":
    main()
