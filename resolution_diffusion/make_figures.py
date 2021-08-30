import argparse
import os
from collections import OrderedDict

import numpy as np
import torch
from imageio import imsave
from torchvision.utils import make_grid

from . import visualization as viz
from .common import interpolate2d
from .dataset import load_dataset
from .evaluation import generate_vae_samples, inception_score
from .model import Model


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--checkpoint", type=str, required=True)
    argparser.add_argument("--incremental-scale", type=float, default=1.25)
    argparser.add_argument("--dataset", type=str, default="MNIST")
    argparser.add_argument("--add-dataset-noise", type=bool)
    argparser.add_argument("--features", type=int, default=32)
    argparser.add_argument("--num-levels", type=int, default=2)
    argparser.add_argument("--attention", nargs="*", type=int, default=[1, 2])
    argparser.add_argument("--num-samples", type=int, default=32)
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument(
        "--save-path",
        type=str,
        required=True,
    )
    options = argparser.parse_args()
    make_figures(options)


def make_figures(options):
    dataset = load_dataset(options.dataset, options.add_dataset_noise)

    incremental_scale = options.incremental_scale
    img_channels = dataset[0][0].shape[0]

    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    model = Model(
        incremental_scale=incremental_scale,
        img_channels=img_channels,
        num_latent_features=options.features,
        num_levels=options.num_levels,
        attention_levels=options.attention,
    )

    checkpoint = torch.load(options.checkpoint)
    model.load_state_dict(
        OrderedDict([(k[7:], v) for k, v in checkpoint["model"].items()])
        # ^ Removes the "module." prefix added by DistributedDataParallel
    )
    model.eval()

    os.makedirs(options.save_path, exist_ok=True)

    ## Sampling
    sampling, sampling_scale_factors = viz.sampling(
        model, incremental_scale, dataset, num_samples=options.num_samples
    )
    sampling = (sampling + 1.0) / 2
    sampling_frames = viz.make_animation_frames(sampling, sampling_scale_factors)
    imsave(
        os.path.join(options.save_path, "sampling.png"),
        make_grid(
            sampling[
                # torch.linspace(0, sampling.shape[0] - 1, 10).round().long().unique(),
                : options.num_samples // 2,
            ].flatten(0, 1),
            nrow=options.num_samples // 2,
        ).permute(1, 2, 0),
    )
    imsave(
        os.path.join(options.save_path, "sampling-resized.png"),
        make_grid(
            sampling_frames[
                :,
                torch.linspace(0, sampling_frames.shape[0] - 1, 10)
                .round()
                .long()
                .unique(),
                -options.num_samples // 2 :,
            ].flatten(0, 1),
            nrow=options.num_samples // 2,
        ).permute(1, 2, 0),
    )
    viz.make_animation_clip(
        2 * torch.stack([make_grid(frame) for frame in sampling_frames]) - 1,
        animation_duration=8,
    ).write_gif(os.path.join(options.save_path, "sampling-animated.gif"), fps=32)
    viz.make_animation_clip(
        2 * torch.stack([make_grid(frame) for frame in sampling_frames]) - 1,
        animation_duration=8,
    ).write_gif(
        os.path.join(options.save_path, "sampling-resized-animated.gif"), fps=32
    )

    # VAE comparison
    reference = torch.stack(
        [dataset[i][0] for i in np.random.randint(len(dataset), size=16 * 32)]
    )
    sampling, _ = viz.sampling(model, incremental_scale, dataset, num_samples=16 * 32)
    sampling = sampling[-1]
    sampling_vae = generate_vae_samples(dataset, num_samples=16 * 32).clamp(-1.0, 1.0)
    imsave(
        os.path.join(options.save_path, "sampling-reference-16x32.png"),
        make_grid((reference + 1.0) / 2, nrow=32).permute(1, 2, 0),
    )
    imsave(
        os.path.join(options.save_path, "sampling-16x32.png"),
        make_grid((sampling + 1.0) / 2, nrow=32).permute(1, 2, 0),
    )
    imsave(
        os.path.join(options.save_path, "sampling-vae-16x32.png"),
        make_grid((sampling_vae + 1.0) / 2, nrow=32).permute(1, 2, 0),
    )
    for name, data in [
        ("dataset", reference),
        ("resolution_diffusion", sampling),
        ("vae", sampling_vae),
    ]:
        with open(
            os.path.join(options.save_path, f"sampling-inception-scores-{name}.txt"),
            "w",
        ) as fp:
            score = inception_score(data, dataset)
            fp.write(f"{score:f}")

    ## Super resolution
    superres, superres_scale_factors, reference = viz.super_resolution(
        model, incremental_scale, dataset, num_samples=options.num_samples
    )
    superres = (superres + 1.0) / 2
    reference = (reference + 1.0) / 2
    superres_frames = viz.make_animation_frames(superres, superres_scale_factors)
    imsave(
        os.path.join(options.save_path, "superres.png"),
        make_grid(
            torch.cat(
                [
                    superres[
                        torch.linspace(0, superres.shape[0] - 1, 10)
                        .round()
                        .long()
                        .unique(),
                        : options.num_samples // 2,
                    ],
                    reference[:16].unsqueeze(0),
                ]
            ).flatten(0, 1),
            nrow=options.num_samples // 2,
        ).permute(1, 2, 0),
    )
    imsave(
        os.path.join(options.save_path, "superres-resized.png"),
        make_grid(
            torch.cat(
                [
                    superres_frames[
                        torch.linspace(0, superres_frames.shape[0] - 1, 10)
                        .round()
                        .long()
                        .unique(),
                        -options.num_samples // 2 :,
                    ],
                    reference[-options.num_samples // 2 :].unsqueeze(0),
                ]
            ).flatten(0, 1),
            nrow=options.num_samples // 2,
        ).permute(1, 2, 0),
    )
    viz.make_animation_clip(
        2
        * torch.stack(
            [
                make_grid(torch.stack([frame, reference], 1).flatten(0, 1))
                for frame in superres_frames
            ]
        )
        - 1,
        animation_duration=8,
    ).write_gif(os.path.join(options.save_path, "superres-animated.gif"), fps=32)
    viz.make_animation_clip(
        2
        * torch.stack(
            [
                make_grid(torch.stack([frame, reference], 1).flatten(0, 1))
                for frame in superres_frames
            ]
        )
        - 1,
        animation_duration=8,
    ).write_gif(
        os.path.join(options.save_path, "superres-resized-animated.gif"), fps=32
    )

    # Bilinear comparison
    superres, superres_scale_factors, reference = viz.super_resolution(
        model, incremental_scale, dataset, num_samples=16 * 32
    )
    bilinear = interpolate2d(
        superres[0],
        scale_factors=(1.0 / superres_scale_factors[0])[None, None],
        mode="bilinear",
    )
    bilinear = bilinear.squeeze(1)
    imsave(
        os.path.join(options.save_path, "superres-reference-16x32.png"),
        make_grid((reference + 1.0) / 2, nrow=32).permute(1, 2, 0),
    )
    imsave(
        os.path.join(options.save_path, "superres-16x32.png"),
        make_grid((superres[-1] + 1.0) / 2, nrow=32).permute(1, 2, 0),
    )
    imsave(
        os.path.join(options.save_path, "superres-bilinear-16x32.png"),
        make_grid((bilinear + 1.0) / 2, nrow=32).permute(1, 2, 0),
    )
    for name, data in [
        ("dataset", reference),
        ("resolution_diffusion", superres[-1]),
        ("bilinear", bilinear),
    ]:
        with open(
            os.path.join(options.save_path, f"superres-inception-scores-{name}.txt"),
            "w",
        ) as fp:
            score = inception_score(data, dataset)
            fp.write(f"{score:f}")


if __name__ == "__main__":
    main()
