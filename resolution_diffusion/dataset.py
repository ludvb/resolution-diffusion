import torch
import torchvision.transforms as image_transforms
from torchvision.datasets import CIFAR10, MNIST

from .common import add_noise as _add_noise_fn


def load_dataset(name: str, add_noise: bool) -> torch.utils.data.Dataset:
    if name.lower() == "mnist":
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
                        _add_noise_fn,
                        # ^ Add isotropic Gaussian noise to make the data manifold
                        #   slightly smoother. MNIST images have many extreme values
                        #   (0 or 1) that otherwise may make training more difficult.
                    ]
                    if add_noise
                    else []
                )
            ),
        )
    elif name.lower() == "cifar10":
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
                + ([_add_noise_fn] if add_noise else [])
            ),
        )
    else:
        raise RuntimeError(f'Unknown dataset "{name}"')

    return dataset
