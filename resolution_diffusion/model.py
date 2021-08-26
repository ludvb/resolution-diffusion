import numpy as np
import torch

from .common import interpolate2d


class ResidualBlock(torch.jit.ScriptModule):
    def __init__(self, num_features: int):
        super().__init__()
        self._forward = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._forward(x)


class SelfAttentionBlock(torch.jit.ScriptModule):
    def __init__(self, num_features: int, num_latent_features: int):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(num_features)
        self.proj_q = torch.nn.Conv2d(num_features, num_latent_features, kernel_size=1)
        self.proj_k = torch.nn.Conv2d(num_features, num_latent_features, kernel_size=1)
        self.proj_v = torch.nn.Conv2d(num_features, num_latent_features, kernel_size=1)
        self.proj_orig = torch.nn.Conv2d(
            num_latent_features, num_features, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        w = torch.einsum("bchw,bcHW->bhwHW", q, k)
        w_ = w.flatten(-2)
        w_ = torch.nn.functional.softmax(w_, dim=-1)
        w = w_.reshape_as(w)
        y = torch.einsum("bhwHW,bcHW->bchw", w, v)
        return x + self.proj_orig(y)


class UNet(torch.jit.ScriptModule):
    def __init__(
        self,
        num_features: int,
        num_levels: int = 4,
        attention_levels: list[int] = [1, 4],
    ):
        super().__init__()

        def _level_features(k):
            return num_features * 2 ** k

        def _make_block_at_level(k):
            if k in attention_levels:
                return torch.nn.Sequential(
                    ResidualBlock(_level_features(k)),
                    SelfAttentionBlock(_level_features(k), _level_features(k)),
                )
            return ResidualBlock(_level_features(k))

        self._downsample_blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        _level_features(k),
                        _level_features(k + 1),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    _make_block_at_level(k + 1),
                )
                for k in range(num_levels)
            ]
        )
        self._bottom_block = torch.nn.Sequential(
            ResidualBlock(num_features * 2 ** num_levels),
        )
        self._upsample_blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=True
                    ),
                    torch.nn.Conv2d(
                        2 * _level_features(k + 1),
                        _level_features(k),
                        kernel_size=3,
                        padding=1,
                    ),
                    _make_block_at_level(k),
                )
                for k in reversed(range(num_levels))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = [x]
        for block in self._downsample_blocks:
            hs.append(block(hs[-1]))
        y = self._bottom_block(hs[-1])
        for block, h in zip(self._upsample_blocks, hs[::-1]):
            y = torch.cat([y, h], dim=1)
            y = block(y)
        return y


class Model(torch.jit.ScriptModule):
    def __init__(
        self,
        incremental_scale: float,
        img_channels: int = 1,
        num_latent_features: int = 32,
        num_levels: int = 2,
        attention_levels: list[int] = [1, 2],
    ):
        super().__init__()
        self.incremental_scale = incremental_scale
        self._unet = UNet(
            num_latent_features,
            num_levels=num_levels,
            attention_levels=attention_levels,
        )
        self._pre_transform = torch.nn.Sequential(
            torch.nn.Conv2d(
                img_channels, num_latent_features, kernel_size=3, padding=1
            ),
            torch.nn.Dropout2d(inplace=True),
        )
        self._post_transform_mu = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_latent_features),
            torch.nn.Conv2d(num_latent_features, num_latent_features, kernel_size=1),
            torch.nn.SiLU(inplace=True),
            torch.nn.BatchNorm2d(num_latent_features),
            torch.nn.Conv2d(
                num_latent_features, img_channels, kernel_size=1, bias=False
            ),
        )
        torch.nn.init.normal_(
            self._post_transform_mu[-1].weight,
            mean=0.0,
            std=1e-2 / np.sqrt(num_latent_features),
        )
        self._post_transform_sd = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_latent_features),
            torch.nn.Conv2d(num_latent_features, num_latent_features, kernel_size=1),
            torch.nn.SiLU(inplace=True),
            torch.nn.BatchNorm2d(num_latent_features),
            torch.nn.Conv2d(num_latent_features, img_channels, kernel_size=1),
            torch.nn.Softplus(),
        )
        torch.nn.init.constant_(
            self._post_transform_sd[-2].bias,
            val=np.log(0.1),
        )
        torch.nn.init.normal_(
            self._post_transform_sd[-2].weight,
            mean=0.0,
            std=1e-2 / np.sqrt(num_latent_features),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = interpolate2d(x, torch.tensor([[self.incremental_scale]], device=x.device))
        x = x.squeeze(1)
        h = self._pre_transform(x)
        h = self._unet(h)
        mu = x + self._post_transform_mu(h)
        sd = self._post_transform_sd(h)
        return mu, sd.clamp_min(1e-2)
