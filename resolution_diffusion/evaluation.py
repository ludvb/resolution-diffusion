import os
from typing import Optional

import numpy as np
import torch
import torch.distributions as distr
from tqdm import tqdm


class MLPClassifier(torch.nn.Module):
    def __init__(self, c, h, w, classes):
        super().__init__()
        self._forward = torch.nn.Sequential(
            torch.nn.Linear(h * w * c, 2048),
            torch.nn.SiLU(inplace=True),
            torch.nn.Linear(2048, 2048),
            torch.nn.SiLU(inplace=True),
            torch.nn.Linear(2048, classes),
            torch.nn.LogSoftmax(-1),
        )

    def forward(self, x):
        x = x.flatten(-3)
        return self._forward(x)


def dataset_hash(dataset: torch.utils.data.Dataset) -> int:
    return int(sum([hash(x) for x in dataset.resources]) % 1e9)


def get_mlp(dataset: torch.utils.data.Dataset, device: Optional[torch.device]):
    model_filename = os.path.join("./data", f"mlp-classifier-{dataset_hash(dataset)}.pkl")

    if os.path.exists(model_filename):
        return torch.load(model_filename)

    c, h, w = next(iter(dataset))[0].shape
    model = MLPClassifier(c, h, w, len(dataset.classes))
    model = model.to(device=device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, shuffle=True, drop_last=True
    )

    accuracy = 0.0
    while accuracy < 0.98:
        accuracies = []
        for x, target in dataloader:
            x = x.to(device=device)

            ps = model(x)
            loss = torch.nn.functional.nll_loss(ps, target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            accuracies.append(((ps.argmax(1) == target).sum() / target.shape[0]).item())
        accuracy = np.mean(accuracies)

    torch.save(model, model_filename)
    return model


class VAE(torch.nn.Module):
    def __init__(self, c: int, h: int, w: int, num_latent: int):
        super().__init__()
        self._c, self._h, self._w = c, h, w
        self._num_latent = num_latent
        self._fq_mu = torch.nn.Sequential(
            torch.nn.Linear(h * w * c, 512),
            torch.nn.SiLU(inplace=True),
            torch.nn.Linear(512, num_latent),
        )
        self._fq_sd = torch.nn.Sequential(
            torch.nn.Linear(h * w * c, 512),
            torch.nn.SiLU(inplace=True),
            torch.nn.Linear(512, num_latent),
            torch.nn.Softplus(),
        )
        self._fp_mu = torch.nn.Sequential(
            torch.nn.Linear(num_latent, 512),
            torch.nn.SiLU(inplace=True),
            torch.nn.Linear(512, h * w * c),
        )
        self._fp_sd = torch.nn.Sequential(
            torch.nn.Linear(num_latent, 512),
            torch.nn.SiLU(inplace=True),
            torch.nn.Linear(512, h * w * c),
            torch.nn.Softplus(),
        )

    def q(self, x):
        x = x.flatten(-3)
        mu = self._fq_mu(x)
        sd = self._fq_sd(x)
        return mu, sd

    def p(self, z):
        mu = self._fp_mu(z).reshape(-1, self._c, self._h, self._w)
        sd = self._fp_sd(z).reshape(-1, self._c, self._h, self._w)
        return mu, torch.nn.functional.softplus(sd)

    def forward(self, x):
        qmu, qsd = self.q(x)
        z = qmu + qsd * torch.randn_like(qsd)
        pmu, psd = self.p(z)
        return qmu, qsd, z, pmu, psd


def get_vae(
    dataset: torch.utils.data.Dataset,
    num_latent: int = 20,
    device: Optional[torch.device] = None,
) -> VAE:
    model_filename = os.path.join("./data", f"vae-{dataset_hash(dataset)}.pkl")

    if os.path.exists(model_filename):
        return torch.load(model_filename)

    c, h, w = next(iter(dataset))[0].shape
    model = VAE(c, h, w, num_latent)
    model = model.to(device=device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, shuffle=True, drop_last=True
    )
    iscore_ma = 0.0
    best_iscore = 0.0
    best_params = model.state_dict()
    while True:
        progress = tqdm(dataloader)
        losses = []
        for x, _ in progress:
            x = x.to(device=device)

            qmu, qsd, _, pmu, psd = model(x)
            lp_x = distr.Normal(pmu, psd).log_prob(x)
            lp_x = lp_x.sum((1, 2, 3))
            dkl_z = distr.kl_divergence(distr.Normal(qmu, qsd), distr.Normal(0.0, 1.0))
            dkl_z = dkl_z.sum(1)
            dkl_xz = dkl_z - lp_x
            loss = dkl_xz.mean(0)

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.append(loss.item())
            progress.set_description(f"LOSS = {np.mean(losses):.3e}")

        model.eval()
        with torch.no_grad():
            pmu, _ = model.p(torch.randn(512, num_latent, device=device))
        model.train()
        iscore = inception_score(pmu, dataset, device=device)
        if iscore > best_iscore:
            best_params = model.state_dict()
            best_iscore = iscore
        if iscore < iscore_ma:
            break
        print(iscore)
        iscore_ma = 0.99 * iscore_ma + 0.01 * iscore

    model.load_state_dict(best_params)
    torch.save(model, model_filename)
    return model


def generate_vae_samples(
    dataset: torch.utils.data.Dataset,
    device: Optional[torch.device] = None,
    num_latent: int = 20,
    num_samples: int = 512,
) -> torch.Tensor:
    model = get_vae(dataset, num_latent, device=device)
    model.eval()
    with torch.no_grad():
        pmu, _ = model.p(torch.randn(num_samples, num_latent, device=device))
    model.train()
    return pmu


def inception_score(
    imgs: torch.Tensor,
    dataset: torch.utils.data.Dataset,
    device: Optional[torch.device] = None,
) -> float:
    model = get_mlp(dataset, device=device)
    model = model.to(device=device)

    def _compute_is(imgs):
        with torch.no_grad():
            ps_gen = model(imgs)
        kl = ps_gen.exp() * (ps_gen - ps_gen.logsumexp(0) + np.log(ps_gen.shape[0]))
        return kl.sum(1).mean().item()

    return _compute_is(imgs)
