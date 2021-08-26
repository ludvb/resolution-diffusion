import os

import numpy as np
import torch


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
        x = x.reshape(x.shape[0], -1)
        return self._forward(x)


def get_model(dataset):
    model_filename = os.path.join("./data", f"mlp-classifier-{hash(dataset)}.pkl")

    if os.path.exists(model_filename):
        return torch.load(model_filename)

    c, h, w = next(iter(dataset))[0].shape
    model = MLPClassifier(c, h, w, len(dataset.classes))

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, shuffle=True, drop_last=True
    )

    accuracy = 0.0
    while accuracy < 0.98:
        accuracies = []
        for x, target in dataloader:
            ps = model(x)
            loss = torch.nn.functional.nll_loss(ps, target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            accuracies.append(((ps.argmax(1) == target).sum() / target.shape[0]).item())
        accuracy = np.mean(accuracies)

    torch.save(model, model_filename)
    return model


def inception_score(imgs: torch.Tensor, dataset: torch.utils.data.Dataset) -> float:
    model = get_model(dataset)
    model = model.to(imgs)

    def _compute_is(imgs):
        ps_gen = model(imgs)
        kl = ps_gen.exp() * (ps_gen - ps_gen.logsumexp(0) + np.log(ps_gen.shape[0]))
        return kl.sum(1).mean().item()

    return _compute_is(imgs)
