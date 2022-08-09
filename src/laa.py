from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mode
from torch.utils.data import Dataset


def majority_vote(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("task_id").answer.apply(lambda x: mode(x)[0][0]).values


@dataclass
class Batch:
    x: torch.Tensor
    mask: torch.Tensor


class ReshapeWrapper(nn.Module):
    def __init__(self, pattern: Tuple):
        super().__init__()
        self.pattern = pattern

    def forward(self, x: torch.Tensor):
        batch_size = x.size()[0]
        return x.view((batch_size, *self.pattern))


class LAA(nn.Module):
    def __init__(
        self, n_voters: int, n_classes: int, d_kl: float, reg_1: float = 0.001
    ):
        super().__init__()

        self.n_voters = n_voters
        self.n_classes = n_classes
        self.d_kl = d_kl
        self.reg_1 = reg_1

        self.encoder = nn.Sequential(
            nn.Linear(n_voters * n_classes, n_classes), nn.Softmax()
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_classes, n_voters * n_classes),
            ReshapeWrapper((n_voters, n_classes)),
            nn.Softmax(dim=-1),
        )

    def encode(
        self, x: torch.FloatTensor, mask: torch.FloatTensor
    ) -> torch.FloatTensor:
        # x has shape (batch_size, n_voters * n_classes)
        # mask has shape (batch_size, n_voters * n_classes)
        assert x.size() == (
            x.size()[0],
            self.n_voters * self.n_classes,
        ), f"{x.size()}, {self.n_voters}, {self.n_classes}"
        q_theta: torch.FloatTensor = self.encoder(
            x * mask
        )  # shape (batch_size, n_classes)
        return q_theta

    def sample(self, q_theta: torch.FloatTensor) -> torch.FloatTensor:
        # q_theta has shape (batch_size, n_classes)
        y_sampled = torch.distributions.Categorical(
            probs=q_theta
        ).sample()  # shape (n_classes)

        y_sampled = F.one_hot(y_sampled, num_classes=self.n_classes).type(
            torch.FloatTensor
        )  # shape (batch_size, n_classes)
        return y_sampled

    def forward(self, x: torch.FloatTensor, mask: torch.FloatTensor) -> dict:

        batch_size = x.size()[0]

        q_theta = self.encode(x, mask)  # shape (batch_size, n_classes)

        y_sampled = self.sample(q_theta)  # shape (batch_size, n_classes)

        y_estim = F.one_hot(
            q_theta.argmax(-1, keepdim=True), num_classes=self.n_classes
        ).type(torch.FloatTensor)[
            :, 0, :
        ]  # shape (batch_size, n_classes)

        p_phi: torch.Tensor = self.decoder(
            y_sampled
        )  # shape (batch_size, n_voters, n_classes)

        reconstruction_loss = (
            +(mask * x * p_phi.view((batch_size, -1)).log()).sum(-1).mean()
        )  # shape ()

        prior = (
            (mask * x).view((batch_size, self.n_voters, self.n_classes)).sum(dim=1)
        )  # shape (batch_size, n_classes)
        prior = (prior.transpose(0, 1) / prior.sum(dim=-1)).transpose(0, 1)
        D_kl = (
            (q_theta * ((q_theta / prior.clamp(1.0e-10)).log())).sum(-1).mean()
        )  # shape ()

        loss = -reconstruction_loss + self.d_kl * D_kl
        l1_reg = torch.autograd.Variable(
            torch.zeros_like(torch.FloatTensor(1)), requires_grad=True
        )
        for W in self.parameters():
            l1_reg = l1_reg + W.norm(1)

        loss = loss + self.reg_1 * l1_reg

        return dict(
            p_phi=p_phi,
            reconstruction_loss=reconstruction_loss,
            D_kl=D_kl,
            loss=loss,
            y_estim=y_estim,
            q_theta=q_theta,
        )


class CrowdDataset(Dataset):
    def __init__(self, answers: pd.DataFrame, n_classes: int, n_voters: int) -> None:
        super().__init__()
        self.answers = answers.pivot(index="task_id", columns=["worker_id"])
        self.n_classes = n_classes

        missed_workers = set(range(n_voters)) - set(answers["worker_id"].unique())
        for worker in missed_workers:
            self.answers[("answer", worker)] = np.nan
        self.answers = self.answers.sort_index(axis=1)

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index) -> dict:
        row = self.answers.iloc[index].values
        idx = self.answers.iloc[index].name

        mask = torch.from_numpy(row).ge(-1).type(torch.FloatTensor)
        mask = mask.repeat_interleave(self.n_classes).view(-1, self.n_classes).flatten()
        x = torch.from_numpy(row).nan_to_num(0).type(torch.int64)
        x = (
            F.one_hot(
                x,
                num_classes=self.n_classes,
            )
            .type(torch.FloatTensor)
            .flatten()
        )

        return {"x": x, "mask": mask, "idx": idx}


def train(model, optimizer, dataloader, device="cpu", reg_1=0.001):
    model.train()
    epoch_loss = list()
    for data in dataloader:

        x: torch.Tensor = data["x"].to(device)
        mask = data["mask"].to(device)

        output = model(x, mask)

        loss = output["loss"]

        epoch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(epoch_loss)


def inference(model: nn.Module, dataloader, device="cpu"):
    model.eval()
    with torch.no_grad():
        ground_truth_estim = list()
        idx_list = list()
        loss = list()
        for data in dataloader:
            x = data["x"].to(device)
            mask = data["mask"].to(device)
            idx = data["idx"]
            output = model(x, mask)
            y_sampled = output["y_estim"]
            loss.append(output["loss"].item())
            ground_truth_estim.append(y_sampled)
            idx_list.append(idx)
    idx_list = torch.cat(idx_list).numpy()
    return np.mean(loss), dict(
        zip(idx_list, torch.cat(ground_truth_estim).argmax(-1).flatten().numpy())
    )
