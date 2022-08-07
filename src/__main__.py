import json
import random
from enum import Enum
from unittest.mock import MagicMock

import numpy as np
import torch
import typer
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from src.crowddatasets import DATASET_MAP
from src.laa import LAA, CrowdDataset, inference, majority_vote, train

try:
    import wandb
except ImportError:
    wandb = MagicMock()

app = typer.Typer()


class DatasetName(str, Enum):
    classification_dataset_generator = "classification_dataset_generator"
    releveance2 = "releveance2"
    bluebirds = "bluebirds"


@app.command()
def main(
    batch_size: int = 64,
    lr: float = 0.00001,
    d_kl: float = 0.001,
    reg_1: float = 0.001,
    n_epoch: int = 15,
    dataset_name: DatasetName = "bluebirds",
    dataset_kwargs: str = "{}",
    seed: int = 11,
    logging: bool = True,
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    BATCH_SIZE = batch_size
    LR = lr
    N_EPOCH = n_epoch

    dataset_kwargs = json.loads(dataset_kwargs)
    sample_dataset = DATASET_MAP[dataset_name](**dataset_kwargs)

    if not logging:
        wandb = MagicMock()

    config = dict(
        batch_size=batch_size,
        lr=lr,
        n_epoch=n_epoch,
        d_kl=d_kl,
        dataset_name=dataset_name,
        dataset_kwargs=dataset_kwargs,
        seed=seed,
        reg_1=reg_1,
    )

    print(config)
    wandb.init(project="laa", config=config, tags=[dataset_name])

    model = LAA(sample_dataset.n_workers, sample_dataset.n_classes, d_kl=d_kl)

    dataset = CrowdDataset(
        sample_dataset.df_answers, sample_dataset.gt, sample_dataset.n_classes
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCH):
        epoch_loss = train(model, optimizer, dataloader, reg_1=reg_1)
        ground_truth_estim = inference(model, eval_dataloader)
        train_loss = epoch_loss
        train_accuracy = np.mean(ground_truth_estim == sample_dataset.gt)
        wandb.log(
            dict(train_loss=train_loss, train_accuracy=train_accuracy, epoch=epoch)
        )
        print(f"{epoch} loss {train_loss:0.4f} accuracy laa: {train_accuracy}")

    ground_truth_estim = inference(model, eval_dataloader)

    print(ground_truth_estim, sample_dataset.gt)
    print(f"laa: {np.mean(ground_truth_estim == sample_dataset.gt)}")

    mv_result = majority_vote(sample_dataset.df_answers)

    print(mv_result[:10])
    print(sample_dataset.gt[:10])

    print(f"mv: {np.mean(mv_result == sample_dataset.gt)}")


if __name__ == "__main__":
    app()
