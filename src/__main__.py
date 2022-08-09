import json
import random
from enum import Enum
from unittest.mock import MagicMock

import numpy as np
import torch
import typer
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from src.crowddatasets import DATASET_MAP, Dataset
from src.laa import LAA, CrowdDataset, inference, majority_vote, train

app = typer.Typer()


class DatasetName(str, Enum):
    classification_dataset_generator = "classification_dataset_generator"
    relevance2 = "relevance2"
    bluebirds = "bluebirds"


@app.command(name="train")
def _train(
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
    else:
        import wandb

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

    wandb.init(project="laa", config=config, tags=[dataset_name, "train"])

    model = LAA(
        sample_dataset.n_workers, sample_dataset.n_classes, d_kl=d_kl, reg_1=reg_1
    )

    dataset = CrowdDataset(
        sample_dataset.df_answers, sample_dataset.n_classes, sample_dataset.n_workers
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCH):
        epoch_loss = train(model, optimizer, dataloader)
        eval_loss, ground_truth_estim = inference(model, eval_dataloader)
        train_loss = epoch_loss
        train_accuracy = np.mean(
            [sample_dataset.gt[i] == value for i, value in ground_truth_estim.items()]
        )
        wandb.log(
            dict(train_loss=train_loss, train_accuracy=train_accuracy, epoch=epoch)
        )
        print(f"{epoch} loss {train_loss:0.4f} accuracy laa: {train_accuracy}")

    eval_loss, ground_truth_estim = inference(model, eval_dataloader)

    print(
        f"laa: {np.mean([sample_dataset.gt[i] == value for i, value in ground_truth_estim.items()])}"
    )

    mv_result = majority_vote(sample_dataset.df_answers)

    print(mv_result[:10])
    print(sample_dataset.gt[:10])

    print(f"mv: {np.mean(mv_result == sample_dataset.gt)}")


@app.command(name="inference")
def _inference(
    batch_size: int = 64,
    lr: float = 0.00001,
    d_kl: float = 0.001,
    reg_1: float = 0.001,
    n_epoch: int = 15,
    dataset_name: DatasetName = "bluebirds",
    dataset_kwargs: str = "{}",
    seed: int = 11,
    logging: bool = True,
    train_frac: float = 0.8,
    patience: int = 3,
    patience_epsilon: float = 0,
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    BATCH_SIZE = batch_size
    LR = lr
    N_EPOCH = n_epoch

    dataset_kwargs = json.loads(dataset_kwargs)
    sample_dataset: Dataset = DATASET_MAP[dataset_name](**dataset_kwargs)

    train_dataset, eval_dataset = sample_dataset.train_test_split(train_frac, seed)

    if not logging:
        wandb = MagicMock()
    else:
        import wandb

    config = dict(
        batch_size=batch_size,
        lr=lr,
        n_epoch=n_epoch,
        d_kl=d_kl,
        dataset_name=dataset_name,
        dataset_kwargs=dataset_kwargs,
        seed=seed,
        reg_1=reg_1,
        train_frac=train_frac,
        patience=patience,
        patience_epsilon=patience_epsilon,
    )

    wandb.init(project="laa", config=config, tags=[dataset_name, "inference"])

    model = LAA(
        train_dataset.n_workers, train_dataset.n_classes, d_kl=d_kl, reg_1=reg_1
    )

    dataset = CrowdDataset(
        sample_dataset.df_answers, sample_dataset.n_classes, sample_dataset.n_workers
    )
    train_torch_dataset = CrowdDataset(
        train_dataset.df_answers, train_dataset.n_classes, train_dataset.n_workers
    )
    eval_torch_dataset = CrowdDataset(
        eval_dataset.df_answers, train_dataset.n_classes, train_dataset.n_workers
    )

    dataloader = DataLoader(train_torch_dataset, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(
        eval_torch_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    full_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = Adam(model.parameters(), lr=LR)

    eval_loss_list = [1e10]
    patience_count = 0

    for epoch in range(N_EPOCH):
        train_loss = train(model, optimizer, dataloader)
        eval_loss, ground_truth_estim = inference(model, eval_dataloader)
        eval_loss_list.append(eval_loss)
        eval_accuracy = np.mean(
            [train_dataset.gt[i] == value for i, value in ground_truth_estim.items()]
        )
        wandb.log(
            dict(
                train_loss=train_loss,
                eval_accuracy=eval_accuracy,
                epoch=epoch,
                eval_loss=eval_loss,
            )
        )
        print(
            f"{epoch} train_loss {train_loss:0.4f}  eval_loss {eval_loss:0.4f} accuracy laa: {eval_accuracy}"
        )

        if eval_loss > min(eval_loss_list) + patience_epsilon:
            patience_count += 1
        else:
            patience_count = 0
        if patience_count >= patience - 1:
            break

    _, ground_truth_estim = inference(model, full_dataloader)

    full_accuracy = np.mean(
        [sample_dataset.gt[i] == value for i, value in ground_truth_estim.items()]
    )

    _, ground_truth_estim = inference(model, dataloader)

    train_accuracy = np.mean(
        [sample_dataset.gt[i] == value for i, value in ground_truth_estim.items()]
    )

    mv_result = majority_vote(sample_dataset.df_answers)

    wandb.log(
        {
            "full_accuracy": full_accuracy,
            "mv_accuracy": np.mean(mv_result == sample_dataset.gt),
        }
    )

    print(f"laa full accuracy: {full_accuracy}")
    print(f"laa train accuracy: {train_accuracy}")
    print(f"mv full: {np.mean(mv_result == sample_dataset.gt)}")
    print(
        f"mv train: {np.mean([sample_dataset.gt[i] == mv_result[i] for i, value in ground_truth_estim.items()])}"
    )


if __name__ == "__main__":
    app()
