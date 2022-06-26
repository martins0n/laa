import json
import random
from enum import Enum

import numpy as np
import torch
import typer
from torch.optim.adam import Adam
from torch.utils.data import DataLoader

from src.crowddatasets import DATASET_MAP
from src.laa import LAA, CrowdDataset, inference, majority_vote, train

app = typer.Typer()


class DatasetName(str, Enum):
    classification_dataset_generator = "classification_dataset_generator"
    releveance2 = "releveance2"
    bluebirds = "bluebirds"


@app.command()
def main(
    batch_size: int = 100,
    lr: float = 0.00001,
    n_epoch: int = 15,
    dataset_name: DatasetName = "classification_dataset_generator",
    dataset_kwargs: str = "{}",
    seed: int = 11,
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    BATCH_SIZE = batch_size
    LR = lr
    N_EPOCH = n_epoch

    dataset_kwargs = json.loads(dataset_kwargs)
    sample_dataset = DATASET_MAP[dataset_name](**dataset_kwargs)

    model = LAA(sample_dataset.n_workers, sample_dataset.n_classes)

    dataset = CrowdDataset(
        sample_dataset.df_answers, sample_dataset.gt, sample_dataset.n_classes
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = Adam(model.parameters(), lr=LR)

    for epoch in range(N_EPOCH):
        epoch_loss = train(model, optimizer, dataloader)
        print(f"{epoch} : {epoch_loss:0.4f}")
        ground_truth_estim = inference(model, eval_dataloader)
        print(f"{epoch} laa: {np.mean(ground_truth_estim == sample_dataset.gt)}")

    ground_truth_estim = inference(model, eval_dataloader)

    print(ground_truth_estim, sample_dataset.gt)
    print(f"laa: {np.mean(ground_truth_estim == sample_dataset.gt)}")

    mv_result = majority_vote(sample_dataset.df_answers)

    print(mv_result[:10])
    print(sample_dataset.gt[:10])

    print(f"mv: {np.mean(mv_result == sample_dataset.gt)}")


if __name__ == "__main__":
    app()
