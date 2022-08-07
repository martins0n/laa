import random

import numpy as np
import pytest
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.crowddatasets import DATASET_MAP
from src.laa import LAA, CrowdDataset, inference, train


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)

@pytest.fixture()
def dummy_dataset():
    data = DATASET_MAP["classification_dataset_generator"]
    data = data(n_workers=2, n_classes=3, n_tasks=1000, overlap=2, good_probability=1, good_workers_frac=1)
    return data

def test_dummy(dummy_dataset):
    dataset = dummy_dataset
    batch_size = 10
    lr = 0.001
    n_epoch = 10
    gt = dataset.gt
    model = LAA(dataset.n_workers, dataset.n_classes, d_kl=1000)

    dataset = CrowdDataset(
        dataset.df_answers, dataset.gt, dataset.n_classes
    )

    dataloader = DataLoader(dataset, batch_size=batch_size)
    optimizer = Adam(model.parameters(), lr=lr)
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for _ in range(n_epoch):
        epoch_loss = train(model, optimizer, dataloader, reg_1=0.001)
        ground_truth_estim = inference(model, eval_dataloader)
    
    train_accuracy = np.mean(ground_truth_estim == gt)
    assert train_accuracy == 1.0