from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import requests
import yaml
from crowdkit.datasets import load_dataset


@dataclass
class Dataset:
    n_classes: int
    n_tasks: int
    n_workers: int
    df_answers: pd.DataFrame
    gt: pd.DataFrame


def encoder(ids: np.ndarray) -> dict:
    return dict((y, x) for x, y in enumerate(sorted(set(ids))))


def bluebirds(*args, **kwargs) -> Dataset:
    LABELS = "https://raw.githubusercontent.com/welinder/cubam/public/demo/bluebirds/labels.yaml"
    GT = (
        "https://raw.githubusercontent.com/welinder/cubam/public/demo/bluebirds/gt.yaml"
    )

    labels_response = requests.get(LABELS)
    gt_response = requests.get(GT)
    df_labels = pd.DataFrame(yaml.load(labels_response.text, Loader=yaml.FullLoader))

    df_labels = df_labels.melt(ignore_index=False)
    df_labels = df_labels.reset_index()
    df_labels.columns = ["task_id", "worker_id", "answer"]
    x2y = encoder(df_labels.task_id.values)
    df_labels.task_id = df_labels.task_id.apply(lambda x: x2y[x])
    x2y = encoder(df_labels.worker_id.values)
    df_labels.worker_id = df_labels.worker_id.apply(lambda x: x2y[x])
    df_labels.answer = df_labels.answer.astype(int)

    df_gt = pd.Series(yaml.load(gt_response.text, Loader=yaml.FullLoader))
    df_gt = df_gt.astype(int)

    dataset = Dataset(
        n_classes=2,
        n_tasks=len(df_gt),
        n_workers=df_labels.worker_id.nunique(),
        df_answers=df_labels,
        gt=df_gt.values,
    )
    return dataset


def releveance2(n_tasks: Optional[int] = 1000, *args, **kwargs) -> Dataset:

    df_labels, gt_response = load_dataset("relevance-2")

    if n_tasks:
        gt_response = gt_response.iloc[:n_tasks]
        df_labels = df_labels[df_labels.task.isin(gt_response.index)]

    gt_response = gt_response.reset_index()
    gt_response.columns = ["task_id", "answer"]

    df_labels.columns = ["worker_id", "task_id", "answer"]
    x2y = encoder(df_labels.task_id.values)
    df_labels.task_id = df_labels.task_id.apply(lambda x: x2y[x])
    gt_response.task_id = gt_response.task_id.apply(lambda x: x2y[x])

    x2y = encoder(df_labels.worker_id.values)
    df_labels.worker_id = df_labels.worker_id.apply(lambda x: x2y[x])
    df_labels.answer = df_labels.answer.astype(int)
    gt_response.answer = gt_response.answer.astype(int)

    gt_response = gt_response.sort_values(by="task_id")

    dataset = Dataset(
        n_classes=2,
        n_tasks=len(gt_response),
        n_workers=df_labels.worker_id.nunique(),
        df_answers=df_labels,
        gt=gt_response.answer.values,
    )
    return dataset


def classification_dataset_generator(
    n_workers: int = 20,
    n_tasks: int = 1000,
    good_workers_frac: float = 0.8,
    overlap: int = 3,
    good_probability: float = 0.9,
    bad_probability: float = 0.6,
    n_classes: int = 3,
) -> Dataset:
    agents = np.arange(n_workers)
    classes = np.arange(n_classes)
    true_labels = np.random.choice(classes, size=n_tasks)
    agent_quality = agents <= int(good_workers_frac * n_workers)
    answers = np.empty(shape=(n_tasks, overlap))
    workers_ids = list()
    task_ids = list()
    for task_id in range(n_tasks):
        agents_to_make = np.random.choice(agents, size=overlap, replace=False)
        for answer_id, agent_id in enumerate(agents_to_make):
            workers_ids.append(agent_id)
            task_ids.append(task_id)
            if agent_quality[agent_id]:
                answers[task_id, answer_id] = (
                    np.random.uniform(0, 1) <= good_probability
                )
            else:
                answers[task_id, answer_id] = np.random.uniform(0, 1) <= bad_probability
            if answers[task_id, answer_id] == 1:
                answers[task_id, answer_id] = true_labels[task_id]
            else:
                answers[task_id, answer_id] = np.random.choice(
                    classes[classes != true_labels[task_id]]
                )

    dataset = Dataset(
        n_classes=n_classes,
        n_workers=n_workers,
        n_tasks=n_tasks,
        df_answers=pd.DataFrame(
            {
                "task_id": task_ids,
                "answer": answers.flatten().astype(int),
                "worker_id": workers_ids,
            }
        ),
        gt=true_labels,
    )
    return dataset


DATASET_MAP = {
    "classification_dataset_generator": classification_dataset_generator,
    "releveance2": releveance2,
    "bluebirds": bluebirds,
}
