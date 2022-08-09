# Aggregating Crowd Wisdoms with Label-aware Autoencoders

[Paper](https://www.researchgate.net/publication/318830014_Aggregating_Crowd_Wisdoms_with_Label-aware_Autoencoders)

![](assets/2022-08-10-01-18-43.png)

## How inference works:

* Split the dataset into training and validation parts
* Start training model
* When loss on validation part starts to increase authors stop training
    * No explicit stopping strategy has been given by the authors
    * So we stop training after max patience step succeeded
* To find a final solution we should use hyperparameters search using validation loss

### Accuracy - validation loss:

| dataset | LAA-B | Majority Vote | LAA-B(Paper) | 
| ---- | --- | --- | --- |
| bluebirds | 0.8056 | 0.7593 | 0.889 | 
| syntetic overlap 3 n_classes 3 n_tasks 5000 | 0.8932 | 0.9006 | 
| syntetic overlap 2 n_classes 3 n_tasks 5000 | 0.777 | 0.7892 | 

## P.S.

It looks like you could achieve greater results in case of using golden labeled data for the best model search:

### Accuracy - validation via golden dataset:
| dataset | LAA-B | Majority Vote |
| ---- | --- | --- |
| bluebirds | 0.907 | 0.7674 | 
| syntetic overlap 3 n_classes 3 n_tasks 5000 | 0.8998 | 0.8985 |
| syntetic overlap 2 n_classes 3 n_tasks 5000 | 0.8085 | 0.7822 |

## Wandb Sweeps and best run

* [bluebirds](https://wandb.ai/martins0n/laa/sweeps/i87vtebk)

```bash
python -m src inference --dataset-name=classification_dataset_generator  --no-logging --dataset-name=bluebirds --batch-size=33 --d-kl=0.0014471807379961906 --lr=0.1038585651189214 --n-epoch=152 --patience=15 --reg-1=6.420109871402643e-05
```

* [syntetic overlap 3 n_classes 3](https://wandb.ai/martins0n/laa/sweeps/4kk1c6ei)

```bash
python -m src inference --dataset-name=classification_dataset_generator --dataset-kwargs="{\"n_workers\": 100, \"n_tasks\": 5000, \"overlap\": 3, \"n_classes\": 3, \"good_probability\": 0.9, \"good_workers_frac\": 0.6, \"bad_probability\": 0.6}" --batch-size=93 --d-kl=0.007161637022033341 --lr=0.12966202803623433 --n-epoch=78 --patience=2 --reg-1=4.9742867272127485e-05 --no-logging
```

* [syntetic overlap 2 n_classes 3](https://wandb.ai/martins0n/laa/sweeps/nr1ib8n8)

```bash
python -m src inference --dataset-name=classification_dataset_generator --dataset-kwargs="{\"n_workers\": 100, \"n_tasks\": 5000, \"overlap\": 2, \"n_classes\": 3, \"good_probability\": 0.9, \"good_workers_frac\": 0.6, \"bad_probability\": 0.6}" --batch-size=86 --d-kl=0.009459693322090797 --lr=0.003334278174792921 --n-epoch=242 --patience=8 --reg-1=0.0001390032305118152 --no-logging
```
