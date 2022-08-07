### Commands

```bash
python -m src --dataset-name=classification_dataset_generator --dataset-kwargs='{"n_workers": 2, "n_tasks": 100, "overlap": 2, "n_classes": 3, "good_probability": 0.9, "good_workers_frac": 1}' --no-logging --lr=0.008 --n-epoch=100
```