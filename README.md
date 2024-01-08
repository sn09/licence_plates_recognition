# licence_plates_recognition

Current repo is example of solving Licence Plates Recognition task using MLFlow
technologies.

## Description

This task is already solved with pretty high quality, in this project the key
focus is on using the majority of currently popular MLFlow techonologies.

**Used technologies:**

- `poetry`, `pre-commit`, `GitHub Actions` - code management and quality
- `dvc` - data management
- `hydra` - configs management
- `mlflow`, `tensorboard` - experiments and metrics tracking
- `PyTorch Lightning` - models and datasets implementation

## Quick start

### Project structure

- `cli.py` - CLI interface for training/inference (all parameters are managed by
  hydra configs)
- `configs` - hydra configs for: training, inference, models, data, tracking
- `recognintion_app` - source code for project: implementation of datasets,
  models, training and inference processes

### Install dependencies

```bash
poetry install --without dev
```

### Load data and models

```bash
dvc pull
```

### Model training and inference

All training parameters should be configured with configs: `configs/*`

**Training:**

```bash
python cli.py train
```

**Inference:**

```bash
python cli.py infer
```

## Development

**Installing dependencies:**

```bash
poetry install
```

**Pre-commit setup:**

```bash
pre-commit install
```
