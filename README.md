# Content-Based Recommendation System

This repository contains an end-to-end pipeline for a content-based rating prediction task.
The workflow is structured in three stages:

1. Build model-ready parquet datasets from raw CSV files using the notebook.
2. Train the neural model with the Python training pipeline.
3. Generate test predictions for submission.

## Project Overview

The model predicts review star ratings (1 to 5) by combining:

- User metadata features
- User embedding vectors
- Business embedding vectors
- Review/context engineered features

The training objective is ordinal classification using CORN loss.

## Repository Structure

- `notebooks/data_modeling.ipynb`: data preparation and feature engineering, including parquet export.
- `train.py`: training entry point.
- `main.py`: inference entry point to build submission files.
- `data.py`: dataset assembly, normalization, and dataloader utilities.
- `model.py`: neural architecture definition.
- `config.py`: feature configuration and model dimensions.
- `data/`: raw CSV files and generated parquet datasets.
- `results/`: prediction artifacts.

## Requirements

- Python 3.13 or higher (recommended by `pyproject.toml`)
- Sufficient RAM to load train/test datasets and embeddings
- Optional CUDA-enabled GPU for faster training

## Environment Setup

Install dependencies from `pyproject.toml`:

```bash
pip install -e .
```

If you prefer `uv`:

```bash
uv sync
```

## End-to-End Execution

### Step 1. Generate parquet datasets (mandatory first step)

Open and run all cells in:

- `notebooks/data_modeling.ipynb`

The notebook includes an initial `cd ..` so paths resolve from the project root.

This step generates the parquet files consumed by training and inference:

- `data/train.parquet`
- `data/test.parquet`
- `data/user_embeddings.parquet`
- `data/business_embeddings.parquet`

Do not run training or inference before these parquet files exist.

### Step 2. Train the model

Run:

```bash
python train.py
```

What happens in this stage:

- `train.py` loads parquet datasets.
- `data.py` builds normalized tensors and dataloaders.
- `model.py` defines and instantiates `RatingModel`.
- `config.py` provides feature dimensions and constants.

Output artifact:

- `best_model.pt`

### Step 3. Generate submission predictions

Run:

```bash
python main.py
```

`main.py` loads:

- The same parquet datasets generated in Step 1
- The trained model checkpoint (`best_model.pt` by default)

Output artifact:

- `submission.csv`

## Typical Run Order

Execute this sequence every time you rebuild the pipeline from raw data:

1. Run `notebooks/data_modeling.ipynb` (all cells).
2. Run `python train.py`.
3. Run `python main.py`.

## Notes

- If you want to use a different checkpoint for inference, update `MODEL_PATH` in `main.py`.
- Training uses GPU automatically when available; otherwise it falls back to CPU.
- Keep generated parquet files in `data/` to ensure both training and inference use the same feature schema.

## Troubleshooting

- **File not found errors for parquet files**: rerun `notebooks/data_modeling.ipynb` completely.
- **Model checkpoint not found**: ensure `train.py` finished successfully and produced `best_model.pt`.
- **Slow execution**: use a CUDA-enabled environment and verify PyTorch detects the GPU.
