# predict.py
import polars as pl
import torch
from torch.utils.data import DataLoader

from data import ReviewDataset, prepare_data
from model import RatingModel
from train import generate_submission

MODEL_PATH = "best_model.pt"
BATCH_SIZE = 8192
OUTPUT_PATH = "submission.csv"

if __name__ == "__main__":
    train_df = pl.read_parquet("data/train.parquet")
    test_df = pl.read_parquet("data/test.parquet")
    user_df = pl.read_parquet("data/user_embeddings.parquet")
    biz_df = pl.read_parquet("data/business_embeddings.parquet")

    # Pipeline de datos completo (para hacer la normalización consistente)
    arrays = prepare_data(train_df, test_df, user_df, biz_df)
    test_ds = ReviewDataset(arrays["test"], arrays)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RatingModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Modelo cargado desde {MODEL_PATH}")

    generate_submission(model, test_loader, test_df, device, path=OUTPUT_PATH)
