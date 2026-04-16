import numpy as np
import polars as pl
import torch
from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss
from torch.utils.data import DataLoader

from config import N_CLASSES
from data import ReviewDataset, prepare_data
from model import RatingModel


def train(train_df, test_df, user_df, biz_df, cfg: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arrays = prepare_data(train_df, test_df, user_df, biz_df)
    train_ds = ReviewDataset(arrays["train"], arrays)
    # test_ds solo para inferencia, sin targets
    test_ds = ReviewDataset(arrays["test"], arrays)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 4096),
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.get("batch_size", 4096) * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = RatingModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("wd", 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 30)
    )

    # # Comrobación rápida de que los datos se cargan correctamente y no contienen NaNs
    # batch = next(iter(train_loader))
    # print("Check data batch:")
    # print(f"Batch keys: {batch.keys()}")
    # print("user_meta  NaN:", batch["user_meta"].isnan().any().item())
    # print(f"Example user_meta row: {batch['user_meta'][0]}")
    # print("user_emb   NaN:", batch["user_emb"].isnan().any().item())
    # print(f"Example user_emb row: {batch['user_emb'][0][:5]}...")
    # print("biz_emb    NaN:", batch["biz_emb"].isnan().any().item())
    # print(f"Example biz_emb row: {batch['biz_emb'][0][:5]}...")
    # print("review     NaN:", batch["review_feats"].isnan().any().item())
    # print(f"Example review_feats row: {batch['review_feats'][0]}")
    # print("target min/max:", batch["target"].min().item(), batch["target"].max().item())
    # print(
    #     "user_meta  range:",
    #     batch["user_meta"].min().item(),
    #     batch["user_meta"].max().item(),
    # )
    # print(
    #     "biz_emb    range:",
    #     batch["biz_emb"].min().item(),
    #     batch["biz_emb"].max().item(),
    # )

    # model.eval()
    # with torch.no_grad():
    #     logits = model(
    #         batch["user_meta"],
    #         batch["user_emb"],
    #         batch["biz_emb"],
    #         batch["review_feats"],
    #     )
    #     print("logits NaN:", logits.isnan().any().item())
    #     print("logits range:", logits.min().item(), logits.max().item())

    for epoch in range(cfg.get("epochs", 30)):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                batch["user_meta"].to(device),
                batch["user_emb"].to(device),
                batch["biz_emb"].to(device),
                batch["review_feats"].to(device),
            )
            loss = corn_loss(logits, batch["target"].to(device), num_classes=N_CLASSES)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1:02d} | train_loss={running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "best_model.pt")
    return model, {"test_loader": test_loader}


def generate_submission(model, test_loader, test_df, device, path="submission.csv"):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["user_meta"].to(device),
                batch["user_emb"].to(device),
                batch["biz_emb"].to(device),
                batch["review_feats"].to(device),
            )
            preds = corn_label_from_logits(logits).cpu().numpy() + 1  # [1, 5]
            all_preds.append(preds)

    preds = np.concatenate(all_preds)
    (
        pl.DataFrame(
            {"review_id": test_df["review_id"], "stars": preds.astype(float)}
        ).write_csv(path)
    )
    print(f"Submission guardada en {path} ({len(preds)} filas)")


if __name__ == "__main__":

    train_df = pl.read_parquet("data/train.parquet")
    test_df = pl.read_parquet("data/test.parquet")
    user_df = pl.read_parquet("data/user_embeddings.parquet")
    biz_df = pl.read_parquet("data/business_embeddings.parquet")

    train_df = train_df.join(user_df, on="user_id", how="semi")
    train_df = train_df.join(biz_df, on="business_id", how="semi")

    cfg = {
        "batch_size": 4096,
        "lr": 1e-3,
        "wd": 1e-4,
        "epochs": 50,
    }

    model, data = train(train_df, test_df, user_df, biz_df, cfg)
