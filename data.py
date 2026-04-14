import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from config import (
    REVIEW_BASE_COLS,
    REVIEW_BOOL_COLS,
    REVIEW_CYCLIC_COLS,
    USER_META_COLS,
)


def add_cyclic_features(df: pl.DataFrame) -> pl.DataFrame:
    """Reemplaza month/weekday/hour por sus codificaciones sin/cos."""
    exprs = []
    for col, period in REVIEW_CYCLIC_COLS.items():
        exprs += [
            (2 * np.pi * pl.col(col) / period).sin().alias(f"{col}_sin"),
            (2 * np.pi * pl.col(col) / period).cos().alias(f"{col}_cos"),
        ]
    return df.with_columns(exprs).drop(list(REVIEW_CYCLIC_COLS.keys()))


def cast_bools(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([pl.col(c).cast(pl.Float32) for c in REVIEW_BOOL_COLS])


def compute_stats(df: pl.DataFrame, cols: list[str]) -> dict:
    """Calcula mean/std sobre las columnas indicadas (para normalizar con train)."""
    stats = {}
    for col in cols:
        series = df[col].cast(pl.Float64)
        stats[col] = {"mean": series.mean(), "std": series.std() + 1e-8}
    return stats


def normalize(df: pl.DataFrame, stats: dict) -> pl.DataFrame:
    return df.with_columns(
        [
            ((pl.col(c) - stats[c]["mean"]) / stats[c]["std"]).cast(pl.Float32).alias(c)
            for c in stats
        ]
    )


def emb_array(df: pl.DataFrame, id_col: str, emb_col: str) -> tuple[dict, np.ndarray]:
    """
    Devuelve:
      - id2idx: dict {id_str => row_index}
      - matrix: np.ndarray shape (N, D)
      - unk_idx: índice a usar para IDs desconocidos (la fila N, que es la media de todas las embeddings)
    """
    ids = df[id_col].to_list()
    embs = np.stack(df[emb_col].to_numpy(), axis=0)  # (N, D)

    # Fila UNK = media de todos los embeddings → índice N
    unk_row = embs.mean(axis=0, keepdims=True)
    embs = np.concatenate([embs, unk_row], axis=0)  # (N+1, D)

    id2idx = {uid: i for i, uid in enumerate(ids)}  # IDs conocidos → 0..N-1
    # Desconocidos → N (la fila UNK)
    unk_idx = len(ids)

    return id2idx, embs, unk_idx


def prepare_data(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    user_df: pl.DataFrame,
    biz_df: pl.DataFrame,
) -> dict:
    """
    Devuelve un dict con todo lo necesario para construir los datasets: arrays de embeddings, arrays de features normalizadas, targets...
    """

    # 1. Embeddings => lookup tables (fijos, sin normalizar)
    user_id2idx, user_emb_matrix, user_unk = emb_array(user_df, "user_id", "embedding")
    biz_id2idx, biz_emb_matrix, biz_unk = emb_array(biz_df, "business_id", "embedding")

    # 2. Metadatos de usuario: normalizar con estadísticas de train (join para tener los meta de cada reseña de train)
    user_meta_df = user_df.select(["user_id"] + USER_META_COLS)
    user_meta_stats = compute_stats(user_meta_df, USER_META_COLS)

    user_meta_train = normalize(user_meta_df, user_meta_stats)
    user_meta_matrix_train = (
        user_meta_train
        # .sort("user_id")  # orden consistente con user_id2idx
        .select(USER_META_COLS)
        .to_numpy()
        .astype(np.float32)
    )
    # Misma normalización para test (ya está en user_df global)
    user_meta_matrix = (
        normalize(user_meta_df, user_meta_stats)
        .select(USER_META_COLS)
        .to_numpy()
        .astype(np.float32)
    )

    # 3. Review features: cyclic + bool + normalización
    def process_reviews(df: pl.DataFrame, stats=None):
        df = add_cyclic_features(df)
        df = cast_bools(df)

        all_feat_cols = (
            REVIEW_BASE_COLS
            + [f"{c}_sin" for c in REVIEW_CYCLIC_COLS]
            + [f"{c}_cos" for c in REVIEW_CYCLIC_COLS]
            + REVIEW_BOOL_COLS
        )

        if stats is None:
            stats = compute_stats(df, all_feat_cols)

        df = normalize(df, stats)
        # print(f"valores nulos por columna:\n{df.select(all_feat_cols).null_count()}")
        # print(
        #     f"Suma de cada columna (debería ser ~0):\n{df.select(all_feat_cols).sum()}"
        # )
        feats = df.select(all_feat_cols).to_numpy().astype(np.float32)
        return feats, stats

    train_review_feats, review_stats = process_reviews(train_df)
    test_review_feats, _ = process_reviews(test_df, stats=review_stats)

    # 4. User meta stats: reconstruir matrix con orden de user_id2idx
    sorted_user_ids = sorted(user_id2idx, key=lambda x: user_id2idx[x])
    user_meta_norm = (
        normalize(user_meta_df, user_meta_stats)
        # .sort("user_id")
        .select(USER_META_COLS)
        .to_numpy()
        .astype(np.float32)
    )
    # Fila UNK = media de las features
    unk_meta_row = user_meta_norm.mean(axis=0, keepdims=True)
    user_meta_norm = np.concatenate([user_meta_norm, unk_meta_row], axis=0)

    # 5. Targets
    train_targets = (train_df["stars"].cast(pl.Float32).to_numpy() - 1).astype(
        np.int64
    )  # [0, 4]
    # test_targets = test_df["stars"].cast(pl.Float32).to_numpy() - 1

    # 6. Índices de usuario y negocio por reseña
    train_user_idx = np.array(
        [user_id2idx.get(u, user_unk) for u in train_df["user_id"].to_list()]
    )
    train_biz_idx = np.array(
        [biz_id2idx.get(b, biz_unk) for b in train_df["business_id"].to_list()]
    )
    test_user_idx = np.array(
        [user_id2idx.get(u, user_unk) for u in test_df["user_id"].to_list()]
    )
    test_biz_idx = np.array(
        [biz_id2idx.get(b, biz_unk) for b in test_df["business_id"].to_list()]
    )

    return {
        "user_emb_matrix": user_emb_matrix,  # (n_users, 64)
        "biz_emb_matrix": biz_emb_matrix,  # (n_biz, 384)
        "user_meta_matrix": user_meta_norm,  # (n_users, 16)
        "train": {
            "user_idx": train_user_idx,
            "biz_idx": train_biz_idx,
            "review_feats": train_review_feats,
            "targets": train_targets,
        },
        "test": {
            "user_idx": test_user_idx,
            "biz_idx": test_biz_idx,
            "review_feats": test_review_feats,
            # "targets": test_targets,
        },
    }


class ReviewDataset(Dataset):
    def __init__(self, split_data: dict, arrays: dict):
        self.user_emb = arrays["user_emb_matrix"]
        self.biz_emb = arrays["biz_emb_matrix"]
        self.user_meta = arrays["user_meta_matrix"]

        self.user_idx = split_data["user_idx"]
        self.biz_idx = split_data["biz_idx"]
        self.review_feats = split_data["review_feats"]
        self.targets = split_data.get("targets")

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, idx):
        u = self.user_idx[idx]
        b = self.biz_idx[idx]
        item = {
            "user_meta": torch.from_numpy(self.user_meta[u]),
            "user_emb": torch.from_numpy(self.user_emb[u]),
            "biz_emb": torch.from_numpy(self.biz_emb[b]),
            "review_feats": torch.from_numpy(self.review_feats[idx]),
        }
        if self.targets is not None:
            item["target"] = torch.tensor(int(self.targets[idx]), dtype=torch.long)
        return item
