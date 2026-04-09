"""
Sistema de Recomendación Basado en Contenido — MVP
===================================================
Predice el rating (stars) que un usuario otorgará a un negocio.
Evaluación: MAE entre el rating predicho y el voto real.

Dependencias:
    pip install polars lightgbm scikit-learn

Uso:
    python recommender.py

Salida:
    submission.csv  →  review_id, stars
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# 0. CONFIGURACIÓN
DATA_DIR = Path("./data")  # Cambia si los CSVs están en otra carpeta
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)
TRAIN_FILE = DATA_DIR / "train_reviews.csv"
TEST_FILE = DATA_DIR / "test_reviews.csv"
BUSINESS_FILE = DATA_DIR / "negocios.csv"
USER_FILE = DATA_DIR / "usuarios.csv"
OUTPUT_FILE = RESULTS_DIR / "submission.csv"

SEED = 42
N_FOLDS = 5

# 1. CARGA DE DATOS
print("Cargando datos...")

train = pl.read_csv(TRAIN_FILE, infer_schema_length=10_000)
test = pl.read_csv(TEST_FILE, infer_schema_length=10_000)

# Columnas de negocios: solo las numéricas (sin dicts)
business = pl.read_csv(
    BUSINESS_FILE,
    infer_schema_length=10_000,
    columns=[
        "business_id",
        "stars",
        "review_count",
        "is_open",
        "latitude",
        "longitude",
    ],
)

# Columnas de usuarios: solo las numéricas simples
user_cols = [
    "user_id",
    "review_count",
    "useful",
    "funny",
    "cool",
    "fans",
    "average_stars",
    "compliment_hot",
    "compliment_more",
    "compliment_profile",
    "compliment_cute",
    "compliment_list",
    "compliment_note",
    "compliment_plain",
    "compliment_cool",
    "compliment_funny",
    "compliment_writer",
    "compliment_photos",
]
users = pl.read_csv(USER_FILE, infer_schema_length=10_000, columns=user_cols)

print(f"  train:    {train.shape}")
print(f"  test:     {test.shape}")
print(f"  negocios: {business.shape}")
print(f"  usuarios: {users.shape}")

# 2. FEATURE ENGINEERING


def add_stats_from_train(df: pl.DataFrame, train: pl.DataFrame) -> pl.DataFrame:
    """Añade estadísticas de usuario/negocio calculadas sobre train."""

    # — Estadísticas de usuario desde reviews de entrenamiento —
    user_stats = train.group_by("user_id").agg(
        [
            pl.col("stars").mean().alias("user_mean_stars"),
            pl.col("stars").std().alias("user_std_stars"),
            pl.col("stars").count().alias("user_n_reviews"),
        ]
    )

    # — Estadísticas de negocio desde reviews de entrenamiento —
    biz_stats = train.group_by("business_id").agg(
        [
            pl.col("stars").mean().alias("biz_mean_stars"),
            pl.col("stars").std().alias("biz_std_stars"),
            pl.col("stars").count().alias("biz_n_reviews"),
        ]
    )

    df = df.join(user_stats, on="user_id", how="left")
    df = df.join(biz_stats, on="business_id", how="left")
    return df


def build_features(df: pl.DataFrame, train: pl.DataFrame) -> pl.DataFrame:
    """Une todas las fuentes y calcula features finales."""

    # Prefijamos columnas de business para evitar colisiones
    biz = business.rename(
        {
            "stars": "biz_catalog_stars",
            "review_count": "biz_catalog_review_count",
        }
    )

    # Prefijamos columnas de usuario
    usr = users.rename(
        {
            "review_count": "user_catalog_review_count",
            "useful": "user_catalog_useful",
            "funny": "user_catalog_funny",
            "cool": "user_catalog_cool",
            "fans": "user_catalog_fans",
            "average_stars": "user_catalog_avg_stars",
        }
    )
    # Renombrar compliments en bloque
    compliment_cols = [c for c in usr.columns if c.startswith("compliment_")]
    usr = usr.rename({c: f"user_{c}" for c in compliment_cols})

    df = df.join(biz, on="business_id", how="left")
    df = df.join(usr, on="user_id", how="left")
    df = add_stats_from_train(df, train)

    # Global mean (para rellenar NaN en cold-start)
    global_mean = train["stars"].mean()

    # Features derivadas
    df = df.with_columns(
        [
            # Diferencia entre la media del usuario y la del negocio en catálogo
            (pl.col("user_catalog_avg_stars") - pl.col("biz_catalog_stars")).alias(
                "user_biz_star_diff"
            ),
            # Popularidad combinada (log)
            (pl.col("biz_catalog_review_count").log(base=10)).alias(
                "biz_log_review_count"
            ),
            (pl.col("user_catalog_review_count").log(base=10)).alias(
                "user_log_review_count"
            ),
        ]
    )

    # Rellenar nulos en estadísticas transaccionales con la media global
    df = df.with_columns(
        [
            pl.col("user_mean_stars").fill_null(global_mean),
            pl.col("biz_mean_stars").fill_null(global_mean),
            pl.col("user_std_stars").fill_null(0.0),
            pl.col("biz_std_stars").fill_null(0.0),
            pl.col("user_n_reviews").fill_null(0),
            pl.col("biz_n_reviews").fill_null(0),
        ]
    )

    return df


print("Construyendo features...")
train_feat = build_features(train, train)
test_feat = build_features(test, train)  # Estadísticas siempre de train

# 3. SELECCIÓN DE FEATURES PARA EL MODELO
FEATURE_COLS = [
    # Del negocio (catálogo)
    "biz_catalog_stars",
    "biz_catalog_review_count",
    "is_open",
    "latitude",
    "longitude",
    "biz_log_review_count",
    # Del usuario (catálogo)
    "user_catalog_review_count",
    "user_catalog_useful",
    "user_catalog_funny",
    "user_catalog_cool",
    "user_catalog_fans",
    "user_catalog_avg_stars",
    "user_log_review_count",
    # Compliments
    "user_compliment_hot",
    "user_compliment_more",
    "user_compliment_profile",
    "user_compliment_cute",
    "user_compliment_list",
    "user_compliment_note",
    "user_compliment_plain",
    "user_compliment_cool",
    "user_compliment_funny",
    "user_compliment_writer",
    "user_compliment_photos",
    # Estadísticas transaccionales del train
    "user_mean_stars",
    "user_std_stars",
    "user_n_reviews",
    "biz_mean_stars",
    "biz_std_stars",
    "biz_n_reviews",
    # Features derivadas
    "user_biz_star_diff",
    # Contexto de la reseña
    "useful",
    "funny",
    "cool",
]

# Filtrar sólo las que existen en el DataFrame (seguridad ante cambios de schema)
FEATURE_COLS = [c for c in FEATURE_COLS if c in train_feat.columns]
print(f"Features usadas ({len(FEATURE_COLS)}): {FEATURE_COLS}")

X = train_feat[FEATURE_COLS].to_numpy().astype(np.float32)
y = train_feat["stars"].to_numpy().astype(np.float32)
X_test = test_feat[FEATURE_COLS].to_numpy().astype(np.float32)

# 4. ENTRENAMIENTO CON VALIDACIÓN CRUZADA
lgb_params = {
    "objective": "regression_l1",  # MAE directo
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "n_estimators": 1000,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "seed": SEED,
}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))
fold_maes = []

print(f"\nEntrenando con {N_FOLDS}-fold CV...")

for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(X_test) / N_FOLDS

    mae = mean_absolute_error(y_val, oof_preds[val_idx])
    fold_maes.append(mae)
    print(f"  Fold {fold} — MAE: {mae:.4f}  |  best iter: {model.best_iteration_}")

overall_mae = mean_absolute_error(y, oof_preds)
print(f"\n{'='*40}")
print(f"  OOF MAE global:  {overall_mae:.4f}")
print(f"  MAE por fold:    {[round(m,4) for m in fold_maes]}")
print(f"{'='*40}")

# Clampear predicciones al rango válido [1, 5]
test_preds = np.clip(test_preds, 1.0, 5.0)

# 5. GUARDAR SUBMISSION
submission = pl.DataFrame(
    {
        "review_id": test_feat["review_id"].to_list(),
        "stars": test_preds.tolist(),
    }
)

submission.write_csv(OUTPUT_FILE)
print(f"\nSubmission guardada en: {OUTPUT_FILE}")
print(submission.head(10))
