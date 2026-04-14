USER_META_COLS = [
    "average_stars",
    "user_review_count_log",
    "user_fans_log",
    "user_useful_log",
    "user_funny_log",
    "user_cool_log",
    "friends_count_log",
    "elite_count",
    "user_bias",
    "useful_per_review",
    "funny_per_review",
    "cool_per_review",
    "comp_total_log",
    "comp_writer_log",
    "comp_photos_log",
    "years_active",
]

# month/weekday/hour se encodean como sin+cos (3×2 = 6 features)
# is_weekend y elite se castean a float  → total 15 features
REVIEW_BASE_COLS = [
    "polarization",
    "funny_log",
    "useful_log",
    "cool_log",
    "avg_stars_user",
    "fans",
    "avg_stars_biz",
]
REVIEW_CYCLIC_COLS = {
    "month": 12,
    "weekday": 7,
    "hour": 24,
}
REVIEW_BOOL_COLS = ["is_weekend", "elite"]

USER_EMB_DIM = 64
BIZ_EMB_DIM = 384
USER_META_DIM = len(USER_META_COLS)  # 16
REVIEW_FEAT_DIM = (
    len(REVIEW_BASE_COLS) + 2 * len(REVIEW_CYCLIC_COLS) + len(REVIEW_BOOL_COLS)
)  # 7 + 6 + 2 = 15
N_CLASSES = 5
