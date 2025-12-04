import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "prop_training_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "over_model.pkl")

# Columns we *never* want to feed into the model even if numeric
EXCLUDE_COLS = {
    "player",
    "player_id",
    "team",
    "opponent",
    "market",
    "book",
    "game_date",
    "game_time",
    "key",
    "prop_id",
    "edge_tag",
    "ml_prob_over",
    "label_over",  # target
    "hit",
    "over_hit",
    "target",
    "y",
}

# Order of preference for the target column
TARGET_CANDIDATES = [
    "label_over",  # <--- your current training file uses this
    "hit",
    "over_hit",
    "target",
    "y",
]


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def find_target_column(df: pd.DataFrame) -> str:
    """Pick the first target column that exists in the CSV."""
    available = set(df.columns)
    for col in TARGET_CANDIDATES:
        if col in available:
            log(f"Using target column: {col}")
            return col
    raise ValueError(
        "FATAL ERROR: Could not find a suitable target column. "
        f"Looked for: {TARGET_CANDIDATES}. "
        f"Available columns: {sorted(df.columns.tolist())}"
    )


def build_feature_frame(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # Drop target + excluded columns
    cols_to_drop = {target_col} | EXCLUDE_COLS
    cols_to_use = [c for c in df.columns if c not in cols_to_drop]

    # Keep only numeric / boolean columns
    feat_df = df[cols_to_use].select_dtypes(include=["number", "bool"]).copy()

    if feat_df.empty:
        raise RuntimeError(
            "No numeric features left after filtering. "
            "Check prop_training_data.csv and EXCLUDE_COLS."
        )

    return feat_df


def coerce_target(y_raw: pd.Series) -> np.ndarray:
    # Handle common encodings: 0/1, True/False, 'hit'/'miss', 'Y'/'N', etc.
    if y_raw.dtype == bool:
        return y_raw.astype(int).values

    if y_raw.dtype == object:
        mapping = {
            "hit": 1,
            "miss": 0,
            "over": 1,
            "under": 0,
            "y": 1,
            "n": 0,
            "true": 1,
            "false": 0,
            "t": 1,
            "f": 0,
        }
        y_processed = (
            y_raw.astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
        )
        if y_processed.isna().all():
            # Fall back to trying to cast directly
            y_processed = pd.to_numeric(y_raw, errors="coerce")

        return y_processed.fillna(0).astype(int).values

    # numeric-ish
    return pd.to_numeric(y_raw, errors="coerce").fillna(0).astype(int).values


def main() -> None:
    log("NBA Over-Model Training – v3.1")
    log("===================================")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Could not find training file '{DATA_PATH}'. "
            "Make sure prop_training_data.csv is in the same folder."
        )

    log(f"Loading training data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    log(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    target_col = find_target_column(df)
    y = coerce_target(df[target_col])

    X = build_feature_frame(df, target_col)
    feature_names = X.columns.tolist()

    # Impute + train
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    model = GradientBoostingClassifier(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
    )

    log("Training GLOBAL model...")
    model.fit(X_imputed, y)

    # Basic metrics
    y_prob = model.predict_proba(X_imputed)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)
    try:
        auc = roc_auc_score(y, y_prob)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(y, y_prob)

    log(f"GLOBAL Model – Accuracy: {acc:.4f} | AUC: {auc:.4f} | Brier: {brier:.4f}")

    # Make sure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    payload = {
        "model": model,
        "imputer": imputer,
        "feature_names": feature_names,
    }
    joblib.dump(payload, MODEL_PATH)
    log(f"Saved global model to {MODEL_PATH}")


if __name__ == "__main__":
    main()