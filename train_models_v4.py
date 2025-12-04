"""
train_models_v4.py (FIXED & CLEAN VERSION)

Multi-market trainer for PrizePicks-style Outlier apps.

This version:
- Handles missing columns safely
- Creates target column consistently
- Adds rolling features without errors
- Trains one model per market
- Saves models into /models folder
- Logs full metrics and progress
"""

import os
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


# ==============================================
# USER CONFIG
# ==============================================
DATA_PATH = "prop_training_data.csv"         # your main CSV
ACTUAL_COL = "actual_stat"                  # actual result
LINE_COL = "line"                           # line offered
TARGET_COL = "target_over"                  # binary target
CREATE_TARGET_FROM_ACTUAL = True

MARKET_COL = "market"
STAT_TYPE_COL = "stat_type"

PLAYER_COL = "player_name"
TEAM_COL = "team"
OPP_COL = "opponent"
MINUTES_COL = "minutes"
HOME_AWAY_COL = "home_away"
GAME_DATE_COL = "game_date"

OPTIONAL_NUMERIC_COLS = []

MIN_MINUTES = 10
MAX_SCORE_DIFF = 35
SCORE_DIFF_COL = "score_diff"

LOG_FILE = "train_log_v4.txt"


# ==============================================
# LOGGING
# ==============================================
def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = f"[{ts}] {msg}"
    print(out)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(out + "\n")


# ==============================================
# LOADING / CLEANING
# ==============================================
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training CSV not found at {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("CSV is empty.")

    if GAME_DATE_COL in df.columns:
        df[GAME_DATE_COL] = pd.to_datetime(df[GAME_DATE_COL], errors="coerce")

    return df


def basic_filters(df: pd.DataFrame) -> pd.DataFrame:
    original = len(df)

    if MINUTES_COL in df.columns:
        df = df[df[MINUTES_COL] >= MIN_MINUTES]

    if SCORE_DIFF_COL in df.columns:
        df = df[df[SCORE_DIFF_COL].abs() <= MAX_SCORE_DIFF]

    df = df.dropna(subset=[LINE_COL, ACTUAL_COL])

    log(f"Filtered {original} -> {len(df)} rows.")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    if CREATE_TARGET_FROM_ACTUAL:
        df[TARGET_COL] = (df[ACTUAL_COL] > df[LINE_COL]).astype(int)
    else:
        if TARGET_COL not in df:
            raise ValueError("TARGET_COL missing and CREATE_TARGET_FROM_ACTUAL=False")
    return df


# ==============================================
# ROLLING FEATURES
# ==============================================
def add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    if GAME_DATE_COL not in df:
        log("No game dates — skipping rolling features.")
        return df

    df = df.sort_values([PLAYER_COL, GAME_DATE_COL])

    def _roll(g):
        g["roll_avg_3"] = g[ACTUAL_COL].rolling(3, min_periods=1).mean().shift(1)
        g["roll_avg_5"] = g[ACTUAL_COL].rolling(5, min_periods=1).mean().shift(1)
        g["roll_avg_10"] = g[ACTUAL_COL].rolling(10, min_periods=1).mean().shift(1)
        g["roll_std_5"] = g[ACTUAL_COL].rolling(5, min_periods=1).std().shift(1)

        g["actual_minus_line"] = g[ACTUAL_COL] - g[LINE_COL]
        g["roll_diff_5"] = g["actual_minus_line"].rolling(5, min_periods=1).mean().shift(1)

        if MINUTES_COL in df:
            g["minutes_roll_5"] = g[MINUTES_COL].rolling(5, min_periods=1).mean().shift(1)

        return g

    df = df.groupby(PLAYER_COL, group_keys=False).apply(_roll)
    return df


# ==============================================
# FEATURE MATRIX
# ==============================================
def build_feature_matrix(df):
    y = df[TARGET_COL].values

    numeric = []
    for col in [
        LINE_COL, ACTUAL_COL,
        "roll_avg_3", "roll_avg_5", "roll_avg_10",
        "roll_std_5",
        "minutes_roll_5",
        "roll_diff_5"
    ]:
        if col in df:
            numeric.append(col)

    for col in OPTIONAL_NUMERIC_COLS:
        if col in df:
            numeric.append(col)

    categorical = []
    for col in [PLAYER_COL, TEAM_COL, OPP_COL, HOME_AWAY_COL]:
        if col in df:
            categorical.append(col)

    X = df[numeric + categorical].copy()

    return X, y, numeric, categorical


# ==============================================
# TRAINING A SINGLE MARKET
# ==============================================
def train_market(df, market):
    df_m = df[df[MARKET_COL] == market].copy()

    if len(df_m) < 150:
        log(f"Skipping {market} — too small ({len(df_m)} rows)")
        return {}

    X, y, num_cols, cat_cols = build_feature_matrix(df_m)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    base = GradientBoostingClassifier(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=42
    )

    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)

    model = Pipeline([
        ("preprocess", pre),
        ("clf", clf),
    ])

    log(f"Training {market} on {len(X_train)} rows...")
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "auc": float(roc_auc_score(y_test, probs)),
        "brier": float(brier_score_loss(y_test, probs)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    os.makedirs("models", exist_ok=True)
    out_path = f"models/model_over_{market.lower()}.pkl"
    joblib.dump(model, out_path)
    log(f"Saved model: {out_path}")

    log(f"{market} → AUC={metrics['auc']:.3f}  ACC={metrics['accuracy']:.3f}")

    return metrics


# ==============================================
# TRAIN ALL
# ==============================================
def train_all(df):
    markets = sorted(df[MARKET_COL].dropna().unique())
    log(f"Found markets: {markets}")

    out = {}
    for m in markets:
        metrics = train_market(df, m)
        if metrics:
            out[m] = metrics
    return out


# ==============================================
# MAIN
# ==============================================
def main():
    log("========= TRAIN MODELS START =========")

    df = load_data(DATA_PATH)
    df = basic_filters(df)
    df = create_target(df)
    df = add_rolling(df)

    df = df.dropna(subset=[TARGET_COL])

    metrics = train_all(df)

    log("===== SUMMARY =====")
    for m, mt in metrics.items():
        log(f"{m}: {mt}")

    log("========= TRAIN MODELS END =========")


if __name__ == "__main__":
    main()
