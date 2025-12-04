"""
train_over_model.py - v3.0
----------------------------------
Unified training script for the NBA Outlier-style app.

What this script does
---------------------
• Reads the main training CSV (same one the app uses).
• Cleans & filters rows to just finished bets with a known result.
• Auto-detects the target column (hit / miss).
• Builds a robust sklearn Pipeline with:
    - SimpleImputer for numeric & categorical features
    - StandardScaler for numeric features
    - OneHotEncoder for categorical features
    - GradientBoostingClassifier (solid tabular model)
• Trains:
    1) A GLOBAL over_model.pkl using ALL markets.
    2) PER-MARKET models in the models/ folder, using the same
       filenames the Streamlit app expects (PER_MARKET_MODEL_PATHS).
• Prints nice metrics (AUC, accuracy, etc.) per model so you can
  see how training is going.

Usage
-----
• Run it from the same folder as your app:
      python train_over_model.py

• Make sure the CSV below (TRAINING_DATA_FILE) exists and contains
  your historical props + results. The app already writes this file
  if you kept that logic from previous versions.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# =====================================================
# PATHS & CONSTANTS (keep in sync with app)
# =====================================================

DATA_DIR = "."
MODEL_DIR = os.path.join(".", "models")

# This is the same file name the app usually writes to.  If your
# file is different, change this value, but keep it in the same
# folder as the app.
TRAINING_DATA_FILE = os.path.join(DATA_DIR, "prop_training_data.csv")

# Single "global" model (fallback if a per-market model is missing)
OVER_MODEL_PATH = os.path.join(MODEL_DIR, "over_model.pkl")

# Per-market model filenames. These are the filenames the app expects.
PER_MARKET_MODEL_PATHS: Dict[str, str] = {
    "points":   os.path.join(MODEL_DIR, "over_model_points.pkl"),
    "rebounds": os.path.join(MODEL_DIR, "over_model_rebounds.pkl"),
    "assists":  os.path.join(MODEL_DIR, "over_model_assists.pkl"),
    "pra":      os.path.join(MODEL_DIR, "over_model_pra.pkl"),
    "pr":       os.path.join(MODEL_DIR, "over_model_pr.pkl"),
    "ra":       os.path.join(MODEL_DIR, "over_model_ra.pkl"),
    "threes":   os.path.join(MODEL_DIR, "over_model_threes.pkl"),
    "points+rebounds+assists": os.path.join(MODEL_DIR, "over_model_pra.pkl"),
}

# Minimum rows required to train a per-market model.  If a market has
# fewer rows than this, we will SKIP training a separate model and
# the app will fall back to the global over_model.pkl for that market.
MIN_ROWS_PER_MARKET = 150

# Candidate columns to use as target (depends on how your CSV is built)
POSSIBLE_TARGET_COLS = ["hit", "over_hit", "target", "y"]

# Candidate columns that tell us which market this row belongs to.
POSSIBLE_MARKET_COLS = ["market_key", "market", "stat_type", "stat_type_full"]

# Columns that clearly aren't features (these will be dropped)
ALWAYS_DROP_COLS = {
    "bet_id",
    "row_id",
    "game_date",
    "created_at",
    "resolved_at",
    "result",          # string like "win/lose/push"
    "bet_result",      # if you keep a text/result column
    "over_under_side", # "over"/"under"
}

# =====================================================
# LOGGING HELPERS
# =====================================================

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# =====================================================
# DATA LOADING / PREP
# =====================================================

def find_target_column(df: pd.DataFrame) -> str:
    """
    Try to auto-detect the target column.
    We look for the first candidate that exists and is binary.
    """
    for col in POSSIBLE_TARGET_COLS:
        if col in df.columns:
            unique_vals = sorted(df[col].dropna().unique().tolist())
            # Accept binary (0/1, True/False, "win"/"lose" style)
            if len(unique_vals) <= 3:
                log(f"Using '{col}' as target column (values: {unique_vals})")
                return col
    raise ValueError(
        f"Could not find a suitable target column. "
        f"Looked for: {POSSIBLE_TARGET_COLS}. "
        f"Available columns: {list(df.columns)}"
    )


def find_market_column(df: pd.DataFrame) -> str:
    """
    Try to auto-detect which column stores the market name
    (points, rebounds, PRA, etc).
    """
    for col in POSSIBLE_MARKET_COLS:
        if col in df.columns:
            log(f"Using '{col}' as market column.")
            return col
    raise ValueError(
        f"Could not find a market column. "
        f"Looked for: {POSSIBLE_MARKET_COLS}. "
        f"Available columns: {list(df.columns)}"
    )


def load_training_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        log(f"ERROR: Training data file not found: {path}")
        log("Make sure your app is set up to write prop_training_data.csv.")
        sys.exit(1)

    log(f"Loading training data from: {path}")
    df = pd.read_csv(path)

    if df.empty:
        log("ERROR: Training data file is empty. Nothing to train on.")
        sys.exit(1)

    log(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df


def filter_finished_bets(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Keep only rows where the target is NOT null.
    """
    before = len(df)
    df2 = df.dropna(subset=[target_col]).copy()
    after = len(df2)
    log(f"Filtered finished bets: {before:,} -> {after:,} rows with non-null '{target_col}'.")

    if after < 500:
        log(
            "WARNING: Less than 500 finished bets in the training data. "
            "Model quality may be limited; consider collecting more history."
        )
    return df2


def make_feature_split(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Determine numeric & categorical feature columns and return X, y.
    """
    # Drop obvious non-features
    to_drop = set()
    for c in ALWAYS_DROP_COLS:
        if c in df.columns:
            to_drop.add(c)
    if target_col in df.columns:
        to_drop.add(target_col)

    df = df.drop(columns=list(to_drop), errors="ignore")

    # Separate target
    y = df[target_col] if target_col in df.columns else None

    # Everything else (except target) is candidate features
    if y is not None:
        X = df.drop(columns=[target_col], errors="ignore")
    else:
        X = df

    # Identify numeric vs categorical
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    log(f"Numeric features  ({len(numeric_features)}): {numeric_features}")
    log(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    if not numeric_features and not categorical_features:
        raise ValueError("No features found after filtering. Check your CSV columns.")

    return X, y, numeric_features, categorical_features


# =====================================================
# MODEL BUILDING
# =====================================================

def build_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """
    Build a sklearn Pipeline with preprocessing + GradientBoosting model.
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ]
    )

    clf = GradientBoostingClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", clf),
        ]
    )

    return model


def train_single_model(
    df: pd.DataFrame,
    target_col: str,
    description: str,
) -> Tuple[Optional[Pipeline], Dict[str, float]]:
    """
    Train a single model (global or per-market) and return the model + metrics.
    If df has fewer than MIN_ROWS_PER_MARKET rows, return (None, {}).
    """
    if len(df) < MIN_ROWS_PER_MARKET and description != "GLOBAL":
        # For per-market models we require a minimum number of rows
        log(
            f"[{description}] Skipping: only {len(df)} rows (< {MIN_ROWS_PER_MARKET}). "
            "This market will fall back to the GLOBAL model."
        )
        return None, {}

    X, y, numeric_features, categorical_features = make_feature_split(df, target_col)

    # Convert target to 0/1 integers if needed
    if y.dtype == "object":
        # Try to map typical strings like "win"/"lose", "W"/"L"
        lower_vals = y.str.lower().fillna("")
        if set(lower_vals.unique()) <= {"win", "loss", "lose", "push", "w", "l"}:
            mapped = lower_vals.map(
                {
                    "win": 1,
                    "w": 1,
                    "loss": 0,
                    "lose": 0,
                    "l": 0,
                    "push": 0,
                }
            )
            y = mapped.fillna(0).astype(int)
        else:
            # Fallback: factorize
            codes, _ = pd.factorize(y)
            y = codes

    # If y is boolean, cast to int
    if y.dtype == bool:
        y = y.astype(int)

    # Simple train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    model = build_pipeline(numeric_features, categorical_features)

    log(f"[{description}] Fitting model on {len(X_train):,} rows...")
    model.fit(X_train, y_train)

    # Evaluate
    metrics: Dict[str, float] = {}
    y_pred = model.predict(X_val)

    try:
        y_proba = model.predict_proba(X_val)[:, 1]
    except Exception:
        # Some classifiers might not have predict_proba; in that case,
        # approximate using decision_function if available.
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_val)
            # Min-max normalize to [0, 1]
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        else:
            y_proba = None

    metrics["accuracy"] = float(accuracy_score(y_val, y_pred))

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_val, y_proba))
        except Exception:
            metrics["roc_auc"] = float("nan")

        try:
            metrics["log_loss"] = float(log_loss(y_val, y_proba))
        except Exception:
            metrics["log_loss"] = float("nan")

    log(f"[{description}] Accuracy: {metrics.get('accuracy', float('nan')):.3f}")
    if "roc_auc" in metrics:
        log(f"[{description}] ROC AUC: {metrics['roc_auc']:.3f}")
    if "log_loss" in metrics:
        log(f"[{description}] Log Loss: {metrics['log_loss']:.3f}")

    # Optional: print a small classification report
    try:
        report = classification_report(y_val, y_pred, digits=3)
        log(f"[{description}] Classification report:\n{report}")
    except Exception as e:
        log(f"[{description}] Could not compute classification report: {e}")

    return model, metrics


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    log("====================================")
    log("  NBA Over-Model Training - v3.0")
    log("====================================")

    df = load_training_data(TRAINING_DATA_FILE)

    # Auto-detect target & market columns
    target_col = find_target_column(df)
    market_col = find_market_column(df)

    # Filter to finished bets
    df = filter_finished_bets(df, target_col)

    # -------------------------------------------------
    # 1) GLOBAL MODEL
    # -------------------------------------------------
    log("Training GLOBAL over_model (all markets combined)...")
    global_model, global_metrics = train_single_model(
        df=df,
        target_col=target_col,
        description="GLOBAL",
    )

    if global_model is None:
        log("ERROR: Could not train global model. Aborting.")
        sys.exit(1)

    joblib.dump(global_model, OVER_MODEL_PATH)
    log(f"Saved GLOBAL model to: {OVER_MODEL_PATH}")

    # -------------------------------------------------
    # 2) PER-MARKET MODELS
    # -------------------------------------------------
    log("Training PER-MARKET models...")
    per_market_results = {}

    # Normalize market strings a bit to increase the chance of matches
    def normalize_market_value(val: str) -> str:
        if not isinstance(val, str):
            return ""
        v = val.strip().lower()
        # Common aliases
        if v in {"pts", "point", "points"}:
            return "points"
        if v in {"reb", "rebound", "rebounds"}:
            return "rebounds"
        if v in {"ast", "assist", "assists"}:
            return "assists"
        if v in {"3ptm", "3pm", "3s", "3 pointers made", "three pointers made", "threes"}:
            return "threes"
        if v in {"pra", "points+rebounds+assists", "points rebounds assists"}:
            return "points+rebounds+assists"
        if v in {"pr", "points+rebounds"}:
            return "pr"
        if v in {"ra", "rebounds+assists"}:
            return "ra"
        return v

    # Create a normalized market column for grouping
    norm_col = f"{market_col}_normalized"
    df[norm_col] = df[market_col].astype(str).map(normalize_market_value)

    for mk, model_path in PER_MARKET_MODEL_PATHS.items():
        # mk is one of the keys above ("points", "rebounds", etc.)
        df_mk = df[df[norm_col] == mk].copy()
        n_rows = len(df_mk)
        log(f"Market '{mk}': {n_rows:,} rows found.")

        if n_rows < MIN_ROWS_PER_MARKET:
            log(
                f"Skipping separate model for '{mk}' (only {n_rows} rows). "
                "This market will use the GLOBAL model in the app."
            )
            continue

        model, metrics = train_single_model(
            df=df_mk,
            target_col=target_col,
            description=f"MARKET={mk}",
        )
        if model is None:
            continue

        joblib.dump(model, model_path)
        log(f"[{mk}] Saved model to: {model_path}")
        per_market_results[mk] = metrics

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    log("Training complete.")
    summary = {
        "global_model_path": OVER_MODEL_PATH,
        "global_metrics": global_metrics,
        "per_market_results": per_market_results,
    }

    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Wrote training summary to: {summary_path}")

    log("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Training interrupted by user.")
        sys.exit(1)
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        raise

