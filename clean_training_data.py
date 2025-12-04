"""
clean_training_data.py

Cleans prop_training_data.csv:
- Backs up original file
- Fills missing player_id using player name via nba_api
- Drops rows that can never be labeled (past games with no player_id, bad markets, missing line)
- Drops duplicate props
- Saves cleaned CSV back to prop_training_data.csv

Run:
    python clean_training_data.py
"""

import os
from datetime import datetime, date

import pandas as pd
from nba_api.stats.static import players as nba_players


CSV_PATH = "prop_training_data.csv"
BACKUP_PATH = "prop_training_data_backup_before_clean.csv"

LABEL_COL = "label_over"
DATE_COL = "game_date_dt"
PLAYER_ID_COL = "player_id"
PLAYER_NAME_COL = "player"
MARKET_COL = "market"
LINE_COL = "line"

# Core markets we actually model
ALLOWED_MARKETS = {"points", "rebounds", "assists", "pra", "ra", "fs"}


# -------------------------------------------------------
# Logging helpers
# -------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# -------------------------------------------------------
# Date / player helpers
# -------------------------------------------------------
def parse_date(x):
    """Safely convert string -> date or None."""
    if pd.isna(x):
        return None
    try:
        d = pd.to_datetime(str(x), errors="coerce")
        if pd.isna(d):
            return None
        return d.date()
    except Exception:
        return None


def build_player_name_index():
    """full_name.lower() -> player_id"""
    plist = nba_players.get_players()
    index = {}
    for p in plist:
        name = str(p.get("full_name", "")).strip().lower()
        pid = p.get("id")
        if name and pid is not None:
            index[name] = pid
    log(f"Built player name index for {len(index)} players.")
    return index


def fill_missing_player_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Fill player_id where possible from the 'player' column."""
    if PLAYER_ID_COL not in df.columns:
        log(f"Column '{PLAYER_ID_COL}' not found; cannot fill player_id.")
        return df
    if PLAYER_NAME_COL not in df.columns:
        log(f"Column '{PLAYER_NAME_COL}' not found; cannot fill player_id from names.")
        return df

    missing_mask = df[PLAYER_ID_COL].isna() & df[PLAYER_NAME_COL].notna()
    total_missing = missing_mask.sum()
    log(f"Rows with missing player_id but a name: {total_missing}")

    if total_missing == 0:
        return df

    name_index = build_player_name_index()
    filled = 0

    for idx in df[missing_mask].index:
        name = str(df.at[idx, PLAYER_NAME_COL]).strip().lower()
        if not name:
            continue
        pid = name_index.get(name)
        if pid:
            df.at[idx, PLAYER_ID_COL] = pid
            filled += 1

    log(f"Filled player_id for {filled} rows using player names.")
    return df


# -------------------------------------------------------
# Main cleaning logic
# -------------------------------------------------------
def main():
    log("Starting clean_training_data.py...")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    # Backup original CSV (once)
    if not os.path.exists(BACKUP_PATH):
        df_orig = pd.read_csv(CSV_PATH)
        df_orig.to_csv(BACKUP_PATH, index=False)
        log(f"Backup saved to {BACKUP_PATH}")
    else:
        log(f"Backup already exists at {BACKUP_PATH} (not overwriting).")

    # Load working copy
    df = pd.read_csv(CSV_PATH)
    log(f"Loaded {len(df)} rows from {CSV_PATH}")

    # Normalize markets to lower-case strings
    if MARKET_COL in df.columns:
        df[MARKET_COL] = df[MARKET_COL].astype(str).str.lower()
    else:
        log(f"No '{MARKET_COL}' column found; nothing to clean.")
        return

    # Fill missing player_id where we can
    df = fill_missing_player_ids(df)

    # Parse dates for filtering
    today = date.today()
    parsed_dates = df[DATE_COL].apply(parse_date) if DATE_COL in df.columns else pd.Series([None] * len(df))
    is_past_game = parsed_dates.apply(lambda d: d is not None and d < today)

    # ---------------------------------------------------
    # Build masks for rows to DROP
    # ---------------------------------------------------

    # 1) Bad markets (not in our allowed set) -> drop completely
    bad_market_mask = ~df[MARKET_COL].isin(ALLOWED_MARKETS)
    n_bad_market = bad_market_mask.sum()
    log(f"Rows with non-core markets to drop: {n_bad_market}")

    # 2) Past games with no player_id -> can never be labeled
    if PLAYER_ID_COL in df.columns:
        no_pid_past_mask = df[PLAYER_ID_COL].isna() & is_past_game
    else:
        no_pid_past_mask = pd.Series([False] * len(df))
    n_no_pid_past = no_pid_past_mask.sum()
    log(f"Past-game rows with no player_id to drop: {n_no_pid_past}")

    # 3) Rows with no line at all -> cannot compute label
    no_line_mask = df[LINE_COL].isna()
    n_no_line = no_line_mask.sum()
    log(f"Rows with missing line to drop: {n_no_line}")

    # Combined drop mask (OR of all above)
    drop_mask = bad_market_mask | no_pid_past_mask | no_line_mask
    n_drop_total = drop_mask.sum()
    log(f"Total rows to drop based on rules: {n_drop_total}")

    df_clean = df[~drop_mask].copy()
    log(f"Row count after dropping invalid rows: {len(df_clean)}")

    # ---------------------------------------------------
    # Drop duplicate props
    # ---------------------------------------------------
    key_cols = [PLAYER_ID_COL, MARKET_COL, LINE_COL, DATE_COL]
    key_cols = [c for c in key_cols if c in df_clean.columns]

    if key_cols:
        before_dupes = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=key_cols, keep="last")
        after_dupes = len(df_clean)
        log(f"Removed {before_dupes - after_dupes} duplicate props based on {key_cols}.")
    else:
        log("No key columns found for duplicate detection; skipped de-duplication.")

    # Save cleaned CSV back
    df_clean.to_csv(CSV_PATH, index=False)
    log(f"Saved cleaned data back to {CSV_PATH}")
    log("Done.")


if __name__ == "__main__":
    main()
