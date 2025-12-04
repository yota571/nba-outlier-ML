"""
backfill_labels.py (v3)

Safely backfills `label_over` + `actual_stat` for completed games only.
- Skips NaT dates correctly
- Skips today and future dates
- Fills missing player_id from player name via nba_api
- Fetches NBA boxscores for past games only
- Writes results back into prop_training_data.csv
- Creates a backup first

Run:
    python backfill_labels.py
"""

import os
import time
from datetime import datetime, date

import pandas as pd

from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import players as nba_players


CSV_PATH = "prop_training_data.csv"
BACKUP_PATH = "prop_training_data_backup_before_backfill_v3.csv"

LABEL_COL = "label_over"
DATE_COL = "game_date_dt"
PLAYER_ID_COL = "player_id"
PLAYER_NAME_COL = "player"
MARKET_COL = "market"
LINE_COL = "line"
ACTUAL_COL = "actual_stat"

PLAYER_REQUEST_SLEEP = 0.60


# -------------------------------------------------------------
# Logging
# -------------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# -------------------------------------------------------------
# File I/O
# -------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found at: {path}")
    df = pd.read_csv(path)
    log(f"Loaded {len(df)} rows from {path}")
    return df


def backup_csv(path: str, backup_path: str) -> None:
    if not os.path.exists(backup_path):
        df = pd.read_csv(path)
        df.to_csv(backup_path, index=False)
        log(f"Backup created at: {backup_path}")
    else:
        log(f"Backup already exists: {backup_path}")


# -------------------------------------------------------------
# Date & Player ID helpers
# -------------------------------------------------------------
def parse_date(x):
    """Safely convert string → datetime.date or return None."""
    if pd.isna(x):
        return None
    try:
        d = pd.to_datetime(str(x), errors="coerce")
        if pd.isna(d):
            return None
        # Convert Timestamp → date safely
        return d.date()
    except Exception:
        return None


def build_player_name_index():
    """Build dict: 'stephen curry' → player_id"""
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
    """Fill rows missing player_id based on player name."""
    if PLAYER_NAME_COL not in df.columns:
        log("No 'player' column found; cannot fill player_id.")
        return df

    missing = df[PLAYER_ID_COL].isna() & df[PLAYER_NAME_COL].notna()
    total_missing = missing.sum()
    log(f"Missing player_id with name available: {total_missing}")

    if total_missing == 0:
        return df

    name_index = build_player_name_index()
    filled = 0

    for idx in df[missing].index:
        name = str(df.at[idx, PLAYER_NAME_COL]).strip().lower()
        if not name:
            continue
        pid = name_index.get(name)
        if pid:
            df.at[idx, PLAYER_ID_COL] = pid
            filled += 1

    log(f"Filled {filled} missing player_id using player names.")
    return df


# -------------------------------------------------------------
# NBA Stats / Backfill logic
# -------------------------------------------------------------
def build_player_gamelog_cache(df: pd.DataFrame):
    """Fetch gamelogs only for players needing backfill for past games."""
    today = date.today()

    mask = (
        df[LABEL_COL].isna()
        & df[PLAYER_ID_COL].notna()
        & df[DATE_COL].notna()
    )

    def is_past_game(x):
        d = parse_date(x)
        return d is not None and d < today

    mask &= df[DATE_COL].apply(is_past_game)

    subset = df[mask]
    player_ids = sorted(subset[PLAYER_ID_COL].dropna().unique())
    log(f"Fetching gamelogs for {len(player_ids)} players...")

    cache = {}

    for i, pid in enumerate(player_ids, start=1):
        try:
            pid_int = int(pid)
        except ValueError:
            log(f"Skipping invalid player_id={pid}")
            continue

        log(f"[{i}/{len(player_ids)}] Fetching logs for player_id={pid_int}...")
        try:
            gl = PlayerGameLog(player_id=pid_int, season=SeasonAll.all)
            gl_df = gl.get_data_frames()[0].copy()
            gl_df["GAME_DATE"] = pd.to_datetime(gl_df["GAME_DATE"]).dt.date
            cache[pid_int] = gl_df
            log(f"  -> Retrieved {len(gl_df)} games.")
        except Exception as e:
            log(f"  !! Failed for player_id={pid_int}: {e}")
            continue

        time.sleep(PLAYER_REQUEST_SLEEP)

    return cache


def compute_stat(log_row, market: str):
    pts = log_row.get("PTS", 0)
    reb = log_row.get("REB", 0)
    ast = log_row.get("AST", 0)
    stl = log_row.get("STL", 0)
    blk = log_row.get("BLK", 0)
    tov = log_row.get("TOV", 0)

    m = str(market).lower()

    if m == "points":
        return float(pts)
    if m == "rebounds":
        return float(reb)
    if m == "assists":
        return float(ast)
    if m == "pra":
        return float(pts + reb + ast)
    if m == "ra":
        return float(reb + ast)
    if m == "fs":
        return float(
            pts * 1.0
            + reb * 1.2
            + ast * 1.5
            + stl * 3.0
            + blk * 3.0
            - tov * 1.0
        )
    return None


def backfill_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Main backfill logic."""
    today = date.today()

    # Normalize date column to string
    df[DATE_COL] = df[DATE_COL].astype(str)

    # Fill missing player IDs
    df = fill_missing_player_ids(df)

    # Identify rows we want to backfill
    def is_valid_candidate(x):
        d = parse_date(x)
        return d is not None and d < today

    mask = (
        df[LABEL_COL].isna()
        & df[PLAYER_ID_COL].notna()
        & df[DATE_COL].apply(is_valid_candidate)
    )

    total_unlabeled = df[LABEL_COL].isna().sum()
    eligible = mask.sum()

    log(f"Total unlabeled rows: {total_unlabeled}")
    log(f"Eligible backfill rows (past games): {eligible}")

    if eligible == 0:
        log("Nothing to backfill.")
        return df

    # Fetch game logs
    cache = build_player_gamelog_cache(df)

    updated = 0
    skipped_no_cache = 0
    skipped_no_game = 0
    skipped_market = 0

    for idx in df[mask].index:
        row = df.loc[idx]

        pid = row[PLAYER_ID_COL]
        try:
            pid_int = int(pid)
        except Exception:
            continue

        gl_df = cache.get(pid_int)
        if gl_df is None:
            skipped_no_cache += 1
            continue

        game_date = parse_date(row[DATE_COL])
        game = gl_df[gl_df["GAME_DATE"] == game_date]

        if game.empty:
            skipped_no_game += 1
            continue

        game_row = game.iloc[0]

        actual = compute_stat(game_row, row[MARKET_COL])
        if actual is None:
            skipped_market += 1
            continue

        df.at[idx, ACTUAL_COL] = actual

        try:
            line = float(row[LINE_COL])
            label = 1 if actual > line else 0
            df.at[idx, LABEL_COL] = label
            updated += 1
        except Exception:
            continue

        if updated % 100 == 0:
            log(f"Updated {updated} rows...")

    log(f"Backfill complete:")
    log(f"  Rows updated: {updated}")
    log(f"  Skipped (no cache): {skipped_no_cache}")
    log(f"  Skipped (no game): {skipped_no_game}")
    log(f"  Skipped (unknown market): {skipped_market}")

    return df


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
def main():
    log("Starting backfill_labels_v3.py...")

    backup_csv(CSV_PATH, BACKUP_PATH)

    df = load_data(CSV_PATH)

    df = backfill_labels(df)

    df.to_csv(CSV_PATH, index=False)
    log(f"Saved updated CSV to {CSV_PATH}")
    log("Done.")


if __name__ == "__main__":
    main()
