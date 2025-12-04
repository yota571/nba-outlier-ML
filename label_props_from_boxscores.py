"""
label_props_from_boxscores.py

Use NBA box scores (via nba_api) to label prop_training_data.csv with:

- actual_stat  (the real stat from the box score)
- label_over   (1 if actual_stat > line, else 0)

This script is standalone – it does NOT use Streamlit.

Run from your app folder, e.g.:

    cd C:\nba_outlier_app
    python label_props_from_boxscores.py
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll


TRAINING_DATA_FILE = "prop_training_data.csv"


# ----------------- helpers: players & logs ----------------- #

def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    for ch in [".", ",", "'", "`"]:
        name = name.replace(ch, "")
    name = " ".join(name.split())
    return name


def get_all_players():
    return players.get_players()


_ALL_PLAYERS = None
_PLAYER_ID_CACHE = {}
_PLAYER_LOG_CACHE = {}


def get_player_id(player_name: str):
    """Approximate name matching for nba_api player ID."""
    global _ALL_PLAYERS

    if not player_name:
        return None

    if player_name in _PLAYER_ID_CACHE:
        return _PLAYER_ID_CACHE[player_name]

    if _ALL_PLAYERS is None:
        _ALL_PLAYERS = get_all_players()

    target = normalize_name(player_name)

    # exact match
    for p in _ALL_PLAYERS:
        if normalize_name(p["full_name"]) == target:
            _PLAYER_ID_CACHE[player_name] = p["id"]
            return p["id"]

    # contains match
    for p in _ALL_PLAYERS:
        norm = normalize_name(p["full_name"])
        if target and (target in norm or norm in target):
            _PLAYER_ID_CACHE[player_name] = p["id"]
            return p["id"]

    # initials like "K. Caldwell-Pope"
    parts = target.split()
    if len(parts) > 1 and len(parts[0]) == 1:
        target_no_initial = " ".join(parts[1:])
        for p in _ALL_PLAYERS:
            norm = normalize_name(p["full_name"])
            if target_no_initial and (target_no_initial in norm or norm in target_no_initial):
                _PLAYER_ID_CACHE[player_name] = p["id"]
                return p["id"]

    # last-name-only unique match
    if len(parts) >= 2:
        last_name = parts[-1]
        candidates = []
        for p in _ALL_PLAYERS:
            norm = normalize_name(p["full_name"])
            if last_name and last_name in norm:
                candidates.append(p["id"])
        if len(candidates) == 1:
            _PLAYER_ID_CACHE[player_name] = candidates[0]
            return candidates[0]

    _PLAYER_ID_CACHE[player_name] = None
    return None


def get_player_gamelog(player_id):
    """Fetch and cache full game log for a player."""
    if player_id is None:
        return pd.DataFrame()

    if player_id in _PLAYER_LOG_CACHE:
        return _PLAYER_LOG_CACHE[player_id]

    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=SeasonAll.all)
        df = gl.get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
        df = df.sort_values("GAME_DATE", ascending=False)
        _PLAYER_LOG_CACHE[player_id] = df
        return df
    except Exception:
        return pd.DataFrame()


# ----------------- helpers: markets & stats ----------------- #

def get_market_series(gamelog_df: pd.DataFrame, market: str) -> pd.Series:
    """Map our prop 'market' to a column/combination in the box score row."""
    market = (market or "").lower().strip()
    if gamelog_df.empty:
        return pd.Series(dtype="float")

    for col in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]:
        if col not in gamelog_df.columns:
            gamelog_df[col] = 0.0

    if market == "points":
        return gamelog_df["PTS"]
    if market == "rebounds":
        return gamelog_df["REB"]
    if market == "assists":
        return gamelog_df["AST"]
    if market == "pra":
        return gamelog_df["PTS"] + gamelog_df["REB"] + gamelog_df["AST"]
    if market == "ra":
        return gamelog_df["REB"] + gamelog_df["AST"]
    if market == "threes":
        return gamelog_df["FG3M"]
    if market == "fs":
        # same fantasy scoring as the app
        return (
            gamelog_df["PTS"]
            + 1.2 * gamelog_df["REB"]
            + 1.5 * gamelog_df["AST"]
            + 3.0 * (gamelog_df["STL"] + gamelog_df["BLK"])
            - gamelog_df["TOV"]
        )

    return pd.Series(dtype="float")


# ----------------- labeling logic ----------------- #

def parse_game_date(row: pd.Series):
    """
    Try to get a date from game_date or game_time columns.
    game_date is usually already 'YYYY-MM-DD' as string.
    """
    # preferred: game_date column
    gd = row.get("game_date")
    if pd.notna(gd):
        try:
            return pd.to_datetime(gd).date()
        except Exception:
            pass

    # fallback: game_time
    gt = row.get("game_time")
    if pd.notna(gt):
        try:
            return pd.to_datetime(gt).date()
        except Exception:
            pass

    return None


def find_game_row(gamelog: pd.DataFrame, target_date):
    """
    Find the game row for target_date.
    - First try exact date match.
    - If none, try closest date within +/- 2 days.
    """
    if gamelog.empty or target_date is None:
        return pd.DataFrame()

    df = gamelog.copy()
    df["GAME_DATE_ONLY"] = df["GAME_DATE"].dt.date

    exact = df[df["GAME_DATE_ONLY"] == target_date]
    if not exact.empty:
        return exact

    # fallback: nearest date within +/- 2 days
    diffs = df["GAME_DATE_ONLY"].apply(lambda d: abs((d - target_date).days))
    min_idx = diffs.idxmin()
    if diffs[min_idx] <= 2:
        return df.loc[[min_idx]]

    return pd.DataFrame()


def label_row(row: pd.Series):
    """
    Return (actual_stat, label_over) for a single prop row.
    label_over = 1 if actual_stat > line, else 0.
    """
    player_name = row.get("player") or row.get("player_name")
    market = row.get("market")
    line_val = row.get("line")

    if pd.isna(player_name) or pd.isna(market) or pd.isna(line_val):
        return (None, None)

    try:
        line_float = float(line_val)
    except Exception:
        return (None, None)

    # player_id column if present, else use name lookup
    pid = row.get("player_id")
    if pd.isna(pid) or pid is None:
        pid = get_player_id(str(player_name))
    else:
        try:
            pid = int(pid)
        except Exception:
            pid = get_player_id(str(player_name))

    if pid is None:
        return (None, None)

    gamelog = get_player_gamelog(pid)
    if gamelog.empty:
        return (None, None)

    target_date = parse_game_date(row)
    game_row = find_game_row(gamelog, target_date)
    if game_row.empty:
        return (None, None)

    series = get_market_series(game_row, market).dropna()
    if series.empty:
        return (None, None)

    actual_stat = float(series.iloc[0])
    label = 1 if actual_stat > line_float else 0
    return (actual_stat, label)


def main():
    if not os.path.exists(TRAINING_DATA_FILE):
        print(f"Training data file not found: {TRAINING_DATA_FILE}")
        return

    df = pd.read_csv(TRAINING_DATA_FILE)
    print(f"Loaded {TRAINING_DATA_FILE} with {len(df)} rows")

    # ensure label_over & actual_stat columns exist
    if "label_over" not in df.columns:
        df["label_over"] = np.nan
    if "actual_stat" not in df.columns:
        df["actual_stat"] = np.nan

    # select rows needing labels
    mask_unlabeled = df["label_over"].isna()
    rows_to_label = df[mask_unlabeled]
    total_to_label = len(rows_to_label)

    if total_to_label == 0:
        print("No unlabeled rows found – nothing to do.")
        return

    print(f"Attempting to label {total_to_label} rows using box scores...\n")

    labeled_count = 0
    failed_count = 0

    for idx, row in rows_to_label.iterrows():
        actual_stat, label = label_row(row)

        if label is None or actual_stat is None:
            failed_count += 1
            continue

        df.at[idx, "actual_stat"] = actual_stat
        df.at[idx, "label_over"] = label
        labeled_count += 1

        if labeled_count % 25 == 0:
            print(f"Labeled {labeled_count} props so far...")

    print("\n=== Labeling complete ===")
    print(f"Labeled rows: {labeled_count}")
    print(f"Failed to label: {failed_count}")

    # save back to CSV (with a one-time backup)
    backup_name = TRAINING_DATA_FILE.replace(".csv", "_backup_before_label.csv")
    if not os.path.exists(backup_name):
        df.to_csv(backup_name, index=False)
        print(f"Backup written to: {backup_name}")

    df.to_csv(TRAINING_DATA_FILE, index=False)
    print(f"Updated training data saved to: {TRAINING_DATA_FILE}")


if __name__ == "__main__":
    main()
