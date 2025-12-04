"""
add_game_env_features.py

Builds game-environment features from existing matchup features and
adds them to prop_training_data.csv:

Required existing columns (from add_matchup_features.py):
    - game_pace_season_avg
    - opp_def_rtg_minus_league_avg
    - game_date_dt

New columns added:
    - env_pace_delta      (how fast this game is vs league avg for that season)
    - env_def_softness    (how soft/tough opponent defense is; higher = softer)
    - game_env_score      (combined environment score; higher = better for overs)

Run:
    python add_game_env_features.py
"""

import os
from datetime import datetime

import pandas as pd


CSV_PATH = "prop_training_data.csv"
BACKUP_PATH = "prop_training_data_backup_before_game_env.csv"

DATE_COL = "game_date_dt"
PACE_COL = "game_pace_season_avg"
OPP_DEF_DELTA_COL = "opp_def_rtg_minus_league_avg"


# -------------------------------------------------------
# Logging
# -------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# -------------------------------------------------------
# Date helpers
# -------------------------------------------------------
def parse_date(s):
    if pd.isna(s):
        return None
    try:
        d = pd.to_datetime(str(s), errors="coerce")
        if pd.isna(d):
            return None
        return d.date()
    except Exception:
        return None


def season_for_date(d):
    """
    Convert a game date to NBA season string, e.g.:
        2025-11-15 -> '2025-26'
    """
    if d is None:
        return None
    year = d.year
    if d.month >= 8:
        start_year = year
    else:
        start_year = year - 1
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    log("Starting add_game_env_features.py...")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    # Backup once
    if not os.path.exists(BACKUP_PATH):
        df_backup = pd.read_csv(CSV_PATH)
        df_backup.to_csv(BACKUP_PATH, index=False)
        log(f"Backup saved to {BACKUP_PATH}")
    else:
        log(f"Backup already exists at {BACKUP_PATH} (not overwriting).")

    df = pd.read_csv(CSV_PATH)
    log(f"Loaded {len(df)} rows from {CSV_PATH}")

    # Check required columns
    for col in [DATE_COL, PACE_COL, OPP_DEF_DELTA_COL]:
        if col not in df.columns:
            log(f"ERROR: Missing required column '{col}'. "
                f"Make sure add_matchup_features.py has been run.")
            return

    # Build season key
    parsed_dates = df[DATE_COL].apply(parse_date)
    season_keys = []
    for d in parsed_dates:
        season_keys.append(season_for_date(d) if d is not None else None)
    df["season_key"] = season_keys

    valid_seasons = sorted({s for s in df["season_key"].dropna().unique()})
    log(f"Seasons detected in data: {valid_seasons}")
    if not valid_seasons:
        log("No valid seasons found; nothing to do.")
        return

    # Ensure numeric
    df[PACE_COL] = pd.to_numeric(df[PACE_COL], errors="coerce")
    df[OPP_DEF_DELTA_COL] = pd.to_numeric(df[OPP_DEF_DELTA_COL], errors="coerce")

    # Compute league average pace per season
    season_pace_means = (
        df.dropna(subset=[PACE_COL, "season_key"])
        .groupby("season_key")[PACE_COL]
        .mean()
        .to_dict()
    )
    log(f"League average game pace per season: {season_pace_means}")

    # Prepare new columns
    for c in ["env_pace_delta", "env_def_softness", "game_env_score"]:
        if c not in df.columns:
            df[c] = pd.NA

    updated = 0
    for idx in df.index:
        season = df.at[idx, "season_key"]
        if season is None or season not in season_pace_means:
            continue

        pace_val = df.at[idx, PACE_COL]
        opp_def_delta = df.at[idx, OPP_DEF_DELTA_COL]

        # Need at least pace and opponent defense delta
        if pd.isna(pace_val) or pd.isna(opp_def_delta):
            continue

        pace_val = float(pace_val)
        opp_def_delta = float(opp_def_delta)

        league_pace = float(season_pace_means[season])

        # How fast this game is vs league average
        env_pace_delta = pace_val - league_pace

        # Defense softness: positive = softer (gives up more)
        # opp_def_rtg_minus_league_avg is (opp_def - league_def_avg),
        # so negative values = soft; we flip sign so higher = softer.
        env_def_softness = -opp_def_delta

        # Combined environment score (bigger = better for overs)
        game_env_score = env_pace_delta + env_def_softness

        df.at[idx, "env_pace_delta"] = env_pace_delta
        df.at[idx, "env_def_softness"] = env_def_softness
        df.at[idx, "game_env_score"] = game_env_score

        updated += 1

    log(f"Game environment features computed for {updated} rows.")

    # Save back
    df.drop(columns=["season_key"], inplace=True)
    df.to_csv(CSV_PATH, index=False)
    log(f"Saved updated CSV with game env features to {CSV_PATH}")
    log("Done.")


if __name__ == "__main__":
    main()
