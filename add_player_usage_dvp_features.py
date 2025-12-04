"""
add_player_usage_dvp_features.py (FINAL)

Adds player- and opponent-based features to prop_training_data.csv:

Per-player (season-level, from NBA advanced stats):
    usage_season         (USG_PCT per game)
    minutes_season       (MIN per game)
    expected_minutes     (same as minutes_season for now)

Defense vs Prop Type (DVP built from YOUR labeled game data):
    opp_pos_dvp_delta    (for each [opponent, market],
                          avg actual_stat vs league avg for that market)

IMPORTANT: This version does NOT use any position column,
so it will not throw "could not find a position column" errors.

Run:
    python add_player_usage_dvp_features.py
"""

import os
from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats


CSV_PATH = "prop_training_data.csv"
BACKUP_PATH = "prop_training_data_backup_before_usage_dvp.csv"

PLAYER_ID_COL = "player_id"
MARKET_COL = "market"
OPP_COL = "opponent"
DATE_COL = "game_date_dt"
ACTUAL_COL = "actual_stat"
LABEL_COL = "label_over"


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
# NBA API helper
# -------------------------------------------------------
def fetch_player_advanced(season: str) -> pd.DataFrame:
    """
    Pull advanced per-game player stats for a season.
    We only need PLAYER_ID, USG_PCT, MIN.
    """
    log(f"Fetching player advanced stats for season={season}...")
    stats = LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="PerGame",
    )
    df = stats.get_data_frames()[0].copy()

    log(f"Player stats columns: {list(df.columns)}")  # debug

    # Try to find columns in a slightly flexible way
    cols_lower = {c.lower(): c for c in df.columns}

    def pick(col_name: str, alt_contains: str | None = None):
        if col_name in df.columns:
            return col_name
        key = col_name.lower()
        if key in cols_lower:
            return cols_lower[key]
        if alt_contains:
            for c in df.columns:
                if alt_contains.lower() in c.lower():
                    return c
        return None

    col_player_id = pick("PLAYER_ID", "player_id")
    col_usg = pick("USG_PCT", "usg")
    col_min = pick("MIN", "min")

    missing = []
    if col_player_id is None:
        missing.append("PLAYER_ID")
    if col_usg is None:
        missing.append("USG_PCT")
    if col_min is None:
        missing.append("MIN")

    if missing:
        raise RuntimeError(f"Missing expected columns in player stats: {missing}")

    df = df[[col_player_id, col_usg, col_min]].copy()
    df.columns = ["PLAYER_ID", "USG_PCT", "MIN"]
    return df


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    log("Starting add_player_usage_dvp_features.py (FINAL)...")

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

    # Basic checks
    for col in [PLAYER_ID_COL, MARKET_COL, OPP_COL, DATE_COL]:
        if col not in df.columns:
            log(f"ERROR: Missing '{col}' column in CSV. Exiting.")
            return

    # Figure out the active season(s)
    parsed_dates = df[DATE_COL].apply(parse_date)
    seasons = sorted(
        {season_for_date(d) for d in parsed_dates.dropna() if season_for_date(d)}
    )
    log(f"Seasons detected from game_date_dt: {seasons}")
    if not seasons:
        log("No valid game dates, aborting.")
        return

    season = seasons[-1]
    log(f"Using season={season} for player advanced stats.")

    # Make sure player_id is numeric
    df[PLAYER_ID_COL] = pd.to_numeric(df[PLAYER_ID_COL], errors="coerce")

    # Fetch NBA advanced stats and build player map
    try:
        adv_df = fetch_player_advanced(season)
    except Exception as e:
        log(f"Failed to fetch player advanced stats: {e}")
        return

    adv_df["PLAYER_ID"] = pd.to_numeric(adv_df["PLAYER_ID"], errors="coerce")
    adv_df = adv_df.dropna(subset=["PLAYER_ID"])
    adv_df["PLAYER_ID"] = adv_df["PLAYER_ID"].astype(int)

    player_map = {}
    for _, row in adv_df.iterrows():
        pid = int(row["PLAYER_ID"])
        usage = float(row["USG_PCT"])
        mins = float(row["MIN"])
        player_map[pid] = {
            "usage_season": usage,
            "minutes_season": mins,
            "expected_minutes": mins,
        }

    log(f"Built advanced-stats map for {len(player_map)} players.")

    # Ensure columns exist
    for c in ["usage_season", "minutes_season", "expected_minutes", "opp_pos_dvp_delta"]:
        if c not in df.columns:
            df[c] = pd.NA

    # Attach usage/minutes to rows
    attached = 0
    for idx in df.index:
        pid_val = df.at[idx, PLAYER_ID_COL]
        try:
            pid = int(pid_val)
        except Exception:
            continue
        info = player_map.get(pid)
        if not info:
            continue
        df.at[idx, "usage_season"] = info["usage_season"]
        df.at[idx, "minutes_season"] = info["minutes_season"]
        df.at[idx, "expected_minutes"] = info["expected_minutes"]
        attached += 1

    log(f"Attached usage/minutes to {attached} rows.")

    # ---------------------------------------------------
    # DVP by opponent + market using labeled rows
    # ---------------------------------------------------
    if ACTUAL_COL not in df.columns or LABEL_COL not in df.columns:
        log("No actual_stat / label_over columns; skipping DVP computation.")
    else:
        mask = (
            df[ACTUAL_COL].notna()
            & df[LABEL_COL].notna()
            & df[OPP_COL].notna()
            & df[MARKET_COL].notna()
        )
        dvp_base = df[mask].copy()
        log(f"DVP: using {len(dvp_base)} labeled rows with known opponent/market.")

        if len(dvp_base) > 0:
            # Opponent+market mean actual
            opp_market = (
                dvp_base.groupby([OPP_COL, MARKET_COL])[ACTUAL_COL]
                .mean()
                .rename("opp_market_mean")
                .reset_index()
            )

            # League-wide mean per market
            league_market = (
                dvp_base.groupby([MARKET_COL])[ACTUAL_COL]
                .mean()
                .rename("league_market_mean")
                .reset_index()
            )

            dvp_merged = opp_market.merge(league_market, on=[MARKET_COL], how="left")
            dvp_merged["opp_pos_dvp_delta"] = (
                dvp_merged["opp_market_mean"] - dvp_merged["league_market_mean"]
            )

            dvp_key = dvp_merged[[OPP_COL, MARKET_COL, "opp_pos_dvp_delta"]].copy()

            df = df.merge(
                dvp_key,
                on=[OPP_COL, MARKET_COL],
                how="left",
                suffixes=("", "_dvpmerge"),
            )
            if "opp_pos_dvp_delta_dvpmerge" in df.columns:
                df["opp_pos_dvp_delta"] = df["opp_pos_dvp_delta_dvpmerge"]
                df.drop(columns=["opp_pos_dvp_delta_dvpmerge"], inplace=True)

            log("DVP features (opponent+market) merged back onto main DataFrame.")
        else:
            log("Not enough labeled rows for DVP; leaving opp_pos_dvp_delta as NaN.")

    # Save CSV
    df.to_csv(CSV_PATH, index=False)
    log(f"Saved updated CSV with usage + DVP features to {CSV_PATH}")
    log("Done.")


if __name__ == "__main__":
    main()
