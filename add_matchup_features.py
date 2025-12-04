"""
add_matchup_features.py  (v3)

Adds advanced matchup features to prop_training_data.csv:

Season-level:
    team_pace_season
    opp_pace_season
    game_pace_season_avg
    team_def_rtg_season
    opp_def_rtg_season
    opp_def_rtg_minus_league_avg

Last-10 games snapshot:
    team_pace_last10
    opp_pace_last10
    team_def_rtg_last10
    opp_def_rtg_last10

Uses TEAM_ID (not TEAM_ABBREVIATION) to avoid column issues.

Run:
    python add_matchup_features.py
"""

import os
from datetime import datetime

import pandas as pd

from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import LeagueDashTeamStats


CSV_PATH = "prop_training_data.csv"
BACKUP_PATH = "prop_training_data_backup_before_matchup_features.csv"

TEAM_COL = "team"
OPP_COL = "opponent"
DATE_COL = "game_date_dt"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# -------------------------------------------------------
# Helpers
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
    2026-03-01 -> '2025-26'
    2025-04-01 -> '2024-25'
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


def fetch_team_advanced(season: str, last10: bool = False) -> pd.DataFrame:
    """
    Fetch advanced team stats for given season.
    If last10=True, uses last_n_games=10 snapshot.
    NOTE: We only rely on TEAM_ID, PACE, DEF_RATING.
    """
    log(f"Fetching team advanced stats for season={season}, last10={last10}...")
    kwargs = {
        "season": season,
        # ✅ correct parameter name for your nba_api version:
        "measure_type_detailed_defense": "Advanced",
        "per_mode_detailed": "PerGame",
    }
    if last10:
        kwargs["last_n_games"] = "10"

    stats = LeagueDashTeamStats(**kwargs)
    df = stats.get_data_frames()[0]
    cols = ["TEAM_ID", "PACE", "DEF_RATING"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in team stats: {missing}")
    return df[cols].copy()


def build_team_lookup():
    """Map team abbreviation -> TEAM_ID using nba_api static teams."""
    teams = nba_teams.get_teams()
    abbr_to_id = {}
    for t in teams:
        abbr = t.get("abbreviation")
        tid = t.get("id")
        if abbr and tid:
            abbr_to_id[abbr.upper()] = tid
    log(f"Built static team map for {len(abbr_to_id)} teams.")
    return abbr_to_id


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    log("Starting add_matchup_features.py v3...")

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

    for col in [TEAM_COL, OPP_COL, DATE_COL]:
        if col not in df.columns:
            log(f"ERROR: Missing '{col}' column in CSV. Exiting.")
            return

    # Normalize team abbreviations
    df[TEAM_COL] = df[TEAM_COL].astype(str).str.upper()
    df[OPP_COL] = df[OPP_COL].astype(str).str.upper()

    # Determine seasons in data
    parsed_dates = df[DATE_COL].apply(parse_date)
    seasons = set()
    for d in parsed_dates.dropna():
        s = season_for_date(d)
        if s:
            seasons.add(s)
    seasons = sorted(seasons)
    log(f"Seasons detected in data: {seasons}")
    if not seasons:
        log("No valid game dates; nothing to do.")
        return

    # Build abbr -> TEAM_ID
    abbr_to_id = build_team_lookup()

    # Fetch stats per season
    season_stats = {}
    season_stats_last10 = {}
    for s in seasons:
        try:
            base = fetch_team_advanced(s, last10=False)
            last10_df = fetch_team_advanced(s, last10=True)
        except Exception as e:
            log(f"Failed to fetch advanced stats for season={s}: {e}")
            continue

        def build_map(df_stats):
            m = {}
            for _, row in df_stats.iterrows():
                tid = int(row["TEAM_ID"])
                m[tid] = {
                    "PACE": float(row["PACE"]),
                    "DEF_RATING": float(row["DEF_RATING"]),
                }
            return m

        season_stats[s] = build_map(base)
        season_stats_last10[s] = build_map(last10_df)

    # League average DEF_RATING per season
    league_def_avg = {}
    for s, m in season_stats.items():
        if not m:
            continue
        vals = [v["DEF_RATING"] for v in m.values()]
        league_def_avg[s] = sum(vals) / len(vals)
    log(f"League DEF_RATING averages: {league_def_avg}")

    # Prepare new columns
    new_cols = [
        "team_pace_season",
        "opp_pace_season",
        "game_pace_season_avg",
        "team_def_rtg_season",
        "opp_def_rtg_season",
        "opp_def_rtg_minus_league_avg",
        "team_pace_last10",
        "opp_pace_last10",
        "team_def_rtg_last10",
        "opp_def_rtg_last10",
    ]
    for c in new_cols:
        if c not in df.columns:
            df[c] = pd.NA

    updated = 0

    for idx in df.index:
        d = parse_date(df.at[idx, DATE_COL])
        if d is None:
            continue
        season = season_for_date(d)
        if season not in season_stats:
            continue

        team_abbr = str(df.at[idx, TEAM_COL]).upper()
        opp_abbr = str(df.at[idx, OPP_COL]).upper()

        team_id = abbr_to_id.get(team_abbr)
        opp_id = abbr_to_id.get(opp_abbr)
        if team_id is None and opp_id is None:
            continue

        season_map = season_stats.get(season, {})
        season_map_l10 = season_stats_last10.get(season, {})
        league_avg = league_def_avg.get(season)

        team_season = season_map.get(team_id)
        opp_season = season_map.get(opp_id)
        team_l10 = season_map_l10.get(team_id)
        opp_l10 = season_map_l10.get(opp_id)

        # Season-level pace / def
        if team_season:
            df.at[idx, "team_pace_season"] = team_season["PACE"]
            df.at[idx, "team_def_rtg_season"] = team_season["DEF_RATING"]
        if opp_season:
            df.at[idx, "opp_pace_season"] = opp_season["PACE"]
            df.at[idx, "opp_def_rtg_season"] = opp_season["DEF_RATING"]
            if league_avg is not None:
                df.at[idx, "opp_def_rtg_minus_league_avg"] = (
                    opp_season["DEF_RATING"] - league_avg
                )

        # Game pace avg
        tp = df.at[idx, "team_pace_season"]
        op = df.at[idx, "opp_pace_season"]
        try:
            if pd.notna(tp) and pd.notna(op):
                df.at[idx, "game_pace_season_avg"] = (float(tp) + float(op)) / 2.0
        except Exception:
            pass

        # Last-10 snapshot
        if team_l10:
            df.at[idx, "team_pace_last10"] = team_l10["PACE"]
            df.at[idx, "team_def_rtg_last10"] = team_l10["DEF_RATING"]
        if opp_l10:
            df.at[idx, "opp_pace_last10"] = opp_l10["PACE"]
            df.at[idx, "opp_def_rtg_last10"] = opp_l10["DEF_RATING"]

        updated += 1

    log(f"Matchup features attempted on {len(df)} rows, updated={updated}")
    df.to_csv(CSV_PATH, index=False)
    log(f"Saved updated CSV with matchup features to {CSV_PATH}")
    log("Done.")


if __name__ == "__main__":
    main()
