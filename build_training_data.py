"""
build_training_data.py

Builds a raw training dataset of game-by-game stats for a bunch of NBA players.

Output:
    data/training_data.csv

Each row is one game for one player with these columns:
    player_id
    player_name
    game_date
    points
    rebounds
    assists
    pra
    ra
    threes
    fantasy

You can later transform this into ML features (season_avg, last_n_avg, etc.)
in a separate script or inside train_model.py.
"""

import os
import time
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
OUT_DIR = "data"
OUT_FILE = os.path.join(OUT_DIR, "training_data.csv")

# Limit number of players so our call to nba_api doesn't take forever.
MAX_PLAYERS = 150  # you can increase later if you want


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def normalize_name(name: str) -> str:
    """Lowercase and strip punctuation/spaces for simple matching."""
    name = name.lower()
    for bad in [".", ",", "'", "`"]:
        name = name.replace(bad, "")
    return " ".join(name.split())


def fetch_gamelog(player_id: int) -> pd.DataFrame:
    """Fetch full career game log for a player."""
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=SeasonAll.all)
        df = gl.get_data_frames()[0].copy()
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
        df = df.sort_values("GAME_DATE", ascending=False)
        return df
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------
# MAIN DATA BUILD
# ---------------------------------------------------------------------
def build_training_dataset():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

    all_players = players.get_players()
    all_players = all_players[:MAX_PLAYERS]

    records = []

    print(f"Building training dataset for {len(all_players)} players...")
    print("---------------------------------------------------------")

    for i, p in enumerate(all_players):
        name = p["full_name"]
        pid = p["id"]

        print(f"[{i + 1}/{len(all_players)}] {name}")
        gl = fetch_gamelog(pid)
        if gl.empty:
            print("  → No game log; skipping.")
            continue

        # Ensure key stat columns exist and are numeric
        for col in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]:
            if col not in gl.columns:
                gl[col] = 0
        gl[["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]] = (
            gl[["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        for _, row in gl.iterrows():
            pts = float(row.get("PTS", 0) or 0)
            reb = float(row.get("REB", 0) or 0)
            ast = float(row.get("AST", 0) or 0)

            # If all zero, probably junk data; skip the row
            if pts == 0 and reb == 0 and ast == 0:
                continue

            fg3 = float(row.get("FG3M", 0) or 0)
            stl = float(row.get("STL", 0) or 0)
            blk = float(row.get("BLK", 0) or 0)
            tov = float(row.get("TOV", 0) or 0)

            fantasy = (
                pts
                + 1.2 * reb
                + 1.5 * ast
                + 3.0 * (stl + blk)
                - tov
            )

            records.append(
                {
                    "player_id": pid,
                    "player_name": name,
                    "game_date": row["GAME_DATE"],
                    "points": pts,
                    "rebounds": reb,
                    "assists": ast,
                    "pra": pts + reb + ast,
                    "ra": reb + ast,
                    "threes": fg3,
                    "fantasy": fantasy,
                }
            )

        # polite delay to avoid hammering nba_api
        time.sleep(0.5)

    if not records:
        print("No records were created – something went wrong.")
        return

    df = pd.DataFrame(records)
    df.to_csv(OUT_FILE, index=False)

    print("---------------------------------------------------------")
    print(f"Saved {len(df)} rows to {OUT_FILE}")
    print("---------------------------------------------------------")


if __name__ == "__main__":
    build_training_dataset()
