# =====================================================
# NBA Prop Edge Finder – PrizePicks
# Version 2.1.0 (with ML integration + What-If Alt Lines tab + chart)
# =====================================================

import math
import os
import time
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st
import joblib  # ML model loader
import pandas as pd

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll

# =====================================================
# OPTIONAL DFS_WRAPPER / PRIZEPICK FALLBACK
# =====================================================
try:
    # If DFS_Wrapper.py exists (e.g., in a local environment), use it.
    from DFS_Wrapper import PrizePick  # type: ignore
except Exception:
    # Fallback stub so the app still runs even if DFS_Wrapper is missing
    import pandas as _pp_pd

    class PrizePick:  # type: ignore
        """
        Minimal fallback PrizePick wrapper used when DFS_Wrapper.py
        is not available (e.g., on Streamlit Cloud).

        This stub DOES NOT call the real PrizePicks API. It simply
        returns empty DataFrames so the rest of the app can still
        function using manual CSV uploads.
        """

        def __init__(self, *args, **kwargs):
            pass

        def get_data(self, organize_data: bool | None = None):
            """Return an empty list to indicate no live props."""
            return []

        # Backwards-compat alias in case older code expects a DataFrame
        def to_dataframe(self):
            cols = [
                "player_name",
                "team",
                "opponent",
                "market",
                "line",
                "game_time",
                "book",
                "odds_type",
            ]
            return _pp_pd.DataFrame(columns=cols)



# =====================================================
# BET PERSISTENCE HELPERS
# =====================================================

BETS_FILE = "bet_tracker.csv"
ML_MODEL_PATH = "over_model.pkl"  # <-- your trained ML model file
PER_MARKET_MODEL_PATHS = {
    "points": "over_model_points.pkl",
    "rebounds": "over_model_rebounds.pkl",
    "assists": "over_model_assists.pkl",
    "pra": "over_model_pra.pkl",
    "ra": "over_model_ra.pkl",
}

TRAINING_DATA_FILE = "prop_training_data.csv"


def load_bets_from_disk() -> list:
    """Load saved bets from CSV if it exists."""
    if not os.path.exists(BETS_FILE):
        return []
    try:
        df = pd.read_csv(BETS_FILE)
        return df.to_dict(orient="records")
    except Exception:
        return []


def save_bets_to_disk(bets: list) -> None:
    """Persist bets to CSV."""
    try:
        df = pd.DataFrame(bets)
        df.to_csv(BETS_FILE, index=False)
    except Exception:
        pass


def append_edges_to_training_dataset(edges_df: pd.DataFrame) -> None:
    """
    Append the current edges dataframe to a long-term training CSV so an
    external script can train an ML model on historical props.

    Uses a stable key of (player, team, opponent, market, line, game_date)
    to avoid duplicate rows. Label ("label_over") is left empty here and
    can be filled by an offline grading script.
    """
    if edges_df is None or edges_df.empty:
        return

    df = edges_df.copy()

    # simple game_date from game_time
    if "game_time" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_time"], errors="coerce").dt.date
    else:
        df["game_date"] = pd.NaT

    # figure out main lookback columns (avg_last_X, over_rate_last_X, edge_last_X)
    avg_cols = [c for c in df.columns if c.startswith("avg_last_") and "7" not in c]
    rate_cols = [c for c in df.columns if c.startswith("over_rate_last_") and "7" not in c]
    edge_cols = [c for c in df.columns if c.startswith("edge_last_") and "7" not in c]

    main_avg_col = avg_cols[0] if avg_cols else None
    main_rate_col = rate_cols[0] if rate_cols else None
    main_edge_col = edge_cols[0] if edge_cols else None

    col_map = {
        "player": "player",
        "player_id": "player_id",
        "team": "team",
        "opponent": "opponent",
        "market": "market",
        "line": "line",
        "book": "book",
        "season_avg": "season_avg",
        "avg_last_7": "avg_last_7",
        "over_rate_last_7": "over_rate_last_7",
        "edge_last_7": "edge_last_7",
        "predicted_score": "predicted_score",
        "confidence_pct": "confidence_pct",
        "game_time": "game_time",
        "game_date": "game_date",
        "ml_prob_over": "ml_prob_over",
    }
    if main_avg_col and main_avg_col in df.columns:
        col_map[main_avg_col] = "avg_last_main"
    if main_rate_col and main_rate_col in df.columns:
        col_map[main_rate_col] = "over_rate_main"
    if main_edge_col and main_edge_col in df.columns:
        col_map[main_edge_col] = "edge_last_main"

    keep_cols = [src for src in col_map.keys() if src in df.columns]
    if not keep_cols:
        return

    out = df[keep_cols].rename(columns=col_map)

    # stable key
    out["game_date"] = out["game_date"].astype(str)
    out["line"] = pd.to_numeric(out["line"], errors="coerce")
    out["key"] = (
        out["player"].astype(str)
        + "|" + out["team"].astype(str)
        + "|" + out["opponent"].astype(str)
        + "|" + out["market"].astype(str)
        + "|" + out["line"].astype(str)
        + "|" + out["game_date"].astype(str)
    )

    if "label_over" not in out.columns:
        out["label_over"] = pd.NA

    if os.path.exists(TRAINING_DATA_FILE):
        try:
            existing = pd.read_csv(TRAINING_DATA_FILE)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    if not existing.empty:
        if "key" not in existing.columns:
            existing["game_date"] = existing.get("game_date", pd.NaT).astype(str)
            existing["line"] = pd.to_numeric(existing.get("line", None), errors="coerce")
            existing["key"] = (
                existing.get("player", "").astype(str)
                + "|" + existing.get("team", "").astype(str)
                + "|" + existing.get("opponent", "").astype(str)
                + "|" + existing.get("market", "").astype(str)
                + "|" + existing.get("line", "").astype(str)
                + "|" + existing.get("game_date", "").astype(str)
            )
        existing_keys = set(existing["key"].astype(str))
    else:
        existing_keys = set()

    new_rows = out[~out["key"].astype(str).isin(existing_keys)]
    if new_rows.empty:
        return

    combined = pd.concat([existing, new_rows], ignore_index=True)
    try:
        combined.to_csv(TRAINING_DATA_FILE, index=False)
    except Exception:
        # don't break app if write fails
        pass


ML_MODEL_PATH = "over_model.pkl"  # <-- keep this

@st.cache_resource
def load_over_model():
    """Load the ML model from disk if available, plus file timestamp."""
    try:
        # when the model file was last saved (approx training time)
        mtime = os.path.getmtime(ML_MODEL_PATH)
        trained_at = datetime.fromtimestamp(mtime)

        model = joblib.load(ML_MODEL_PATH)
        return model, trained_at
    except Exception:
        return None, None


over_model, over_model_trained_at = load_over_model()
def load_market_models():
    """Load per-market ML models if their .pkl files are present.

    Falls back gracefully if a file is missing or fails to load.
    Keys are lowercased market names: e.g. "points", "rebounds".
    """
    models = {}
    for mkt, path in PER_MARKET_MODEL_PATHS.items():
        try:
            if os.path.exists(path):
                models[mkt] = joblib.load(path)
        except Exception:
            # If a specific market model fails, we simply skip it
            continue
    return models


over_models_by_market = load_market_models()




# =====================================================
# STREAMLIT PAGE CONFIG & STATE
# =====================================================
st.set_page_config(
    page_title="NBA Outlier – NBA Prop Edge Finder (PrizePicks)",
    layout="wide",
)

# persistent bet tracker backed by CSV
if "bet_tracker" not in st.session_state:
    st.session_state["bet_tracker"] = load_bets_from_disk()


# =====================================================
# HELPERS: FILTER FULL-GAME PROPS ONLY
# =====================================================
def _is_full_game_prop_pp(item: dict, stat_type: str | None) -> bool:
    """
    PrizePicks: detect and exclude 1H / 2H / quarter / first 5-min props.
    We only want full-game props so we can use full-game stats.
    """
    text_bits = []

    for key in [
        "description",
        "short_description",
        "label",
        "title",
        "stat_type",
        "market_type",
        "game_type",
        "league",
    ]:
        v = item.get(key)
        if v:
            text_bits.append(str(v).lower())

    joined = " ".join(text_bits)

    bad_keywords = [
        "1h", " 1h ", "2h",
        "first half", "1st half", "second half", "2nd half",
        "half points", "half pts", "1h points", "1h pts",
        "1st quarter", "2nd quarter", "3rd quarter", "4th quarter",
        "first quarter", "second quarter", "third quarter", "fourth quarter",
        "q1", "q2", "q3", "q4",
        "first 5", "in first 5", "first five",
        "first 3 min", "first 6 min", "first 7 min",
        "in first six", "in first seven",
    ]

    for kw in bad_keywords:
        if kw in joined:
            return False

    period = str(item.get("period", "")).lower()
    if period and period not in ("", "full", "full game", "game"):
        return False

    scope = str(item.get("scope", "")).lower()
    if any(x in scope for x in ["half", "quarter", "1h", "2h", "q1", "q2", "q3", "q4"]):
        return False

    return True


# =====================================================
# PRIZEPICKS LOADER (NBA, FULL GAME ONLY)
# =====================================================
@st.cache_data
def collect_full_ladders_to_training_from_prizepicks() -> int:
    """Pull ALL full-game PrizePicks NBA lines (board + alts) and
    append them into the training CSV via append_edges_to_training_dataset.

    Returns the number of raw rows considered (before de-dup in training file).
    """
    try:
        pp = PrizePick()
        raw = pp.get_data(organize_data=False)
    except TypeError:
        pp = PrizePick()
        raw = pp.get_data(False)
    except Exception:
        # If PrizePicks is down or DFS_Wrapper fails, just skip
        return 0

    if not isinstance(raw, list):
        return 0

    records: list[dict] = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        league = item.get("league", "")
        if "NBA" not in str(league).upper():
            continue

        # status filter – skip clearly closed/settled/void
        status = str(item.get("status", "")).lower()
        bad_statuses = [
            "settled",
            "graded",
            "closed",
            "void",
            "canceled",
            "cancelled",
            "refunded",
            "suspended",
        ]
        if any(bad in status for bad in bad_statuses):
            continue

        stat_type = item.get("stat_type")
        if not _is_full_game_prop_pp(item, stat_type):
            continue

        # raw line value
        line_val = item.get("line_score")
        if line_val is None:
            line_val = item.get("line_value")
        if line_val is None:
            continue

        try:
            line_val = float(line_val)
        except Exception:
            continue

        player_name = item.get("player_name") or item.get("name")
        if not player_name:
            continue

        # we use 'player' here to match the training schema
        player = player_name

        team = item.get("team")
        opponent = item.get("opponent") or ""
        start_time = item.get("start_time") or item.get("game_date_time")

        market_map = {
            "Points": "points",
            "Rebounds": "rebounds",
            "Assists": "assists",
            "Pts + Rebs + Asts": "pra",
            "Pts+Rebs+Asts": "pra",
            "Points + Rebounds + Assists": "pra",
            "Rebs+Asts": "ra",
            "Rebs + Asts": "ra",
            "Rebounds + Assists": "ra",
            "3-Pointers Made": "threes",
            "Fantasy Score": "fs",
            "Fantasy Points": "fs",
        }
        market = market_map.get(stat_type)
        if market is None:
            # Fallback: keep unknown markets instead of dropping them completely
            # This helps when PrizePicks changes labels (e.g. "Points (Alt)")
            market = (
                str(stat_type)
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("+", "_")
                .replace("-", "_")
            )
            if not market:
                continue

        # skip obvious combos like "Player A + Player B"
        if "+" in player:
            continue

        # skip split opponents like "LAL / LAC"
        if any(sep in opponent for sep in ("/", "|", "+")):
            continue

        # basic sanity filters
        if market == "points" and line_val < 10:
            continue
        if market == "rebounds" and line_val < 3:
            continue
        if market == "assists" and line_val < 3:
            continue
        if market == "pra" and line_val < 15:
            continue
        if market == "ra" and line_val < 6:
            continue
        if market == "fs" and line_val < 15:
            continue

        records.append(
            {
                "player": player,
                "team": team,
                "opponent": opponent,
                "market": market,
                "line": line_val,
                "game_time": start_time,
                "book": "PrizePicks",
            }
        )

    if not records:
        return 0

    df_all = pd.DataFrame(records)
    df_all["line"] = pd.to_numeric(df_all["line"], errors="coerce")
    df_all = df_all.dropna(subset=["line"])

    # Compute board_line per (player, team, opponent, market)
    def _board_from_series(s: pd.Series) -> float:
        s_sorted = s.sort_values()
        n = len(s_sorted)
        if n == 0:
            return float("nan")
        if n % 2 == 1:
            pos = n // 2
        else:
            pos = n // 2 - 1
        return float(s_sorted.iloc[pos])

    df_all["board_line"] = df_all.groupby(
        ["player", "team", "opponent", "market"]
    )["line"].transform(_board_from_series)

    df_all["is_board_line"] = df_all["line"] == df_all["board_line"]

    # Let the existing training helper handle dedupe & file writing
    try:
        append_edges_to_training_dataset(df_all)
    except Exception:
        # don't crash app if write fails
        return 0

    return len(df_all)


def load_prizepicks_nba_props() -> pd.DataFrame:
    """
    Pull cleaned NBA props from PrizePicks.

    - NBA only
    - active/open markets
    - full-game only
    """
    try:
        pp = PrizePick()
        raw = pp.get_data(organize_data=False)
    except TypeError:
        pp = PrizePick()
        raw = pp.get_data(False)
    except Exception as e:
        st.error(f"Error loading PrizePicks data: {e}")
        return pd.DataFrame(
            columns=["player_name", "team", "opponent", "market", "line", "game_time", "book"]
        )

    if not isinstance(raw, list):
        st.warning("PrizePicks data not in expected list format.")
        return pd.DataFrame(
            columns=["player_name", "team", "opponent", "market", "line", "game_time", "book"]
        )

    records = []

    records = []


    for item in raw:
        if not isinstance(item, dict):
            continue

        league = item.get("league", "")
        if "NBA" not in str(league).upper():
            continue

        # status: keep anything that isn't clearly closed/settled/void
        status = str(item.get("status", "")).lower()
        bad_statuses = [
            "settled", "graded", "closed", "void", "canceled",
            "cancelled", "refunded", "suspended"
        ]
        if any(bad in status for bad in bad_statuses):
            continue

        stat_type = item.get("stat_type")

        if not _is_full_game_prop_pp(item, stat_type):
            continue

        line_val = item.get("line_score")
        if line_val is None:
            line_val = item.get("line_value")
        if line_val is None:
            continue

        try:
            line_val = float(line_val)
        except Exception:
            continue

        player_name = item.get("player_name") or item.get("name")
        if not player_name:
            continue

        team = item.get("team")
        opponent = item.get("opponent") or ""
        start_time = item.get("start_time") or item.get("game_date_time")

        market_map = {
            "Points": "points",
            "Rebounds": "rebounds",
            "Assists": "assists",
            "Pts + Rebs + Asts": "pra",
            "Pts+Rebs+Asts": "pra",
            "Points + Rebounds + Assists": "pra",
            "Rebs+Asts": "ra",
            "Rebs + Asts": "ra",
            "Rebounds + Assists": "ra",
            "3-Pointers Made": "threes",
            "Fantasy Score": "fs",
            "Fantasy Points": "fs",
        }
        market = market_map.get(stat_type)
        if market is None:
            # Fallback: keep unknown markets instead of dropping them completely
            # This helps when PrizePicks changes labels (e.g. "Points (Alt)")
            market = (
                str(stat_type)
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("+", "_")
                .replace("-", "_")
            )
            if not market:
                continue

        # skip combos etc.
        if "+" in player_name:
            continue

        # skip split opponents
        if any(sep in opponent for sep in ("/", "|", "+")):
            continue

        odds_type = str(item.get("odds_type", "")).lower()

        records.append(
            {
                "player_name": player_name,
                "team": team,
                "opponent": opponent,
                "market": market,
                "line": line_val,
                "game_time": start_time,
                "book": "PrizePicks",
                "odds_type": odds_type,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=["player_name", "team", "opponent", "market", "line", "game_time", "book"]
        )

    df = pd.DataFrame(records)
    st.write('DEBUG: Records after cleaning/filters:', len(records))
    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df = df.dropna(subset=["line"])
    df = df.sort_values("line")

    df = df.groupby(
        ["player_name", "team", "opponent", "market"],
        as_index=False,
    ).last()

    df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    return df[["player_name", "team", "opponent", "market", "line", "game_time", "book", "odds_type"]]


# =====================================================
# CSV LOADER (OPTIONAL MANUAL PROPS)
# =====================================================
@st.cache_data
def load_props_from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)

    required_cols = ["player_name", "team", "opponent", "market", "line", "game_time"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")
    if "book" not in df.columns:
        df["book"] = "Custom"

    df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df = df.dropna(subset=["line"])

    return df[required_cols + ["book"]]


# =====================================================
# NBA STATS VIA nba_api
# =====================================================
@st.cache_data
def get_all_players():
    return players.get_players()


@st.cache_data
def get_all_teams():
    """Fetch NBA teams from nba_api (logo metadata)."""
    return teams.get_teams()


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower()
    for ch in [".", ",", "'", "`"]:
        name = name.replace(ch, "")
    name = " ".join(name.split())
    return name


def get_team_logo_url(team_name: str | None) -> str | None:
    """
    Map team string (like 'LAL', 'Lakers', 'Los Angeles Lakers') to NBA logo URL.
    """
    if not team_name:
        return None

    raw = str(team_name).strip()
    if not raw:
        return None

    parts = raw.replace(".", " ").split()
    abbrev_candidates = set()

    if len(raw) <= 3:
        abbrev_candidates.add(raw.upper())

    if parts:
        first = parts[0]
        last = parts[-1]
        for p in (first, last):
            if 1 <= len(p) <= 3:
                abbrev_candidates.add(p.upper())

    try:
        all_teams = get_all_teams()
    except Exception:
        return None

    if not all_teams:
        return None

    # 1) Try abbreviation candidates first
    for t in all_teams:
        abbr = str(t.get("abbreviation", "")).upper()
        if abbr in abbrev_candidates:
            return f"https://cdn.nba.com/logos/nba/{t['id']}/primary/L/logo.svg"

    lower_raw = raw.lower()

    # 2) Exact full_name match
    for t in all_teams:
        if lower_raw == str(t.get("full_name", "")).lower():
            return f"https://cdn.nba.com/logos/nba/{t['id']}/primary/L/logo.svg"

    # 3) Exact nickname match (e.g. 'Lakers')
    for t in all_teams:
        if lower_raw == str(t.get("nickname", "")).lower():
            return f"https://cdn.nba.com/logos/nba/{t['id']}/primary/L/logo.svg"

    # 4) Partial match inside full_name
    for t in all_teams:
        if lower_raw in str(t.get("full_name", "")).lower():
            return f"https://cdn.nba.com/logos/nba/{t['id']}/primary/L/logo.svg"

    return None


@st.cache_data
def get_player_id(player_name: str):
    all_players = get_all_players()
    if not player_name:
        return None

    target = normalize_name(player_name)

    # exact match
    for p in all_players:
        if normalize_name(p["full_name"]) == target:
            return p["id"]

    # contains match
    for p in all_players:
        norm = normalize_name(p["full_name"])
        if target and (target in norm or norm in target):
            return p["id"]

    # initials like "K. Caldwell-Pope"
    parts = target.split()
    if len(parts) > 1 and len(parts[0]) == 1:
        target_no_initial = " ".join(parts[1:])
        for p in all_players:
            norm = normalize_name(p["full_name"])
            if target_no_initial and (target_no_initial in norm or norm in target_no_initial):
                return p["id"]

    # last-name-only unique match
    if len(parts) >= 2:
        last_name = parts[-1]
        candidates = []
        for p in all_players:
            norm = normalize_name(p["full_name"])
            if last_name and last_name in norm:
                candidates.append(p["id"])
        if len(candidates) == 1:
            return candidates[0]

    return None


@st.cache_data
def get_player_gamelog(player_id: int) -> pd.DataFrame:
    if player_id is None:
        return pd.DataFrame()
    try:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=SeasonAll.all)
        df = gl.get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
        df = df.sort_values("GAME_DATE", ascending=False)

        # add simple home/away flag
        if "MATCHUP" in df.columns:
            df["IS_AWAY"] = df["MATCHUP"].astype(str).str.contains("@")
        else:
            df["IS_AWAY"] = False

        return df
    except Exception:
        return pd.DataFrame()


def get_market_series(gamelog_df: pd.DataFrame, market: str) -> pd.Series:
    market = (market or "").lower().strip()
    if gamelog_df.empty:
        return pd.Series(dtype="float")

    for col in ["PTS", "REB", "AST", "FG3M", "STL", "BLK", "TOV"]:
        if col not in gamelog_df.columns:
            gamelog_df[col] = 0

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
        return (
            gamelog_df["PTS"]
            + 1.2 * gamelog_df["REB"]
            + 1.5 * gamelog_df["AST"]
            + 3.0 * (gamelog_df["STL"] + gamelog_df["BLK"])
            - gamelog_df["TOV"]
        )

    return pd.Series(dtype="float")


def prob_to_american(p: float) -> str:
    """Convert probability (0–1) to American odds string."""
    try:
        p = float(p)
    except Exception:
        return "N/A"

    if p <= 0.0 or p >= 1.0:
        return "N/A"

    if p >= 0.5:
        odds = -round((p / (1 - p)) * 100)
    else:
        odds = round(((1 - p) / p) * 100)

    return f"{odds:+d}"



def prob_to_tier(p: float) -> str:
    """Map a probability (0–1) to a confidence tier string."""
    try:
        p = float(p)
    except Exception:
        return "N/A"
    if p >= 0.65:
        return "GREEN (High)"
    if p >= 0.55:
        return "YELLOW (Medium)"
    if p >= 0.0:
        return "RED (Low)"
    return "N/A"


def unified_recommendation(bet_side: str | None, ml_prob_over: float | None) -> tuple[str, str]:
    """Combine historical recommendation and ML over-prob into a unified pick.

    Returns (unified_side, unified_label), where unified_side is one of
    "Over", "Under", or "Pass", and unified_label is a human-readable
    description (e.g. "Over (Strong agreement)" or "Pass (Disagree)".
    """
    side_clean = (bet_side or "No clear edge") if isinstance(bet_side, str) else "No clear edge"

    if ml_prob_over is None:
        return side_clean, side_clean

    try:
        p = float(ml_prob_over)
    except Exception:
        return side_clean, side_clean

    if not (0.0 <= p <= 1.0):
        return side_clean, side_clean

    strong_hi = 0.65
    lean_hi = 0.55
    strong_lo = 0.35
    lean_lo = 0.45

    unified_side = "Pass"
    label = "Pass"

    # ML favors OVER
    if p >= lean_hi:
        if p >= strong_hi:
            if side_clean == "Over":
                unified_side = "Over"
                label = "Over (Strong agreement)"
            elif side_clean == "Under":
                unified_side = "Pass"
                label = "Pass (Hist Under vs ML Over)"
            else:
                unified_side = "Over"
                label = "Over (Strong ML edge)"
        else:  # lean over (0.55–0.65)
            if side_clean == "Over":
                unified_side = "Over"
                label = "Over (Lean agreement)"
            elif side_clean == "Under":
                unified_side = "Pass"
                label = "Pass (Disagree: Hist Under vs ML Over)"
            else:
                unified_side = "Over"
                label = "Over (Lean ML edge)"
    # ML favors UNDER
    elif p <= lean_lo:
        if p <= strong_lo:
            if side_clean == "Under":
                unified_side = "Under"
                label = "Under (Strong agreement)"
            elif side_clean == "Over":
                unified_side = "Pass"
                label = "Pass (Hist Over vs ML Under)"
            else:
                unified_side = "Under"
                label = "Under (Strong ML edge)"
        else:  # lean under (0.35–0.45)
            if side_clean == "Under":
                unified_side = "Under"
                label = "Under (Lean agreement)"
            elif side_clean == "Over":
                unified_side = "Pass"
                label = "Pass (Disagree: Hist Over vs ML Under)"
            else:
                unified_side = "Under"
                label = "Under (Lean ML edge)"
    else:
        unified_side = "Pass"
        label = "Pass (ML near 50/50)"

    return unified_side, label


def american_to_decimal(odds_str: str | float | int | None) -> float | None:
    """Convert American odds to decimal odds.

    Returns None if odds are missing or malformed.
    """
    if odds_str is None:
        return None
    try:
        if isinstance(odds_str, (int, float)):
            odds = float(odds_str)
        else:
            s = str(odds_str).strip().upper()
            if s in {"N/A", "", "EVEN"}:
                if s == "EVEN":
                    return 2.0
                return None
            odds = float(s)
    except Exception:
        return None

    if odds == 0:
        return None

    if odds > 0:
        return 1.0 + (odds / 100.0)
    else:
        return 1.0 + (100.0 / abs(odds))


def kelly_fraction(p: float, decimal_odds: float) -> float | None:
    """Compute Kelly fraction for probability p and decimal odds.

    Returns fraction of bankroll to wager (0–1), or None if invalid.
    """
    try:
        p = float(p)
        d = float(decimal_odds)
    except Exception:
        return None

    if not (0.0 < p < 1.0) or d <= 1.0:
        return None

    b = d - 1.0
    q = 1.0 - p
    f = (b * p - q) / b
    return f if f > 0 else None
# =====================================================
# SIDEBAR UI
# =====================================================
st.sidebar.title("🏀 NBA Outlier – Prop Edge Finder v2.1.0 ML")

# Optional: collect full PrizePicks ladders (board + alts) into training CSV
if st.sidebar.button("📥 Collect ALL PrizePicks ladders to training CSV"):
    with st.spinner("Collecting full PrizePicks ladders for training..."):
        n_rows = collect_full_ladders_to_training_from_prizepicks()
    st.sidebar.success(f"Appended ladders for {n_rows} raw lines (before de-dup).")

# ---- NBA ML Status (sidebar) ----


# ---- NBA ML Status (sidebar) ----
st.sidebar.markdown("### 🤖 NBA ML Status")

if over_model is None:
    st.sidebar.write("Model: **fallback only (no over_model.pkl)**")
else:
    st.sidebar.write("Model: **over_model.pkl loaded**")
    if over_model_trained_at is not None:
        st.sidebar.write(
            f"Last trained (file time): `{over_model_trained_at:%Y-%m-%d %H:%M}`"
        )

mode = st.sidebar.radio(
    "Prop source",
    [
        "PrizePicks (live)",
        "Upload CSV manually",
    ],
)

games_to_look_back = st.sidebar.slider(
    "Games to look back (N)", min_value=5, max_value=25, value=10, step=1
)

min_over_rate = st.sidebar.slider(
    "Minimum Over % (last N games)", min_value=0.0, max_value=1.0, value=0.6, step=0.05
)

min_edge = st.sidebar.number_input("Minimum Edge (Avg - Line)", value=1.0, step=0.5)

min_confidence = st.sidebar.slider(
    "Minimum Confidence %", min_value=0, max_value=100, value=60, step=5
)

only_today = st.sidebar.checkbox("Only today's games (by game_time)", value=False)

# ML-style team model controls
min_props_for_ml = st.sidebar.slider(
    "Min props per team for moneyline/spread model", min_value=1, max_value=20, value=3, step=1
)

spread_highlight = st.sidebar.slider(
    "Min |model spread| to flag a game",
    min_value=0.0,
    max_value=15.0,
    value=4.5,
    step=0.5,
)

ml_winprob_highlight = st.sidebar.slider(
    "Min win % to flag ML favorite",
    min_value=50.0,
    max_value=80.0,
    value=60.0,
    step=1.0,
)

# Mobile layout toggle
mobile_mode = st.sidebar.checkbox("Mobile mode (simplified layout)", value=False)

st.sidebar.markdown("### Quick filter preset")
preset = st.sidebar.selectbox(
    "Preset (optional)",
    ["Custom", "High Edge", "High Confidence", "Loose"],
)

edge_threshold = min_edge
over_threshold = min_over_rate
conf_threshold = min_confidence

if preset == "High Edge":
    edge_threshold = max(edge_threshold, 3.0)
    over_threshold = max(over_threshold, 0.6)
    conf_threshold = max(conf_threshold, 65)
elif preset == "High Confidence":
    edge_threshold = max(edge_threshold, 1.5)
    over_threshold = max(over_threshold, 0.7)
    conf_threshold = max(conf_threshold, 70)
elif preset == "Loose":
    edge_threshold = min(edge_threshold, 0.5)
    over_threshold = min(over_threshold, 0.5)
    conf_threshold = min(conf_threshold, 50)

if over_model is None:
    st.sidebar.info(
        "ML file 'over_model.pkl' not found – using fallback ML-style probabilities for Over."
    )


# =====================================================
# LOAD PROPS
# =====================================================
props_df = None

if mode == "PrizePicks (live)":
    with st.spinner("Loading NBA props from PrizePicks (DFS_Wrapper)..."):
        try:
            props_df = load_prizepicks_nba_props()
        except Exception as e:
            st.error(f"PrizePicks load failed: {e}")
            props_df = pd.DataFrame()
elif mode == "Upload CSV manually":
    uploaded = st.sidebar.file_uploader(
        "Upload props CSV",
        type=["csv"],
        help="Must contain: player_name, team, opponent, market, line, game_time (optional: book)",
    )
    if uploaded:
        props_df = load_props_from_csv(uploaded)

if props_df is None or props_df.empty:
    st.info("No props loaded yet (check PrizePicks or upload a CSV).")
    st.stop()

# create simple game label
props_df["game_label"] = props_df.apply(
    lambda r: f"{r['team']} vs {r['opponent']}"
    if pd.notna(r.get("team")) and pd.notna(r.get("opponent"))
    else "Unknown",
    axis=1,
)

# safer "only today" filter: keep NaT rows, only drop clearly non-today games
if only_today and "game_time" in props_df.columns:
    today = datetime.today().date()
    has_time = props_df["game_time"].notna()
    today_mask = has_time & (props_df["game_time"].dt.date == today)
    no_time_mask = ~has_time
    props_df = pd.concat(
        [props_df[today_mask], props_df[no_time_mask]],
        ignore_index=True,
    )

if props_df.empty:
    st.warning("No props after applying date filter.")
    st.stop()

# Top filters
teams = ["All teams"] + sorted(props_df["team"].dropna().unique().tolist())
markets = ["All markets"] + sorted(props_df["market"].dropna().unique().tolist())
games = ["All games"] + sorted(props_df["game_label"].dropna().unique().tolist())

top_col1, top_col2, top_col3, top_col4 = st.columns([2, 2, 3, 3])
with top_col1:
    team_filter = st.selectbox("Team filter", teams)
with top_col2:
    market_filter = st.selectbox("Market filter (points, pra, etc)", markets)
with top_col3:
    game_filter = st.selectbox("Game filter", games)
with top_col4:
    search_name = st.text_input("Search player (optional)", "")

df = props_df.copy()

if team_filter != "All teams":
    df = df[df["team"] == team_filter]
if market_filter != "All markets":
    df = df[df["market"].str.lower() == market_filter.lower()]
if game_filter != "All games":
    df = df[df["game_label"] == game_filter]

if df.empty:
    st.warning("No props match the selected filters.")
    st.stop()


# =====================================================
# EDGE / CONFIDENCE / PREDICTION / BET SIDE / ODDS / ML
# =====================================================
st.title("NBA Prop Edge Finder (PrizePicks) – v2.1.0 ML")

st.write("### Calculating edges…")

rows = []
errors = []

# Debug counters for why rows are skipped
debug_counts = {
    "missing_name_or_market_or_line": 0,
    "missing_player_log": 0,
    "invalid_line": 0,
    "no_stats_for_market": 0,
    "no_recent_games": 0,
    "ok_rows": 0,
}


unique_players = sorted(df["player_name"].dropna().unique().tolist())
player_logs = {}
player_ids = {}

progress = st.progress(0.0)
status_text = st.empty()

total_players = len(unique_players) if unique_players else 1


def build_ml_features(
    season_avg: float,
    avg_last_n: float,
    edge_n: float,
    over_rate_n: float,
    last_game_stat: float,
    line_float: float,
    is_home: int,
    days_rest: int,
) -> dict:
    return {
        "season_avg": season_avg,
        "last_n_avg": avg_last_n,
        "edge_last_n": edge_n,
        "over_rate_last_n": over_rate_n,
        "last_game_stat": last_game_stat,
        "line": line_float,
        "line_minus_season": line_float - season_avg,
        "line_minus_last_n": line_float - avg_last_n,
        "is_home": is_home,
        "days_rest": days_rest,
    }


ML_FEATURE_COLS = [
    "season_avg",
    "last_n_avg",
    "edge_last_n",
    "over_rate_last_n",
    "last_game_stat",
    "line",
    "line_minus_season",
    "line_minus_last_n",
    "is_home",
    "days_rest",
]

# fetch logs
for i, name in enumerate(unique_players):
    status_text.text(f"Fetching NBA stats for players… ({i+1}/{total_players})")

    pid = get_player_id(name)
    player_ids[name] = pid
    if pid is None:
        errors.append(f"Player not found in nba_api: {name}")
        continue

    glog = get_player_gamelog(pid)
    if glog.empty:
        errors.append(f"No game log for player: {name}")
        continue

    player_logs[name] = glog

    time.sleep(0.2)  # be gentle to NBA API
    progress.progress((i + 1) / total_players)

progress.progress(1.0)
status_text.text("Computing edges & ML-style probabilities…")


for _, row in df.iterrows():
    player_name = row.get("player_name")
    market = row.get("market")
    line = row.get("line")
    book = row.get("book") or "PrizePicks"

    if player_name is None or market is None or line is None:
        debug_counts["missing_name_or_market_or_line"] += 1
        continue
    if player_name not in player_logs:
        debug_counts["missing_player_log"] += 1
        continue

    try:
        line_float = float(line)
    except Exception:
        errors.append(f"Invalid line for {player_name}: {line}")
        debug_counts["invalid_line"] += 1
        continue

    gamelog = player_logs[player_name]
    series = get_market_series(gamelog, market).dropna()
    if series.empty:
        errors.append(f"No stats for {player_name} – market '{market}'")
        debug_counts["no_stats_for_market"] += 1
        continue

    season_avg = series.mean()

    last_n = series.iloc[:games_to_look_back]
    if last_n.empty:
        errors.append(f"No recent games for {player_name}")
        debug_counts["no_recent_games"] += 1
        continue

    avg_last_n = last_n.mean()
    hits_n = (last_n > line_float).sum()
    games_n = len(last_n)
    over_rate_n = (hits_n + 1) / (games_n + 2)

    edge_n = avg_last_n - line_float

    last7 = series.iloc[:7]
    if len(last7) > 0:
        avg_last_7 = last7.mean()

        hits_7 = (last7 > line_float).sum()
        games_7 = len(last7)
        over_rate_7 = (hits_7 + 1) / (games_7 + 2)

        edge_7 = avg_last_7 - line_float
    else:
        avg_last_7 = None
        over_rate_7 = None
        edge_7 = None


    # blended prediction (historical-based)
    w_season = 0.4
    w_last_n = 0.4
    w_last_7 = 0.2

    avg7_for_blend = avg_last_7 if avg_last_7 is not None else avg_last_n
    predicted_score = (
        w_season * season_avg
        + w_last_n * avg_last_n
        + w_last_7 * avg7_for_blend
    )

    # Confidence based on historical only (unchanged)
    hit_score = over_rate_n
    if line_float != 0:
        edge_ratio = max(0.0, edge_n / max(1.0, line_float))
    else:
        edge_ratio = 0.0
    edge_score = max(0.0, min(1.0, edge_ratio * 4.0))
    confidence = 0.6 * hit_score + 0.4 * edge_score
    confidence_pct = round(confidence * 100, 1)

    # Historical-based probabilities and odds
    over_prob_hist = float(over_rate_n)
    over_prob_hist_clamped = min(max(over_prob_hist, 0.01), 0.99)
    under_prob_hist = 1.0 - over_prob_hist_clamped
    over_odds_hist = prob_to_american(over_prob_hist_clamped)
    under_odds_hist = prob_to_american(under_prob_hist)

    # Determine bet side from historical blended prediction
    delta = predicted_score - line_float
    if delta >= 0.5:
        bet_side = "Over"
    elif delta <= -0.5:
        bet_side = "Under"
        # otherwise
    else:
        bet_side = "No clear edge"

    if bet_side == "Over":
        bet_odds = over_odds_hist
    elif bet_side == "Under":
        bet_odds = under_odds_hist
    else:
        bet_odds = "N/A"

    # =====================
    # =====================
    # ML PROBABILITY OF OVER
    # =====================

    ml_prob_over = None
    ml_odds_over = None
    ml_prob_pct = None
    ml_conf_tier = "N/A"
    unified_side = bet_side
    unified_pick = bet_side if bet_side is not None else "No clear edge"
    kelly_full = None
    kelly_half = None

    try:
        last_game_stat = float(series.iloc[0]) if len(series) > 0 else avg_last_n

        if "IS_AWAY" in gamelog.columns and len(gamelog) > 0:
            is_away = bool(gamelog["IS_AWAY"].iloc[0])
            is_home = 0 if is_away else 1
        else:
            is_home = 1

        if len(gamelog) >= 2:
            days_rest = (gamelog["GAME_DATE"].iloc[0] - gamelog["GAME_DATE"].iloc[1]).days
            days_rest = max(days_rest, 0)
        else:
            days_rest = 2

        if over_model is not None:
            feat_dict = build_ml_features(
                season_avg=season_avg,
                avg_last_n=avg_last_n,
                edge_n=edge_n,
                over_rate_n=over_rate_n,
                last_game_stat=last_game_stat,
                line_float=line_float,
                is_home=is_home,
                days_rest=days_rest,
            )

            extra_features = {
                "avg_last_7": avg_last_n,
                "avg_last_main": avg_last_n,
                "minutes_avg_last_3": 0.0,
                "predicted_score": predicted_score,
                "over_rate_main": over_rate_n,
                "over_rate_last_7": over_rate_n,
                "opp_over_rate": 0.5,
                "stat_trend": (avg_last_n - season_avg) if season_avg is not None else 0.0,
                "edge_last_main": edge_n,
                "opp_samples": 0,
                "confidence_pct": confidence_pct,
                "minutes_trend": 0.0,
                "market": str(market),
                "minutes_avg_last_n": 0.0,
                "opp_actual_mean": 0.0,
                "edge_last_7": edge_n,
            }

            for k, v in extra_features.items():
                feat_dict.setdefault(k, v)

            # Build DataFrame for ML model
            X_row = pd.DataFrame([feat_dict])

            # Align feature columns with what the model was actually trained on
            market_key = str(market).lower().strip()
            model_for_market = over_models_by_market.get(market_key, over_model)
            feat_cols = getattr(model_for_market, "feature_names_in_", None)
            if feat_cols is not None:
                # Add any missing columns with 0.0 so sklearn doesn't complain
                for col in feat_cols:
                    if col not in X_row.columns:
                        X_row[col] = 0.0
                # Restrict and order columns exactly as in training
                X_row = X_row[list(feat_cols)]

            prob = float(model_for_market.predict_proba(X_row)[0][1])
            prob = max(0.001, min(0.999, prob))

            ml_prob_over = prob
            ml_odds_over = prob_to_american(prob)

        else:
            edge_for_blend = edge_n
            edge_scaled = max(-5.0, min(5.0, edge_for_blend / max(1.0, line_float) * 4.0))
            base_logit = math.log(over_prob_hist_clamped / (1 - over_prob_hist_clamped))
            blended_logit = base_logit + 0.8 * edge_scaled
            prob = 1.0 / (1.0 + math.exp(-blended_logit))
            prob = max(0.001, min(0.999, prob))

            ml_prob_over = prob
            ml_odds_over = prob_to_american(prob)

        # Post-process ML outputs: confidence tier, unified pick, Kelly
        if ml_prob_over is not None:
            ml_prob_pct = ml_prob_over * 100.0
            ml_conf_tier = prob_to_tier(ml_prob_over)

            unified_side, unified_pick = unified_recommendation(bet_side, ml_prob_over)

            odds_for_side = None
            p_side = None
            if unified_side == "Over":
                odds_for_side = over_odds_hist
                p_side = ml_prob_over
            elif unified_side == "Under":
                odds_for_side = under_odds_hist
                p_side = 1.0 - ml_prob_over

            if p_side is not None and odds_for_side not in (None, "N/A"):
                dec_odds = american_to_decimal(odds_for_side)
                if dec_odds is not None:
                    frac = kelly_fraction(p_side, dec_odds)
                    if frac is not None and frac > 0:
                        kelly_full = frac
                        kelly_half = frac / 2.0

    except Exception as e:
        errors.append(f"ML error for {player_name} {market}: {e}")
        ml_prob_over = None
        ml_odds_over = None
        ml_prob_pct = None
        ml_conf_tier = "N/A"
        unified_side = bet_side
        unified_pick = bet_side if bet_side is not None else "No clear edge"
        kelly_full = None
        kelly_half = None

    debug_counts["ok_rows"] += 1

    rows.append(
        {
            "player": player_name,
            "player_id": player_ids.get(player_name),
            "team": row.get("team"),
            "opponent": row.get("opponent"),
            "market": market,
            "line": line_float,
            "book": book,
            "bet_side": bet_side,
            "season_avg": round(season_avg, 2),
            f"avg_last_{games_to_look_back}": round(avg_last_n, 2),
            f"over_rate_last_{games_to_look_back}": round(over_rate_n, 2),
            f"edge_last_{games_to_look_back}": round(edge_n, 2),
            "avg_last_7": round(avg_last_7, 2) if avg_last_7 is not None else None,
            "over_rate_last_7": round(over_rate_7, 2) if over_rate_7 is not None else None,
            "edge_last_7": round(edge_7, 2) if edge_7 is not None else None,
            "predicted_score": round(predicted_score, 2),
            "confidence_pct": confidence_pct,
            "over_prob": round(over_prob_hist_clamped, 3),
            "under_prob": round(under_prob_hist, 3),
            "over_odds": over_odds_hist,
            "under_odds": under_odds_hist,
            "bet_odds": bet_odds,
            "ml_prob_over": round(ml_prob_over, 3) if ml_prob_over is not None else None,
            "ml_prob_pct": round(ml_prob_pct, 1) if ml_prob_pct is not None else None,
            "ml_conf_tier": ml_conf_tier,
            "ml_odds_over": ml_odds_over,
            "unified_side": unified_side,
            "unified_pick": unified_pick,
            "kelly_full": round(kelly_full, 4) if kelly_full is not None else None,
            "kelly_half": round(kelly_half, 4) if kelly_half is not None else None,
            "game_time": row.get("game_time"),
            "game_label": row.get("game_label"),
        }
    )

if not rows:
    st.error("No edges could be calculated from the current props.")
    if errors:
        with st.expander("Warnings/errors (players/markets skipped)"):
            for e in errors:
                st.write("- ", e)
    st.stop()

edges_df = pd.DataFrame(rows)

# DEBUG (disabled in clean # (Bobby Portis edges_df debug removed)

# (Bobby Portis props_df debug removed)

# ------------------------------------------------------------------
# OVERRIDE LINE VALUES USING LIVE PRIZEPICKS DATA
# ------------------------------------------------------------------
# ------------------------------------------------------------------
try:
    if (
        isinstance(props_df, pd.DataFrame)
        and {"player_name", "market", "opponent", "line"}.issubset(props_df.columns)
        and {"player", "market", "opponent"}.issubset(edges_df.columns)
    ):
        # Use the most recent live line for each (player, market, opponent)
        pp_lines = (
            props_df.dropna(subset=["player_name", "market", "opponent", "line"])
            .groupby(["player_name", "market", "opponent"], as_index=False)["line"]
            .last()
            .rename(columns={"player_name": "player", "line": "pp_line"})
        )

        if not pp_lines.empty:
            edges_df = edges_df.merge(
                pp_lines,
                on=["player", "market", "opponent"],
                how="left",
            )
            # If live pp_line exists, override the model/CSV line
            edges_df["line"] = edges_df["pp_line"].combine_first(edges_df["line"])
            edges_df.drop(columns=["pp_line"], inplace=True, errors="ignore")
except Exception as _override_err:
    # Fail quietly; better to show something than crash the app
    pass

# ------------------------------------------------------------------
# Attach odds_type (standard vs goblins/demons) from props_df so we can
# show and analyze alt lines separately from main board lines.
# ------------------------------------------------------------------
try:
    if (
        isinstance(props_df, pd.DataFrame)
        and {"player_name", "team", "opponent", "market", "line", "odds_type"}.issubset(props_df.columns)
    ):
        odds_map = (
            props_df[["player_name", "team", "opponent", "market", "line", "odds_type"]]
            .rename(columns={"player_name": "player"})
            .drop_duplicates()
        )
        edges_df = edges_df.merge(
            odds_map,
            on=["player", "team", "opponent", "market", "line"],
            how="left",
        )
except Exception:
    # odds_type is helpful but optional – don't crash if merge fails
    pass


# Append to training dataset for offline ML model
try:
    append_edges_to_training_dataset(edges_df)
except Exception:
    pass

edge_cols = [c for c in edges_df.columns if c.startswith("edge_last_") and "7" not in c]
rate_cols = [c for c in edges_df.columns if c.startswith("over_rate_last_") and "7" not in c]

if not edge_cols or not rate_cols:
    st.error("Edge/over_rate columns missing from results.")
    st.stop()

edge_col = edge_cols[0]
rate_col = rate_cols[0]

# Toggle: include alt (Goblin/Demon) lines in main views or not
include_alt_main = st.sidebar.checkbox(
    "Include Goblin/Demon alt lines in main views",
    value=False,
    help="When checked, main tables/cards will also show Goblin/Demon (alt) lines. "
         "When unchecked, only standard board lines are shown here (alts stay in the Alt Lines section).",
)

if include_alt_main:
    main_edges_df = edges_df.copy()
else:
    # For main views (cards, table, slips), restrict to standard board lines only.
    # Alt lines (Goblins/Demons) remain in edges_df and are shown separately.
    standard_mask = edges_df.get("odds_type", "standard").fillna("standard").eq("standard")
    main_edges_df = edges_df[standard_mask].copy()

# Apply player search filter (for card + table views)
if search_name.strip():
    view_df = main_edges_df[main_edges_df["player"].str.contains(search_name, case=False, na=False)]
else:
    view_df = main_edges_df.copy()

if view_df.empty:
    st.warning("No props after applying search filter.")
    st.stop()

if view_df.empty:
    st.warning("No props after applying search filter.")
    st.stop()


# =====================================================
# BUILD GAME MONEYLINE PREDICTIONS + MODEL SPREADS
# =====================================================
def build_moneyline_predictions(edges, scoring_markets=None):
    if scoring_markets is None:
        scoring_markets = ["points", "pra", "fs", "threes"]

    if edges is None or edges.empty:
        return pd.DataFrame()

    required_cols = {"team", "opponent", "market", edge_col, "confidence_pct", "bet_side"}
    if not required_cols.issubset(edges.columns):
        return pd.DataFrame()

    df_ml = edges.copy()
    df_ml = df_ml[df_ml["market"].isin(scoring_markets)]
    if df_ml.empty:
        return pd.DataFrame()

    def make_game_key(row):
        t = str(row.get("team", "NA"))
        o = str(row.get("opponent", "NA"))
        teams_sorted = sorted([t, o])
        return " vs ".join(teams_sorted)

    df_ml["game_key"] = df_ml.apply(make_game_key, axis=1)

    def row_signal(row):
        side = str(row.get("bet_side", ""))
        try:
            e_val = float(row[edge_col])
            conf = float(row.get("confidence_pct", 0)) / 100.0
        except Exception:
            return 0.0
        base = e_val * conf
        if side == "Over":
            return base
        elif side == "Under":
            return -base
        else:
            return 0.0

    df_ml["signal"] = df_ml.apply(row_signal, axis=1)

    grp = df_ml.groupby(
        ["game_key", "team", "opponent"],
        as_index=False,
    ).agg(
        props_count=("signal", "size"),
        avg_confidence=("confidence_pct", "mean"),
        avg_edge=(edge_col, "mean"),
        avg_signal=("signal", "mean"),
    )

    if grp.empty:
        return grp

    def to_prob(sig):
        try:
            return 1.0 / (1.0 + math.exp(-0.8 * sig))
        except OverflowError:
            return 1.0 if sig > 0 else 0.0

    grp["raw_prob"] = grp["avg_signal"].apply(to_prob)
    grp["sum_prob_game"] = grp.groupby("game_key")["raw_prob"].transform("sum")

    def norm_prob(row):
        s = row["sum_prob_game"]
        rp = row["raw_prob"]
        if s <= 0:
            return 0.5
        p = rp / s
        return max(0.05, min(0.95, p))

    grp["win_prob"] = grp.apply(norm_prob, axis=1)
    grp["ml_odds"] = grp["win_prob"].apply(prob_to_american)
    grp["win_prob_pct"] = (grp["win_prob"] * 100).round(1)

    grp["signal_centered"] = grp["avg_signal"] - grp.groupby("game_key")["avg_signal"].transform("mean")
    spread_factor = 7.0
    grp["model_spread"] = (grp["signal_centered"] * spread_factor).round(1)
    grp["game_label"] = grp.apply(
        lambda r: f"{r['team']} vs {r['opponent']}", axis=1
    )

    grp = grp.sort_values(["win_prob", "game_key"], ascending=[False, True])

    return grp


games_ml_df = build_moneyline_predictions(edges_df)


# =====================================================
# TABS: CARDS / TABLE / SLIPS / WHAT-IF ALT LINES / PLAYER DETAIL / GAME ML / BETS
# =====================================================
tab_cards, tab_table, tab_slips, tab_whatif, tab_player, tab_games, tab_bets = st.tabs(
    [
        "Cards & Explanation",
        "Table",
        "AI Slip Builder",
        "What-If Alt Lines",
        "Player Detail",
        "Game Moneylines & Spreads (MODEL)",
        "Bet Tracker (ML & learning)",
    ]
)



# =====================================================
# TAB 1: CARD VIEW
# =====================================================
def render_player_card(r, k_suffix: str):
    card = st.container()
    with card:
        top_col1, top_col2 = st.columns([1, 2])

        # Left: headshot
        with top_col1:
            pid = r.get("player_id")
            if pid:
                try:
                    img_url = f"https://cdn.nba.com/headshots/nba/latest/260x190/{int(pid)}.png"
                    st.image(img_url)
                except Exception:
                    st.write(" ")
            else:
                st.write(" ")

        # Right: basic info
        with top_col2:
            st.markdown(f"### {r['player']}")
            team = r.get("team") or ""
            opp = r.get("opponent") or ""
            st.markdown(f"**{team} vs {opp}**")
            st.markdown(f"*{r['market']}* | Line: `{r['line']}`")

            side = r.get("bet_side", "No clear edge")
            if side == "Over":
                side_emoji = "⬆️"
            elif side == "Under":
                side_emoji = "⬇️"
            else:
                side_emoji = "⚖️"
            st.markdown(f"Recommended: {side_emoji} **{side}**")

            # Unified recommendation (Historical + ML)
            unified_pick = r.get("unified_pick")
            unified_side = r.get("unified_side")
            if unified_pick:
                st.markdown(f"Unified Recommendation: **{unified_pick}**")


            st.markdown(
                f"Predicted: **{r['predicted_score']}**  "
                f"(Season avg: {r['season_avg']})"
            )

        # Edge / hit rate / confidence
        conf = r.get("confidence_pct", 0) or 0
        edge_val = r.get(edge_col, 0) or 0
        hit = r.get(rate_col, 0) or 0

        st.markdown(
            f"Edge: `{edge_val:.2f}` | "
            f"Hit rate (N): `{hit:.2f}` | "
            f"Confidence: `{conf:.1f}%`"
        )
        st.progress(min(max(conf / 100.0, 0.0), 1.0))

        # Historical odds line (based on hit rate)
        side = r.get("bet_side", "No clear edge")
        if side == "Over":
            hist_prob = r.get("over_prob", 0)
            hist_odds = r.get("over_odds", "N/A")
        elif side == "Under":
            hist_prob = r.get("under_prob", 0)
            hist_odds = r.get("under_odds", "N/A")
        else:
            hist_prob = None
            hist_odds = "N/A"

        if hist_prob is not None:
            st.markdown(
                f"Historical {side} odds: `{hist_odds}` "
                f"({hist_prob*100:.1f}% implied)"
            )

        # ML line on the card – fall back to historical if missing
        ml_prob = r.get("ml_prob_over")
        ml_odds = r.get("ml_odds_over")
        ml_prob_pct = r.get("ml_prob_pct")
        ml_tier = r.get("ml_conf_tier", "N/A")

        # Fallback to historical if ML prob/odds missing
        if (ml_prob is None or ml_odds is None) and side in {"Over", "Under"}:
            if side == "Over":
                ml_prob = r.get("over_prob", None)
                ml_odds = r.get("over_odds", None)
            elif side == "Under":
                ml_prob = r.get("under_prob", None)
                ml_odds = r.get("under_odds", None)
            if ml_prob is not None and ml_prob_pct is None:
                ml_prob_pct = ml_prob * 100.0

        if ml_prob is not None and ml_odds is not None and ml_prob_pct is not None:
            st.markdown(
                f"ML Over odds: `{ml_odds}`  "
                f"({ml_prob_pct:.1f}% from model)"
            )
            if ml_tier and ml_tier != "N/A":
                st.markdown(
                    f"ML Tier: **{ml_tier}** "
                    f"({ml_prob_pct:.1f}% over from model)"
                )

        # Kelly suggestion based on unified side (if available)
        kelly_half = r.get("kelly_half")
        kelly_full = r.get("kelly_full")
        unified_side = r.get("unified_side")
        if (
            unified_side in {"Over", "Under"}
            and kelly_half is not None
            and kelly_full is not None
            and kelly_full > 0
        ):
            st.markdown(
                f"Kelly (Unified): `{kelly_half*100:.1f}%` (half) | "
                f"`{kelly_full*100:.1f}%` (full)"
            )

        # Buttons
        if st.button(
            "Why this prop?",
            key=f"why_{r['player']}_{r['market']}_{k_suffix}",
        ):
            st.session_state["explain_row"] = r.to_dict()
        if st.button(
            "Track this bet",
            key=f"track_{r['player']}_{r['market']}_{k_suffix}",
        ):
            bet = {
                "bet_category": "player_prop",
                "player": r["player"],
                "team": r.get("team"),
                "opponent": r.get("opponent"),
                "market": r["market"],
                "line": r["line"],
                "bet_side": r.get("bet_side"),
                "bet_odds": r.get("bet_odds"),
                "predicted_score": r.get("predicted_score"),
                "confidence_pct": r.get("confidence_pct"),
                "game_time": r.get("game_time"),
                "game_key": r.get("game_label"),
                "model_spread": None,
                "win_prob_pct": None,
                "actual_stat": None,
                "result": None,
            }
            st.session_state["bet_tracker"].append(bet)
            save_bets_to_disk(st.session_state["bet_tracker"])
            st.success("Bet added to tracker.")


with tab_cards:
    st.write("### Featured Edges (Card View)")

    filtered_edges = view_df[
        (view_df[rate_col] >= over_threshold)
        & (view_df[edge_col] >= edge_threshold)
        & (view_df["confidence_pct"] >= conf_threshold)
    ]

    if filtered_edges.empty:
        featured_df = view_df.copy()
        st.caption(
            "No props match all filters/preset yet – showing best available edges instead."
        )
    else:
        featured_df = filtered_edges.copy()

    featured_df = featured_df.sort_values(
        by=["confidence_pct", rate_col, edge_col], ascending=False
    ).reset_index(drop=True)

    top_n = min(12, len(featured_df))
    if top_n == 0:
        st.info("No edges available to display.")
    else:
        if mobile_mode:
            for k in range(top_n):
                r = featured_df.iloc[k]
                render_player_card(r, f"m_{k}")
        else:
            for idx in range(0, top_n, 2):
                cols = st.columns(2)
                for j in range(2):
                    k = idx + j
                    if k >= top_n:
                        break
                    r = featured_df.iloc[k]
                    with cols[j]:
                        render_player_card(r, f"d_{k}")

    st.write("### Prop Explanation")

    if "explain_row" in st.session_state:
        er = st.session_state["explain_row"]

        player_name = er.get("player")
        market = er.get("market")
        line = er.get("line")

        st.markdown(
            f"Player: **{player_name}**  \n"
            f"Market: **{market}**  \n"
            f"Line: `{line}`"
        )

        mask = (
            (edges_df["player"] == player_name)
            & (edges_df["market"] == market)
            & (edges_df["line"] == line)
        )
        row_matches = edges_df[mask]
        if row_matches.empty:
            st.warning("Could not find full data for this prop in the edges table.")
        else:
            r = row_matches.iloc[0]

            season_avg = r.get("season_avg")
            avg_last_n = r.get(f"avg_last_{games_to_look_back}")
            over_rate_n = r.get(f"over_rate_last_{games_to_look_back}")
            edge_n = r.get(f"edge_last_{games_to_look_back}")
            avg_last_7 = r.get("avg_last_7")
            over_rate_7 = r.get("over_rate_last_7")
            edge_7 = r.get("edge_last_7")
            predicted_score = r.get("predicted_score")
            confidence_pct = r.get("confidence_pct")
            bet_side = r.get("bet_side")
            over_odds = r.get("over_odds")
            under_odds = r.get("under_odds")
            bet_odds = r.get("bet_odds")
            ml_prob = r.get("ml_prob_over")
            ml_odds = r.get("ml_odds_over")

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("Key Numbers")
                st.write(f"- Season average: {season_avg}")
                st.write(f"- Last {games_to_look_back} avg: {avg_last_n}")
                st.write(f"- Hit rate last {games_to_look_back}: {over_rate_n:.2f}")
                st.write(f"- Edge last {games_to_look_back}: {edge_n:.2f} vs line {line}")
                if avg_last_7 is not None:
                    st.write(f"- Last 7 avg: {avg_last_7}")
                    st.write(f"- Hit rate last 7: {(over_rate_7 or 0):.2f}")
                    st.write(f"- Edge last 7: {(edge_7 or 0):.2f}")

            with col_right:
                st.markdown("Model Result")
                st.write(f"- Predicted score: {predicted_score}")
                st.write(f"- Confidence: {confidence_pct:.1f}%")
                st.write(f"- Historical Over odds: {over_odds}")
                st.write(f"- Historical Under odds: {under_odds}")

                ml_prob_pct = r.get("ml_prob_pct")
                ml_tier = r.get("ml_conf_tier", "N/A")
                if ml_prob_pct is None and ml_prob is not None:
                    ml_prob_pct = ml_prob * 100.0

                if ml_prob_pct is not None and ml_odds is not None:
                    st.write(f"- ML Over probability: {ml_prob_pct:.1f}% ({ml_odds})")
                    if ml_tier and ml_tier != "N/A":
                        st.write(f"- ML Confidence Tier: {ml_tier}")
                else:
                    st.write("- ML Over probability: not available")

                unified_side = r.get("unified_side")
                unified_pick = r.get("unified_pick")
                if unified_pick:
                    st.write(f"- Unified recommendation: {unified_pick}")

                kelly_half = r.get("kelly_half")
                kelly_full = r.get("kelly_full")
                if (
                    unified_side in {"Over", "Under"}
                    and kelly_half is not None
                    and kelly_full is not None
                    and kelly_full > 0
                ):
                    st.write(
                        f"- Kelly (Unified): {kelly_half*100:.1f}% of bankroll (half), "
                        f"{kelly_full*100:.1f}% (full)"
                    )

                st.write(f"- Historical recommended side: {bet_side}  (odds: {bet_odds})")
with tab_table:
    st.write("### Full Edges Table + Player Line Browser")

    # ---------------------------
    # Player filter (same as before)
    # ---------------------------
    table_player_options = ["All players"] + sorted(
        view_df["player"].dropna().unique().tolist()
    )
    selected_table_player = st.selectbox(
        "Filter table by player (optional)",
        table_player_options,
        key="table_player_filter",
    )

    # ---------------------------
    # Base table DataFrame
    # ---------------------------
    def highlight_edges(
        row,
        e_thr=edge_threshold,
        o_thr=over_threshold,
        c_thr=conf_threshold,
    ):
        styles = [""] * len(row)
        try:
            val = float(row[edge_col])
            over = float(row[rate_col])
            conf = float(row["confidence_pct"])
            side = str(row["bet_side"])
        except Exception:
            return styles

        if val >= e_thr and over >= o_thr and conf >= c_thr and side in ("Over", "Under"):
            color = "background-color: #d4f8d4"
        elif val < 0:
            color = "background-color: #f8d4d4"
        elif val > 0:
            color = "background-color: #fff4c2"
        else:
            return styles

        return [color] * len(row)

    display_cols = [
        "player",
        "team",
        "opponent",
        "market",
        "line",
        "book",
        "bet_side",
        "bet_odds",
        "predicted_score",
        "season_avg",
        f"avg_last_{games_to_look_back}",
        f"over_rate_last_{games_to_look_back}",
        f"edge_last_{games_to_look_back}",
        "avg_last_7",
        "over_rate_last_7",
        "edge_last_7",
        "over_prob",
        "over_odds",
        "under_prob",
        "under_odds",
        "ml_prob_over",
        "ml_prob_pct",
        "ml_conf_tier",
        "ml_odds_over",
        "unified_side",
        "unified_pick",
        "kelly_full",
        "kelly_half",
        "confidence_pct",
        "game_time",
    ]

    table_df = view_df.copy()
    for col in display_cols:
        if col not in table_df.columns:
            table_df[col] = None

    # -------------------------------------------------
    # Player Line Browser: show ALL lines for 1 player
    # -------------------------------------------------
    detailed_row = None  # will hold the row we drill into

    if selected_table_player != "All players":
        player_lines_df = table_df[table_df["player"] == selected_table_player].copy()

        if player_lines_df.empty:
            st.info("No lines available for this player with the current filters.")
        else:
            st.markdown("#### Lines for selected player")

            # Try to sort by game_time if present, then by market/line
            if "game_time" in player_lines_df.columns:
                try:
                    player_lines_df["game_time"] = pd.to_datetime(
                        player_lines_df["game_time"], errors="coerce"
                    )
                except Exception:
                    pass
                player_lines_df = player_lines_df.sort_values(
                    by=["game_time", "market", "line"],
                    ascending=[True, True, True],
                )
            else:
                player_lines_df = player_lines_df.sort_values(
                    by=["market", "line"], ascending=[True, True]
                )

            line_labels: list[str] = []
            line_indices: list[int] = []

            for idx, r in player_lines_df.iterrows():
                market = r.get("market", "?")
                ln = r.get("line")
                book = r.get("book", "PP")
                side = r.get("bet_side", "?")

                try:
                    edge_val = float(r.get(edge_col, 0) or 0.0)
                except Exception:
                    edge_val = 0.0
                try:
                    conf = float(r.get("confidence_pct", 0) or 0.0)
                except Exception:
                    conf = 0.0

                ml_pct = r.get("ml_prob_pct")

                parts = [
                    f"{market} {ln}",
                    f"Book: {book}",
                    f"Side: {side}",
                    f"Edge: {edge_val:.2f}",
                    f"Conf: {conf:.1f}%",
                ]
                if ml_pct is not None:
                    parts.append(f"ML: {ml_pct:.1f}%")

                label = " | ".join(parts)
                line_labels.append(label)
                line_indices.append(idx)

            selected_label = st.selectbox(
                "Pick a specific line to see detailed odds & predictions",
                ["-- None --"] + line_labels,
                key="table_player_line_picker",
            )

            if selected_label != "-- None --":
                pos = line_labels.index(selected_label)
                detailed_row = player_lines_df.loc[line_indices[pos]]

        # Main table filtered to that player
        table_df_filtered = table_df[table_df["player"] == selected_table_player].copy()
    else:
        # No player filter, we show everything
        table_df_filtered = table_df.copy()

    # ---------------------------
    # Styled table (same as before)
    # ---------------------------
    if table_df_filtered.empty:
        st.warning("No rows for this player with the current filters.")
    else:
        styled_edges = (
            table_df_filtered[display_cols]
            .sort_values(
                by=["confidence_pct", rate_col, edge_col],
                ascending=False,
            )
            .style.apply(highlight_edges, axis=1)
        )
        st.dataframe(styled_edges, use_container_width=True)

    # -------------------------------------------------
    # Detailed panel for the selected line
    # -------------------------------------------------
    if detailed_row is not None:
        r = detailed_row

        st.markdown("---")
        st.markdown("### Selected Line – Detailed View")

        st.markdown(
            f"**{r.get('player', '')} – {r.get('team', '')} vs {r.get('opponent', '')}**"
        )
        st.write(
            f"Market: **{r.get('market')}**, "
            f"Line: **{r.get('line')}**, "
            f"Book: **{r.get('book', 'PP')}**"
        )

        # Basic numbers
        season_avg = r.get("season_avg")
        avg_last_n = r.get(f"avg_last_{games_to_look_back}")
        over_rate_n = r.get(f"over_rate_last_{games_to_look_back}")
        edge_n = r.get(f"edge_last_{games_to_look_back}")
        avg_last_7 = r.get("avg_last_7")
        over_rate_7 = r.get("over_rate_last_7")
        edge_7 = r.get("edge_last_7")
        predicted_score = r.get("predicted_score")
        confidence_pct = r.get("confidence_pct", 0) or 0
        over_odds = r.get("over_odds")
        under_odds = r.get("under_odds")
        ml_prob = r.get("ml_prob_over")
        ml_odds = r.get("ml_odds_over")
        bet_side = r.get("bet_side", "N/A")
        bet_odds = r.get("bet_odds")

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**Key Numbers**")
            st.write(f"- Season average: {season_avg}")
            st.write(f"- Last {games_to_look_back} avg: {avg_last_n}")
            st.write(
                f"- Hit rate last {games_to_look_back}: {float(over_rate_n or 0):.2f}"
            )
            st.write(
                f"- Edge last {games_to_look_back}: {float(edge_n or 0):.2f} vs line {r.get('line')}"
            )
            if avg_last_7 is not None:
                st.write(f"- Last 7 avg: {avg_last_7}")
                st.write(f"- Hit rate last 7: {float(over_rate_7 or 0):.2f}")
                st.write(f"- Edge last 7: {float(edge_7 or 0):.2f}")

        with col_right:
            st.markdown("**Model Result**")
            st.write(f"- Predicted score: {predicted_score}")
            st.write(f"- Confidence: {confidence_pct:.1f}%")
            st.write(f"- Historical Over odds: {over_odds}")
            st.write(f"- Historical Under odds: {under_odds}")

            ml_prob_pct = r.get("ml_prob_pct")
            ml_tier = r.get("ml_conf_tier", "N/A")
            if ml_prob_pct is None and ml_prob is not None:
                ml_prob_pct = ml_prob * 100.0

            if ml_prob_pct is not None and ml_odds is not None:
                st.write(f"- ML Over probability: {ml_prob_pct:.1f}% ({ml_odds})")
                if ml_tier and ml_tier != "N/A":
                    st.write(f"- ML Confidence Tier: {ml_tier}")
            else:
                st.write("- ML Over probability: not available")

            unified_side = r.get("unified_side")
            unified_pick = r.get("unified_pick")
            if unified_pick:
                st.write(f"- Unified recommendation: {unified_pick}")

            kelly_half = r.get("kelly_half")
            kelly_full = r.get("kelly_full")
            if (
                unified_side in {"Over", "Under"}
                and kelly_half is not None
                and kelly_full is not None
                and kelly_full > 0
            ):
                st.write(
                    f"- Kelly (Unified): {kelly_half*100:.1f}% of bankroll (half), "
                    f"{kelly_full*100:.1f}% (full)"
                )

            st.write(f"- Historical recommended side: {bet_side}  (odds: {bet_odds})")




# =====================================================
# TAB 3: AI SLIP BUILDER (POWER / FLEX)
# =====================================================
with tab_slips:
    st.write("### AI Slip Builder (Unified Rec + ML)")

    if view_df.empty:
        st.info("No edges available to build slips from yet.")
    else:
        slips_df = view_df.copy()

        # Keep only props with a clear unified side and ML probabilities
        if "unified_side" in slips_df.columns:
            slips_df = slips_df[slips_df["unified_side"].isin(["Over", "Under"])]
        if "ml_prob_over" in slips_df.columns:
            slips_df = slips_df[slips_df["ml_prob_over"].notna()]
        if "ml_conf_tier" in slips_df.columns:
            slips_df = slips_df[slips_df["ml_conf_tier"].isin(["GREEN (High)", "YELLOW (Medium)"])]

        if slips_df.empty:
            st.info("No high-confidence props available for slip building after filters.")
        else:
            # Sort by model probability first, then by confidence
            sort_cols = []
            if "ml_prob_over" in slips_df.columns:
                sort_cols.append("ml_prob_over")
            if "confidence_pct" in slips_df.columns:
                sort_cols.append("confidence_pct")
            if sort_cols:
                slips_df = slips_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))

            def build_slip(df: pd.DataFrame, n_picks: int) -> list[dict]:
                picks = []
                used_players = set()
                used_teams = set()

                for _, row in df.iterrows():
                    player = row.get("player")
                    team = row.get("team")

                    if player in used_players:
                        continue

                    # Try to respect "different teams" as much as possible,
                    # especially for small slips like 2- or 3-picks.
                    if team and team in used_teams and len(used_teams) < n_picks:
                        continue

                    picks.append(row.to_dict())
                    used_players.add(player)
                    if team:
                        used_teams.add(team)

                    if len(picks) >= n_picks:
                        break

                return picks

            best_2 = build_slip(slips_df, 2)
            best_3 = build_slip(slips_df, 3)

            def render_slip(title: str, picks: list[dict]):
                st.subheader(title)
                if not picks:
                    st.caption("No suitable combination found with current filters.")
                    return
                cols_to_show = [
                    "player",
                    "team",
                    "opponent",
                    "market",
                    "line",
                    "unified_pick",
                    "ml_prob_pct",
                    "ml_conf_tier",
                ]
                df_show = pd.DataFrame(picks)
                cols_existing = [c for c in cols_to_show if c in df_show.columns]
                st.dataframe(df_show[cols_existing], use_container_width=True)

            render_slip("Best 2-Pick Power (AI)", best_2)
            render_slip("Best 3-Pick Flex (AI)", best_3)


# =====================================================
# TAB 4: WHAT-IF ALT LINES
# =====================================================
with tab_whatif:
    st.write("### What-If Alt Lines (Custom Player Lines)")

    if edges_df is None or edges_df.empty:
        st.info("No edges computed yet – use the main table first.")
    else:
        # -------------------------------
        # Player + market selection (sticky)
        # -------------------------------
        players_for_whatif = sorted(edges_df["player"].dropna().unique().tolist())

        if "whatif_player" not in st.session_state:
            st.session_state["whatif_player"] = ""
        if "whatif_market" not in st.session_state:
            st.session_state["whatif_market"] = ""
        if "whatif_center_line" not in st.session_state:
            st.session_state["whatif_center_line"] = None

        player_options = [""] + players_for_whatif
        if st.session_state["whatif_player"] not in player_options:
            st.session_state["whatif_player"] = ""

        selected_player_wi = st.selectbox(
            "Select player for what-if alt lines",
            player_options,
            index=player_options.index(st.session_state["whatif_player"]),
        )
        st.session_state["whatif_player"] = selected_player_wi

        if selected_player_wi:
            df_player = edges_df[edges_df["player"] == selected_player_wi]
            markets_for_player = sorted(df_player["market"].dropna().unique().tolist())

            market_options = [""] + markets_for_player
            if st.session_state["whatif_market"] not in market_options:
                st.session_state["whatif_market"] = ""

            selected_market_wi = st.selectbox(
                "Select market",
                market_options,
                index=market_options.index(st.session_state["whatif_market"]),
            )
            st.session_state["whatif_market"] = selected_market_wi

            if selected_market_wi:
                df_pm = df_player[df_player["market"] == selected_market_wi].copy()
                if df_pm.empty:
                    st.warning("No rows for this player/market in the current edges table.")
                else:
                    sort_col = edge_col if "edge_col" in globals() and edge_col in df_pm.columns else None
                    if sort_col is None:
                        sort_col = next((c for c in df_pm.columns if c.startswith("edge_last_")), None)
                    if sort_col is not None and sort_col in df_pm.columns:
                        df_pm = df_pm.sort_values(sort_col, ascending=False)
                    else:
                        df_pm = df_pm.sort_values("confidence_pct", ascending=False)
                    row_main = df_pm.iloc[0]

                    base_team = row_main.get("team") or row_main.get("team_abbrev") or ""
                    base_opp = row_main.get("opponent") or ""

                    st.markdown(
                        f"**{selected_player_wi}** – *{selected_market_wi}*  "
                        f"{'(' + base_team + ' vs ' + base_opp + ')' if base_team or base_opp else ''}"
                    )

                    # -------------------------------
                    # Pull fresh game log + series
                    # -------------------------------
                    pid_wi = row_main.get("player_id") or get_player_id(selected_player_wi)
                    gamelog_wi = get_player_gamelog(pid_wi)
                    if gamelog_wi.empty:
                        st.warning("No game log for this player; cannot compute what-if lines.")
                    else:
                        series_wi = get_market_series(gamelog_wi, selected_market_wi)
                        series_wi = series_wi.dropna()

                        if series_wi.empty:
                            st.warning("No stat data for this market.")
                        else:
                            n_games = min(len(series_wi), games_to_look_back)
                            last_n = series_wi.head(n_games)
                            avg_last_n_wi = float(last_n.mean()) if n_games > 0 else float("nan")

                            st.write(
                                f"Using last **{n_games}** games. "
                                f"Average: **{avg_last_n_wi:.2f}**."
                            )

                            # Default center line = board line if available, else recent avg
                            default_line = row_main.get("line")
                            try:
                                default_line = float(default_line)
                            except Exception:
                                default_line = float(round(avg_last_n_wi, 1)) if not math.isnan(avg_last_n_wi) else 0.0

                            if st.session_state["whatif_center_line"] is None:
                                st.session_state["whatif_center_line"] = default_line

                            center_line = st.number_input(
                                "Custom what-if line",
                                value=float(st.session_state["whatif_center_line"]),
                                step=0.5,
                            )
                            st.session_state["whatif_center_line"] = center_line

                            if n_games > 0:
                                hit_mask = last_n > center_line
                                push_mask = last_n == center_line
                                over_count = int(hit_mask.sum())
                                push_count = int(push_mask.sum())
                                under_count = int(n_games - over_count - push_count)

                                prob_over = over_count / n_games
                                prob_under = under_count / n_games

                                st.markdown(
                                    f"- Over {center_line}: **{over_count}/{n_games}** games "
                                    f"({prob_over*100:.1f}% )  \n"
                                    f"- Under {center_line}: **{under_count}/{n_games}** games "
                                    f"({prob_under*100:.1f}% )  \n"
                                    f"- Push: **{push_count}/{n_games}** games"
                                )

                                # Simple bar chart of Over / Under / Push vs the custom line
                                chart_data = pd.DataFrame(
                                    {
                                        "Result": ["Over", "Under", "Push"],
                                        "Count": [over_count, under_count, push_count],
                                    }
                                )
                                bar_chart = (
                                    alt.Chart(chart_data)
                                    .mark_bar()
                                    .encode(
                                        x=alt.X("Result:N", title="Result vs custom line"),
                                        y=alt.Y("Count:Q", title=f"Games (last {n_games})"),
                                        tooltip=["Result", "Count"],
                                    )
                                )
                                st.altair_chart(bar_chart, use_container_width=True)

        # -------------------------------
        # Alt-line (Goblin/Demon) predictions table
        # -------------------------------
        st.write("### Goblin / Demon Alt Lines (Predicted)")
        if "odds_type" not in edges_df.columns:
            st.info("No odds_type information available yet.")
        else:
            alt_df = edges_df[edges_df["odds_type"].fillna("standard") != "standard"].copy()
            if alt_df.empty:
                st.info("No Goblin / Demon alt lines in the current dataset.")
            else:
                # show key columns plus model predictions
                show_cols = [
                    c
                    for c in [
                        "player",
                        "team",
                        "opponent",
                        "market",
                        "line",
                        "odds_type",
                        "bet_side",
                        "unified_pick",
                        "predicted_score",
                        "season_avg",
                        f"avg_last_{games_to_look_back}",
                        f"over_rate_last_{games_to_look_back}",
                        f"edge_last_{games_to_look_back}",
                        "ml_prob_pct",
                        "ml_conf_tier",
                        "confidence_pct",
                    ]
                    if c in alt_df.columns
                ]
                if show_cols:
                    st.dataframe(
                        alt_df[show_cols].sort_values(
                            by=[c for c in [
                                "edge_pct",
                                "confidence_pct",
                            ] if c in alt_df.columns],
                            ascending=False,
                        )
                    )
                else:
                    st.dataframe(alt_df)

# =====================================================
# TAB 5: PLAYER DETAIL – GAME LOG & CHART
# =====================================================
with tab_player:
    st.write("### Player Detail – Recent Game Log & Splits")

    if edges_df is None or edges_df.empty:
        st.info("No props available – load PrizePicks or upload a CSV first.")
    else:
        # Player selector based on current edges
        players_pd = sorted(edges_df["player"].dropna().unique().tolist())
        if not players_pd:
            st.info("No players found in the current edges.")
        else:
            pd_col1, pd_col2 = st.columns([2, 2])
            with pd_col1:
                selected_player_pd = st.selectbox(
                    "Select player",
                    [""] + players_pd,
                    index=0,
                    key="player_detail_player",
                )
            markets_pd = []
            if selected_player_pd:
                markets_pd = sorted(
                    edges_df.loc[edges_df["player"] == selected_player_pd, "market"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            with pd_col2:
                selected_market_pd = st.selectbox(
                    "Market",
                    [""] + markets_pd,
                    index=0,
                    key="player_detail_market",
                )

            if selected_player_pd and selected_market_pd:
                # Take the "best" row for this player/market as reference
                df_pd = edges_df[
                    (edges_df["player"] == selected_player_pd)
                    & (edges_df["market"] == selected_market_pd)
                ].copy()

                if df_pd.empty:
                    st.warning("No rows found for that player/market in the current edges.")
                else:
                    df_pd = df_pd.sort_values(
                        by=["confidence_pct", rate_col, edge_col],
                        ascending=False,
                    )
                    row_ref = df_pd.iloc[0]

                    st.markdown(
                        f"**{selected_player_pd} – {row_ref.get('team', '')} vs {row_ref.get('opponent', '')}**  "
                        f"({selected_market_pd}, line: `{row_ref.get('line')}`)"
                    )

                    # Pull fresh game log and series
                    pid_pd = row_ref.get("player_id") or get_player_id(selected_player_pd)
                    gl_pd = get_player_gamelog(pid_pd)
                    if gl_pd.empty:
                        st.warning("No NBA gamelog data available for this player.")
                    else:
                        series_pd = get_market_series(gl_pd, selected_market_pd)
                        series_pd = series_pd.dropna()

                        if series_pd.empty:
                            st.warning("No stats found for this market in the gamelog.")
                        else:
                            n_show = min(len(gl_pd), 25)
                            gl_show = gl_pd.head(n_show).copy()
                            gl_show["STAT"] = series_pd.head(n_show).values

                            st.markdown(f"#### Last {n_show} games")
                            st.dataframe(
                                gl_show[["GAME_DATE", "MATCHUP", "MIN", "STAT"]],
                                use_container_width=True,
                            )

                            # Simple line chart of the stat vs date
                            chart_df = gl_show[["GAME_DATE", "STAT"]].sort_values(
                                "GAME_DATE", ascending=True
                            )
                            line_chart = (
                                alt.Chart(chart_df)
                                .mark_line(point=True)
                                .encode(
                                    x=alt.X("GAME_DATE:T", title="Game date"),
                                    y=alt.Y("STAT:Q", title=selected_market_pd),
                                    tooltip=["GAME_DATE:T", "STAT:Q"],
                                )
                            )
                            st.altair_chart(line_chart, use_container_width=True)

                            # Quick summary
                            st.markdown("#### Summary")
                            st.write(f"- Season mean for this market: **{series_pd.mean():.2f}**")
                            st.write(
                                f"- Last {games_to_look_back} mean: "
                                f"**{series_pd.head(games_to_look_back).mean():.2f}**"
                            )

# =====================================================
# TAB 6: GAME MONEYLINES & SPREADS (MODEL)
# =====================================================
with tab_games:
    st.write("### Game Moneylines & Spreads – Model View")

    if games_ml_df is None or games_ml_df.empty:
        st.info("No game-level model data yet – need enough props across games.")
    else:
        df_g = games_ml_df.copy()

        # Filter by minimum props per team
        df_g = df_g[df_g["props_count"] >= min_props_for_ml]
        if df_g.empty:
            st.info("No games meet the minimum props-per-team threshold yet.")
        else:
            # Highlight strong opinions by spread and/or win prob
            highlight_mask = (
                (df_g["win_prob_pct"] >= ml_winprob_highlight)
                | (df_g["win_prob_pct"] <= 100.0 - ml_winprob_highlight)
                | (df_g["model_spread"].abs() >= spread_highlight)
            )
            df_highlight = df_g[highlight_mask].copy()

            st.markdown("#### All modeled teams")
            st.dataframe(
                df_g[
                    [
                        "game_label",
                        "team",
                        "opponent",
                        "props_count",
                        "avg_edge",
                        "avg_confidence",
                        "model_spread",
                        "win_prob_pct",
                        "ml_odds",
                    ]
                ].sort_values(["game_label", "team"]),
                use_container_width=True,
            )

            st.markdown("#### Highlighted edges (big spread or strong win %)")
            if df_highlight.empty:
                st.caption("No particularly strong edges given the current sliders.")
            else:
                st.dataframe(
                    df_highlight[
                        [
                            "game_label",
                            "team",
                            "opponent",
                            "props_count",
                            "avg_edge",
                            "avg_confidence",
                            "model_spread",
                            "win_prob_pct",
                            "ml_odds",
                        ]
                    ]
                    .sort_values("win_prob_pct", ascending=False),
                    use_container_width=True,
                )

# =====================================================
# TAB 7: BET TRACKER (ML & LEARNING)
# =====================================================
with tab_bets:
    st.write("### Bet Tracker – Saved Bets & Results")

    bets = st.session_state.get("bet_tracker", [])
    if not bets:
        st.info("No bets tracked yet. Use **Track this bet** buttons on the Cards/Table views.")
    else:
        bets_df = pd.DataFrame(bets)

        # Ensure consistent columns
        required_cols = [
            "bet_category",
            "player",
            "team",
            "opponent",
            "market",
            "line",
            "bet_side",
            "bet_odds",
            "predicted_score",
            "confidence_pct",
            "game_time",
            "game_key",
            "model_spread",
            "win_prob_pct",
            "actual_stat",
            "result",
        ]
        for c in required_cols:
            if c not in bets_df.columns:
                bets_df[c] = None

        # Display most recent bets first
        bets_df = bets_df.reset_index().rename(columns={"index": "bet_id"})
        bets_df = bets_df.sort_values("bet_id", ascending=False)

        st.markdown("#### Tracked Bets")
        st.dataframe(
            bets_df[
                [
                    "bet_id",
                    "bet_category",
                    "player",
                    "team",
                    "opponent",
                    "market",
                    "line",
                    "bet_side",
                    "bet_odds",
                    "predicted_score",
                    "confidence_pct",
                    "game_time",
                    "actual_stat",
                    "result",
                ]
            ],
            use_container_width=True,
        )

        st.markdown("#### Update Result for a Bet")

        bet_ids = bets_df["bet_id"].tolist()
        selected_bet_id = st.selectbox(
            "Select bet to update",
            [""] + bet_ids,
            format_func=lambda x: "" if x == "" else f"Bet #{x}",
            key="bet_tracker_select_id",
        )

        if selected_bet_id != "":
            sel_row = bets_df[bets_df["bet_id"] == selected_bet_id].iloc[0]
            st.write(
                f"Updating **{sel_row['player']} – {sel_row['market']} {sel_row['line']} ({sel_row['bet_side']})**"
            )

            col_bt1, col_bt2 = st.columns(2)
            with col_bt1:
                new_actual = st.number_input(
                    "Final stat (actual result)",
                    value=float(sel_row["actual_stat"])
                    if pd.notna(sel_row["actual_stat"])
                    else 0.0,
                    step=0.5,
                    key="bet_tracker_actual_input",
                )
            with col_bt2:
                new_result = st.selectbox(
                    "Result",
                    ["Pending", "Win", "Loss", "Push"],
                    index=["Pending", "Win", "Loss", "Push"].index(
                        sel_row["result"] if sel_row["result"] in ["Pending", "Win", "Loss", "Push"] else "Pending"
                    ),
                    key="bet_tracker_result_select",
                )

            if st.button("Save result update", key="bet_tracker_save_button"):
                # Apply update back into session_state list
                idx_in_state = int(selected_bet_id)
                if 0 <= idx_in_state < len(st.session_state["bet_tracker"]):
                    st.session_state["bet_tracker"][idx_in_state]["actual_stat"] = new_actual
                    st.session_state["bet_tracker"][idx_in_state]["result"] = new_result
                    save_bets_to_disk(st.session_state["bet_tracker"])
                    st.success("Bet updated and saved.")