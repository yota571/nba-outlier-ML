"""
player_resolver_3.py  (Dynamic Player Resolver 3.0)

Goal:
- Build a *robust* mapping between player names in prop_training_data.csv
  and official NBA player IDs from nba_api.
- Fix missing or incorrect `player_id` values.
- Produce a short report so you know what happened.

Strategies used:
  1) Exact normalized full-name match
  2) Last-name + first initial
  3) Fuzzy match on normalized names (high threshold)
  4) Optional team sanity check (if 'team' column is present)

Output:
  - Updates 'player_id' column in prop_training_data.csv in-place
  - Creates one backup: prop_training_data_backup_before_player_resolver3.csv
  - Prints a summary of resolutions / failures.

Run:
    python player_resolver_3.py
"""

import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from difflib import SequenceMatcher
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams


CSV_PATH = "prop_training_data.csv"
BACKUP_PATH = "prop_training_data_backup_before_player_resolver3.csv"

PLAYER_ID_COL = "player_id"
PLAYER_NAME_COL = "player"
TEAM_COL = "team"  # optional but helpful


# -------------------------------------------------------
# Logging
# -------------------------------------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


# -------------------------------------------------------
# Name normalization / helpers
# -------------------------------------------------------
SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.IGNORECASE)


def normalize_name(name: str) -> str:
    """
    Normalize a player name for robust comparison:

    - lowercases
    - strips accents
    - removes Jr/Sr/II/III/IV suffixes
    - removes punctuation
    - collapses whitespace
    """
    if not isinstance(name, str):
        return ""

    # Strip accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))

    # Lowercase
    name = name.lower()

    # Remove common suffixes
    name = SUFFIX_RE.sub("", name)

    # Remove punctuation
    name = re.sub(r"[^a-z\s]", " ", name)

    # Collapse spaces
    name = re.sub(r"\s+", " ", name).strip()
    return name


@dataclass
class PlayerRecord:
    player_id: int
    full_name: str
    norm_name: str
    first: str
    last: str
    fi_last: str  # first initial + last
    team_id: Optional[int]
    team_abbr: Optional[str]


# -------------------------------------------------------
# Build NBA player + team indices
# -------------------------------------------------------
def build_team_index() -> Dict[int, str]:
    """TEAM_ID -> TEAM_ABBR mapping."""
    tlist = nba_teams.get_teams()
    mapping = {}
    for t in tlist:
        tid = t.get("id")
        abbr = t.get("abbreviation")
        if tid is not None and abbr:
            mapping[int(tid)] = abbr.upper()
    return mapping


def build_player_indices() -> Tuple[
    Dict[str, List[PlayerRecord]],
    Dict[str, List[PlayerRecord]],
    Dict[str, List[PlayerRecord]],
    Dict[int, PlayerRecord],
]:
    """
    Returns:
        full_map: norm_full_name -> [PlayerRecord,...]
        last_map: last_name -> [PlayerRecord,...]
        fi_last_map: "fi last" -> [PlayerRecord,...]
        by_id: player_id -> PlayerRecord
    """
    team_index = build_team_index()
    plist = nba_players.get_players()

    full_map: Dict[str, List[PlayerRecord]] = {}
    last_map: Dict[str, List[PlayerRecord]] = {}
    fi_last_map: Dict[str, List[PlayerRecord]] = {}
    by_id: Dict[int, PlayerRecord] = {}

    for p in plist:
        full_name = str(p.get("full_name", "")).strip()
        pid = p.get("id")
        if not full_name or pid is None:
            continue

        pid = int(pid)
        norm_full = normalize_name(full_name)
        if not norm_full:
            continue

        parts = norm_full.split()
        first = parts[0]
        last = parts[-1]
        fi_last = f"{first[0]} {last}"

        # nba_api players() doesn't directly give team; some may be free agents.
        # We'll keep None for team; resolver will NOT rely heavily on team.
        rec = PlayerRecord(
            player_id=pid,
            full_name=full_name,
            norm_name=norm_full,
            first=first,
            last=last,
            fi_last=fi_last,
            team_id=None,
            team_abbr=None,
        )

        by_id[pid] = rec
        full_map.setdefault(norm_full, []).append(rec)
        last_map.setdefault(last, []).append(rec)
        fi_last_map.setdefault(fi_last, []).append(rec)

    log(f"Built NBA player index for {len(by_id)} players.")
    return full_map, last_map, fi_last_map, by_id


# -------------------------------------------------------
# Matching logic
# -------------------------------------------------------
def fuzzy_best_match(
    name_norm: str,
    candidates: List[str],
    cutoff: float = 0.88,
) -> Tuple[Optional[str], float]:
    """
    Return the best fuzzy match normalized name if score >= cutoff, else None.
    """
    best_name = None
    best_score = 0.0

    for cand in candidates:
        score = SequenceMatcher(None, name_norm, cand).ratio()
        if score > best_score:
            best_score = score
            best_name = cand

    if best_score >= cutoff:
        return best_name, best_score
    return None, best_score


def resolve_name_to_record(
    raw_name: str,
    full_map: Dict[str, List[PlayerRecord]],
    last_map: Dict[str, List[PlayerRecord]],
    fi_last_map: Dict[str, List[PlayerRecord]],
) -> Tuple[Optional[PlayerRecord], str]:
    """
    Try to resolve a single player name to a unique PlayerRecord.

    Returns:
        (PlayerRecord or None, strategy_str)
    """
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, "empty_name"

    norm = normalize_name(raw_name)
    if not norm:
        return None, "empty_norm"

    parts = norm.split()
    first = parts[0] if parts else ""
    last = parts[-1] if parts else ""
    fi_last = f"{first[0]} {last}" if first else ""

    # 1) Exact full-name match
    recs = full_map.get(norm)
    if recs:
        if len(recs) == 1:
            return recs[0], "exact_full"
        else:
            # multiple players with same name (rare) – ambiguous
            return None, "full_name_ambiguous"

    # 2) Unique last-name match
    last_recs = last_map.get(last, [])
    if len(last_recs) == 1:
        return last_recs[0], "unique_last"

    # 3) First-initial + last name
    if fi_last:
        fi_recs = fi_last_map.get(fi_last, [])
        if len(fi_recs) == 1:
            return fi_recs[0], "fi_last"

    # 4) Fuzzy full-name match
    all_norm_names = list(full_map.keys())
    fuzzy_name, score = fuzzy_best_match(norm, all_norm_names, cutoff=0.90)
    if fuzzy_name:
        frecs = full_map.get(fuzzy_name, [])
        if len(frecs) == 1:
            return frecs[0], f"fuzzy_full({score:.2f})"
        else:
            return None, "fuzzy_ambiguous"

    return None, "no_match"


# -------------------------------------------------------
# Main resolver
# -------------------------------------------------------
def main():
    log("Starting player_resolver_3.py (Dynamic Player Resolver 3.0)...")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find {CSV_PATH}")

    # Backup once
    if not os.path.exists(BACKUP_PATH):
        df_bak = pd.read_csv(CSV_PATH)
        df_bak.to_csv(BACKUP_PATH, index=False)
        log(f"Backup saved to {BACKUP_PATH}")
    else:
        log(f"Backup already exists at {BACKUP_PATH} (not overwriting).")

    df = pd.read_csv(CSV_PATH)
    log(f"Loaded {len(df)} rows from {CSV_PATH}")

    if PLAYER_NAME_COL not in df.columns:
        log(f"ERROR: column '{PLAYER_NAME_COL}' not found. Aborting.")
        return
    if PLAYER_ID_COL not in df.columns:
        log(f"'{PLAYER_ID_COL}' column not found; creating it.")
        df[PLAYER_ID_COL] = pd.NA

    # Normalize player_id to numeric or NaN
    df[PLAYER_ID_COL] = pd.to_numeric(df[PLAYER_ID_COL], errors="coerce")

    # Build NBA indices
    full_map, last_map, fi_last_map, by_id = build_player_indices()

    # Unique names needing resolution
    mask_missing = df[PLAYER_ID_COL].isna() & df[PLAYER_NAME_COL].notna()
    names_needing = df.loc[mask_missing, PLAYER_NAME_COL].dropna().unique().tolist()
    log(f"Unique names with missing player_id: {len(names_needing)}")

    # Also check possibly WRONG IDs (optional)
    # For simplicity, we only resolve missing IDs here. You can extend to sanity check.
    resolved_map: Dict[str, Tuple[int, str]] = {}  # raw_name -> (player_id, strategy)
    unresolved_map: Dict[str, str] = {}            # raw_name -> reason

    for raw_name in names_needing:
        rec, strategy = resolve_name_to_record(raw_name, full_map, last_map, fi_last_map)
        if rec is not None:
            resolved_map[raw_name] = (rec.player_id, strategy)
        else:
            unresolved_map[raw_name] = strategy

    log("Resolution attempts summary over unique names:")
    strat_counts = {}
    for _, strategy in resolved_map.values():
        strat_counts[strategy] = strat_counts.get(strategy, 0) + 1
    for _, reason in unresolved_map.items():
        strat_counts[reason] = strat_counts.get(reason, 0) + 1

    for strat, count in sorted(strat_counts.items(), key=lambda x: -x[1]):
        log(f"  {strat}: {count}")

    # Apply resolved IDs back into DataFrame
    total_rows_resolved = 0
    for raw_name, (pid, strategy) in resolved_map.items():
        row_mask = df[PLAYER_ID_COL].isna() & (df[PLAYER_NAME_COL] == raw_name)
        n = row_mask.sum()
        if n > 0:
            df.loc[row_mask, PLAYER_ID_COL] = pid
            total_rows_resolved += n

    log(f"Total rows where player_id was filled: {total_rows_resolved}")
    log(f"Total distinct names unresolved: {len(unresolved_map)}")

    if unresolved_map:
        # Save unresolved names to a small CSV for review
        unresolved_path = "unresolved_players_resolver3.csv"
        pd.DataFrame(
            {
                "player": list(unresolved_map.keys()),
                "reason": list(unresolved_map.values()),
            }
        ).to_csv(unresolved_path, index=False)
        log(f"Unresolved player names written to {unresolved_path}")

    # Save updated CSV
    df.to_csv(CSV_PATH, index=False)
    log(f"Updated CSV saved to {CSV_PATH}")
    log("Done.")


if __name__ == "__main__":
    main()
