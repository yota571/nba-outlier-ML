"""
resolve_player_ids.py  (Dynamic Player Resolver 2.0)

Goal:
- Fill missing `player_id` values in prop_training_data.csv using the `player`
  name column and nba_api.
- Uses multiple strategies:
    1. Exact normalized full-name match
    2. Last-name + first-initial match
    3. Fuzzy matching with difflib

Run:
    python resolve_player_ids.py

Then you can run:
    python backfill_labels.py
    python train_over_model_v2.py
"""

import os
import re
import unicodedata
from datetime import datetime

import pandas as pd
from nba_api.stats.static import players as nba_players
from difflib import SequenceMatcher


CSV_PATH = "prop_training_data.csv"
BACKUP_PATH = "prop_training_data_backup_before_resolve_ids.csv"

PLAYER_ID_COL = "player_id"
PLAYER_NAME_COL = "player"


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


# -------------------------------------------------------
# Build player lookup structures
# -------------------------------------------------------
def build_player_indices():
    """
    Returns:
        full_map: normalized_full_name -> [player_id,...]
        last_map: last_name -> [player_id,...]
        fi_last_map: "f last" (first initial + last) -> [player_id,...]
        id_to_name: player_id -> official full_name
        all_norm_names: list of normalized full names (for fuzzy)
    """
    plist = nba_players.get_players()
    full_map = {}
    last_map = {}
    fi_last_map = {}
    id_to_name = {}

    for p in plist:
        full_name = str(p.get("full_name", "")).strip()
        pid = p.get("id")
        if not full_name or pid is None:
            continue

        id_to_name[pid] = full_name

        norm_full = normalize_name(full_name)
        full_map.setdefault(norm_full, []).append(pid)

        parts = norm_full.split()
        if not parts:
            continue
        first = parts[0]
        last = parts[-1]
        last_map.setdefault(last, []).append(pid)

        fi_last = f"{first[0]} {last}"
        fi_last_map.setdefault(fi_last, []).append(pid)

    all_norm_names = list(full_map.keys())
    log(f"Built indices for {len(id_to_name)} NBA players.")
    return full_map, last_map, fi_last_map, id_to_name, all_norm_names


def fuzzy_best_match(name_norm: str, all_norm_names: list[str], cutoff: float = 0.88):
    """
    Return the best fuzzy match normalized name if score >= cutoff, else None.
    """
    best_name = None
    best_score = 0.0

    for official in all_norm_names:
        score = SequenceMatcher(None, name_norm, official).ratio()
        if score > best_score:
            best_score = score
            best_name = official

    if best_score >= cutoff:
        return best_name, best_score
    return None, best_score


def resolve_single_name(
    raw_name: str,
    full_map,
    last_map,
    fi_last_map,
    all_norm_names,
    id_to_name,
):
    """
    Try to resolve a single player name to a unique player_id.

    Returns:
        (player_id, strategy_str) or (None, reason_str)
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
    ids = full_map.get(norm)
    if ids:
        if len(ids) == 1:
            return ids[0], "exact_full"
        else:
            return None, "full_name_ambiguous"

    # 2) Unique last name
    last_ids = last_map.get(last, [])
    if len(last_ids) == 1:
        return last_ids[0], "unique_last_name"

    # 3) First-initial + last-name match
    if fi_last:
        fi_ids = fi_last_map.get(fi_last, [])
        if len(fi_ids) == 1:
            return fi_ids[0], "fi_last"

    # 4) Fuzzy match on full name
    fuzzy_name, score = fuzzy_best_match(norm, all_norm_names, cutoff=0.90)
    if fuzzy_name:
        fuzzy_ids = full_map.get(fuzzy_name, [])
        if len(fuzzy_ids) == 1:
            return fuzzy_ids[0], f"fuzzy_full({score:.2f})"
        else:
            return None, "fuzzy_ambiguous"

    return None, "no_match"


# -------------------------------------------------------
# Main resolve logic
# -------------------------------------------------------
def main():
    log("Starting resolve_player_ids.py (Dynamic Player Resolver 2.0)...")

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

    if PLAYER_ID_COL not in df.columns:
        log(f"ERROR: {PLAYER_ID_COL} column not found in CSV.")
        return
    if PLAYER_NAME_COL not in df.columns:
        log(f"ERROR: {PLAYER_NAME_COL} column not found in CSV.")
        return

    # Identify rows needing resolution
    missing_mask = df[PLAYER_ID_COL].isna() & df[PLAYER_NAME_COL].notna()
    total_missing = missing_mask.sum()
    log(f"Rows with missing player_id but a name: {total_missing}")

    if total_missing == 0:
        log("No missing player_id rows to resolve. Done.")
        return

    full_map, last_map, fi_last_map, id_to_name, all_norm_names = build_player_indices()

    resolved = 0
    unresolved = 0

    strategy_counts = {}

    for idx in df[missing_mask].index:
        raw_name = df.at[idx, PLAYER_NAME_COL]
        pid, strategy = resolve_single_name(
            raw_name, full_map, last_map, fi_last_map, all_norm_names, id_to_name
        )

        if pid is not None:
            df.at[idx, PLAYER_ID_COL] = pid
            resolved += 1
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        else:
            unresolved += 1
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    log("Resolution summary:")
    log(f"  Resolved rows:   {resolved}")
    log(f"  Unresolved rows: {unresolved}")
    for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        log(f"  {strat}: {count}")

    # Save back
    df.to_csv(CSV_PATH, index=False)
    log(f"Saved updated CSV to {CSV_PATH}")
    log("Done.")


if __name__ == "__main__":
    main()
