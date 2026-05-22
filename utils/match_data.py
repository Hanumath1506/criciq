"""
utils/match_data.py
-------------------
Loads match-level metadata from Cricsheet-format *_info.csv files.

Each info file is a key-value store (not columnar), structured like:
    info,team,Sunrisers Hyderabad
    info,toss_winner,Royal Challengers Bangalore
    ...
This module parses those rows into one flat record per match.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

from utils.normalizer import normalize_team_name

logger = logging.getLogger(__name__)

_WANTED_FIELDS = {
    "venue", "toss_winner", "toss_decision", "season", "winner",
}


def _parse_info_file(path: Path) -> dict | None:
    """Parse a single *_info.csv file into a flat dict.

    Returns None (and logs a warning) if the file cannot be read.
    """
    record: dict = {
        "match_id": path.stem.replace("_info", ""),
        "team1": None,
        "team2": None,
        "venue": None,
        "toss_winner": None,
        "toss_decision": None,
        "season": None,
        "winner": None,
    }
    teams: list[str] = []

    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                # All metadata rows start with "info" in column 0
                if not row or row[0] != "info" or len(row) < 3:
                    continue

                key, value = row[1].strip(), row[2].strip()

                if key == "team":
                    teams.append(value)
                elif key in _WANTED_FIELDS:
                    record[key] = value

    except (OSError, csv.Error) as exc:
        logger.warning("Skipping %s — could not read file: %s", path.name, exc)
        return None

    if len(teams) >= 1:
        record["team1"] = teams[0]
    if len(teams) >= 2:
        record["team2"] = teams[1]

    return record


def load_match_info(folder_path: str | Path) -> pd.DataFrame:
    """Read all *_info.csv files in *folder_path* and return a DataFrame.

    Each row in the returned DataFrame represents one match with columns:
        match_id, team1, team2, venue, toss_winner, toss_decision, season, winner

    Parameters
    ----------
    folder_path:
        Directory containing Cricsheet-format *_info.csv files.

    Returns
    -------
    pd.DataFrame
        One row per match. Missing fields are filled with ``pd.NA``.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    info_files = sorted(folder.glob("*_info.csv"))
    if not info_files:
        logger.warning("No *_info.csv files found in %s", folder)
        return pd.DataFrame(columns=[
            "match_id", "team1", "team2", "venue",
            "toss_winner", "toss_decision", "season", "winner",
        ])

    records = []
    for path in info_files:
        record = _parse_info_file(path)
        if record is not None:
            records.append(record)

    df = pd.DataFrame(records).fillna(value=pd.NA)

    column_order = [
        "match_id", "team1", "team2", "venue",
        "toss_winner", "toss_decision", "season", "winner",
    ]
    # Keep only known columns — extra keys from future format versions are dropped
    df = df[[c for c in column_order if c in df.columns]]

    for col in ("team1", "team2", "toss_winner", "winner"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: normalize_team_name(x) if pd.notna(x) else x)

    logger.info("Loaded %d match records from %s", len(df), folder)
    return df


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add historical win-rate features to a match DataFrame without data leakage.

    For each match, only matches that occurred **before** the current match are
    used to compute statistics.  This prevents the model from "seeing the future"
    during training.

    Two columns are appended:

    team1_venue_win_rate
        Fraction of matches won by team1 at the given venue, considering only
        prior matches.  Default 0.5 when no history exists.

    head_to_head_win_rate
        Fraction of matches won by team1 against team2, considering only prior
        matches between those two teams.  Default 0.5 when no history exists.

    Parameters
    ----------
    df:
        DataFrame with at least these columns:
        ``season``, ``team1``, ``team2``, ``venue``, ``winner``.
        A ``season`` column is used as a proxy for chronological order when a
        proper date column is absent.

    Returns
    -------
    pd.DataFrame
        A *new* DataFrame (the original is never modified) with two extra columns.
    """
    required = {"season", "team1", "team2", "venue", "winner"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Work on a copy so the caller's dataframe is never mutated.
    df = df.copy()

    # ── 1. Sort chronologically ───────────────────────────────────────────────
    # We use season as the ordering key (e.g. "2008", "2008/09", "2023").
    # Converting to str makes mixed season formats sort lexicographically,
    # which is good enough for IPL data where seasons don't overlap.
    df = df.sort_values("season", kind="stable").reset_index(drop=True)

    # ── 2. Running-statistics accumulators ───────────────────────────────────
    # Each accumulator maps a key → [wins, total_matches].
    # We update them *after* recording features for the current row so that
    # the current match is never included in its own feature value.

    # venue_stats[(team, venue)] = [wins, total]
    venue_stats: dict[tuple, list[int, int]] = {}

    # h2h_stats[(team1, team2)] = [wins_by_team1, total]
    # We always store the pair in sorted order to avoid duplicate keys, then
    # reconstruct the perspective for team1 at look-up time.
    h2h_stats: dict[tuple, list[int, int]] = {}

    venue_win_rates: list[float] = []
    h2h_win_rates: list[float] = []

    for _, row in df.iterrows():
        team1  = row["team1"]
        team2  = row["team2"]
        venue  = row["venue"]
        winner = row["winner"]

        # ── 2a. Compute features using history accumulated SO FAR ─────────────

        # --- team1_venue_win_rate ---
        vkey = (team1, venue)
        if vkey in venue_stats and venue_stats[vkey][1] > 0:
            wins, total = venue_stats[vkey]
            venue_rate = wins / total
        else:
            venue_rate = 0.5  # no history → neutral prior

        # --- head_to_head_win_rate ---
        # Canonical key: always sort the two team names so (A, B) == (B, A).
        canonical = tuple(sorted([team1, team2]))
        if canonical in h2h_stats and h2h_stats[canonical][1] > 0:
            wins_canonical, total_h2h = h2h_stats[canonical]
            # wins_canonical counts wins by canonical[0]; adjust if team1 ≠ canonical[0]
            if team1 == canonical[0]:
                h2h_rate = wins_canonical / total_h2h
            else:
                h2h_rate = (total_h2h - wins_canonical) / total_h2h
        else:
            h2h_rate = 0.5  # no history → neutral prior

        venue_win_rates.append(venue_rate)
        h2h_win_rates.append(h2h_rate)

        # ── 2b. Update accumulators with the CURRENT match result ─────────────
        # This ensures future rows can see this match, but the current row cannot.

        # Update venue stats for team1
        if vkey not in venue_stats:
            venue_stats[vkey] = [0, 0]
        venue_stats[vkey][1] += 1
        if pd.notna(winner) and winner == team1:
            venue_stats[vkey][0] += 1

        # Update head-to-head stats (stored under canonical key)
        if canonical not in h2h_stats:
            h2h_stats[canonical] = [0, 0]
        h2h_stats[canonical][1] += 1
        # Count a win only when the canonical[0] team wins
        if pd.notna(winner) and winner == canonical[0]:
            h2h_stats[canonical][0] += 1

    df["team1_venue_win_rate"] = venue_win_rates
    df["head_to_head_win_rate"] = h2h_win_rates

    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # ── Demo: add_historical_features ────────────────────────────────────────
    print("=" * 60)
    print("Demo: add_historical_features()")
    print("=" * 60)

    sample_data = pd.DataFrame([
        # season  team1   team2   venue               winner
        {"season": "2020", "team1": "MI",  "team2": "CSK", "venue": "Wankhede",   "winner": "MI"},
        {"season": "2020", "team1": "CSK", "team2": "MI",  "venue": "Chepauk",    "winner": "CSK"},
        {"season": "2021", "team1": "MI",  "team2": "CSK", "venue": "Wankhede",   "winner": "CSK"},
        {"season": "2021", "team1": "RCB", "team2": "MI",  "venue": "Chinnaswamy","winner": "RCB"},
        {"season": "2022", "team1": "MI",  "team2": "CSK", "venue": "Wankhede",   "winner": "MI"},
        {"season": "2022", "team1": "CSK", "team2": "RCB", "venue": "Chepauk",    "winner": "CSK"},
        {"season": "2023", "team1": "MI",  "team2": "CSK", "venue": "Wankhede",   "winner": "MI"},
    ])

    enriched = add_historical_features(sample_data)

    display_cols = [
        "season", "team1", "team2", "venue", "winner",
        "team1_venue_win_rate", "head_to_head_win_rate",
    ]
    print(enriched[display_cols].to_string(index=False))
    print()
    print("Notes:")
    print("  • First MI vs CSK match → both rates = 0.5 (no prior history)")
    print("  • Subsequent matches show rates computed from earlier matches only")

    # ── Original __main__: load real data if a folder is provided ────────────
    if len(sys.argv) > 1:
        print("\n" + "=" * 60)
        folder = sys.argv[1]
        df = load_match_info(folder)
        print(f"\nLoaded {len(df)} matches from '{folder}'")
        print(df.head(10).to_string(index=False))
