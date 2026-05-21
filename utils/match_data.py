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

    logger.info("Loaded %d match records from %s", len(df), folder)
    return df


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    folder = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ipl"
    df = load_match_info(folder)

    print(f"\nLoaded {len(df)} matches from '{folder}'")
    print(df.head(10).to_string(index=False))
