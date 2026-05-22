"""
utils/normalizer.py
------------------------
Standardizes IPL team names across historical seasons so that renamed
franchises map to their current/canonical names.
"""

from __future__ import annotations

# Add new mappings here as franchises rename.
# Keys are historical names; values are the current canonical names.
_TEAM_NAME_MAP: dict[str, str] = {
    "Delhi Daredevils":          "Delhi Capitals",
    "Kings XI Punjab":           "Punjab Kings",
    "Rising Pune Supergiant":    "Rising Pune Supergiants",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}


def normalize_team_name(team_name: str | None) -> str | None:
    """Return the canonical name for a given IPL team name.

    Parameters
    ----------
    team_name:
        Raw team name string, as it may appear in match data across
        different seasons. ``None`` is passed through unchanged.

    Returns
    -------
    str | None
        Normalized team name if a mapping exists, otherwise the original
        name (stripped of surrounding whitespace). Returns ``None`` if the
        input is ``None``.
    """
    if team_name is None:
        return None

    stripped = team_name.strip()
    return _TEAM_NAME_MAP.get(stripped, stripped)


if __name__ == "__main__":
    test_cases = [
        "Delhi Daredevils",
        "Kings XI Punjab",
        "Rising Pune Supergiant",
        "Royal Challengers Bangalore",
        "Mumbai Indians",          # no mapping — returned as-is
        "  Chennai Super Kings  ", # whitespace stripped
        None,                      # None passed through
    ]

    for name in test_cases:
        result = normalize_team_name(name)
        print(f"  {name!r:42} -> {result!r}")
