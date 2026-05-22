"""
models/predictor.py
-------------------
Trains a Random Forest classifier to predict cricket match winners,
and provides a predict_winner helper for single-match inference.

Typical usage
-------------
    from utils.match_data import load_match_info
    from models.predictor import train_model, predict_winner

    df    = load_match_info("data/raw/ipl")
    model = train_model(df)

    result = predict_winner(model, {
        "venue": "Wankhede Stadium",
        "team1": "Mumbai Indians",
        "team2": "Chennai Super Kings",
        "toss_winner": "Mumbai Indians",
        "toss_decision": "bat",
        "season": "2023",
    })
    print(result)  # {"predicted_winner": "Mumbai Indians", "probability": 61.4}
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from utils.match_data import add_historical_features

logger = logging.getLogger(__name__)

# Categorical columns — one-hot encoded
CAT_COLS = ["venue", "team1", "team2", "toss_winner", "toss_decision", "season"]

# Numeric columns — passed through as-is (no encoding needed)
NUM_COLS = ["team1_venue_win_rate", "head_to_head_win_rate"]

# Full feature list (order matters for ColumnTransformer)
FEATURE_COLS = CAT_COLS + NUM_COLS
TARGET_COL = "winner"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame) -> Pipeline:
    """Train a Random Forest classifier on match-level data.

    Parameters
    ----------
    df:
        DataFrame produced by ``load_match_info``.  Must contain the columns
        in ``FEATURE_COLS`` plus ``winner``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline ready for ``predict`` / ``predict_proba`` calls.

    Raises
    ------
    ValueError
        If required columns are missing or if too few samples remain after
        dropping rows with a missing target.
    """
    _validate_columns(df, CAT_COLS + [TARGET_COL])

    # Drop matches with no recorded winner (e.g. abandoned games)
    df_clean = df.dropna(subset=[TARGET_COL]).copy()

    if len(df_clean) < 10:
        raise ValueError(
            f"Only {len(df_clean)} rows have a winner value — not enough to train."
        )

    # Coerce season to string so it's treated as a categorical label, not a number
    df_clean["season"] = df_clean["season"].astype(str)

    # Compute historical win-rate features (leakage-free: each row only sees
    # matches that came before it chronologically).
    df_clean = add_historical_features(df_clean)

    X = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print(f"\nModel trained on {len(X_train)} matches, evaluated on {len(X_test)}.")
    print(f"Accuracy: {accuracy:.2%}\n")
    print("Classification report (top classes shown):")
    # zero_division=0 silences warnings for classes with no predicted samples
    print(classification_report(y_test, y_pred, zero_division=0))

    logger.info("Training complete — accuracy %.2f%%", accuracy * 100)
    return pipeline


def _build_pipeline() -> Pipeline:
    """Construct the sklearn Pipeline with preprocessing and classifier."""
    # handle_unknown="ignore" makes the encoder emit all-zero rows for
    # team/venue names it has never seen, so predict_winner won't crash on
    # new stadiums or expansion franchises.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder,        CAT_COLS),  # OHE for categorical features
            ("num", "passthrough",  NUM_COLS),  # keep numeric features as-is
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,        # grow full trees; forest depth controls variance
        min_samples_leaf=2,    # avoid overfitting on very small leaves
        random_state=42,
        n_jobs=-1,             # use all available CPU cores
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_winner(model: Pipeline, match_details: dict) -> dict:
    """Predict the winner of a single match.

    Parameters
    ----------
    model:
        A fitted pipeline returned by ``train_model``.
    match_details:
        Dictionary with keys: venue, team1, team2, toss_winner,
        toss_decision, season.
        The numeric keys ``team1_venue_win_rate`` and ``head_to_head_win_rate``
        are optional and default to 0.5 (neutral prior) when absent.

    Returns
    -------
    dict
        ``{"predicted_winner": str, "probability": float}``
        where ``probability`` is the confidence for the predicted class
        as a percentage rounded to 2 decimal places.

    Raises
    ------
    ValueError
        If any required categorical key is missing from ``match_details``.
    """
    _validate_keys(match_details, CAT_COLS)

    # Build a single-row DataFrame with columns in the same order as training.
    # Categorical cols are coerced to str; numeric cols stay as float.
    row: dict = {col: [str(match_details[col])] for col in CAT_COLS}
    for col in NUM_COLS:
        row[col] = [float(match_details.get(col, 0.5))]
    X = pd.DataFrame(row)

    predicted_class: str = model.predict(X)[0]

    # predict_proba returns one probability per class; grab the winner's confidence
    classes = model.classes_
    proba_row = model.predict_proba(X)[0]
    winner_idx = list(classes).index(predicted_class)
    confidence = round(float(proba_row[winner_idx]) * 100, 2)

    return {
        "predicted_winner": predicted_class,
        "probability": confidence,
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def _validate_keys(d: dict, required: list[str]) -> None:
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(
            f"match_details is missing required keys: {missing}\n"
            f"Expected keys: {required}"
        )


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Try loading real data if available, otherwise use a toy dataset
    try:
        from utils.match_data import load_match_info
        folder = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ipl"
        df = load_match_info(folder)
        print(f"Loaded {len(df)} real matches from '{folder}'.")
    except Exception as exc:
        print(f"Could not load real data ({exc}). Using toy dataset instead.")
        df = pd.DataFrame([
            {"venue": "Stadium A", "team1": "Team 1", "team2": "Team 2",
             "toss_winner": "Team 1", "toss_decision": "bat",   "season": "2022", "winner": "Team 1"},
            {"venue": "Stadium B", "team1": "Team 2", "team2": "Team 3",
             "toss_winner": "Team 2", "toss_decision": "field", "season": "2022", "winner": "Team 3"},
            {"venue": "Stadium A", "team1": "Team 1", "team2": "Team 3",
             "toss_winner": "Team 3", "toss_decision": "bat",   "season": "2023", "winner": "Team 1"},
        ] * 20)

    model = train_model(df)

    sample = {
        "venue":                   df["venue"].iloc[0],
        "team1":                   df["team1"].iloc[0],
        "team2":                   df["team2"].iloc[0],
        "toss_winner":             df["toss_winner"].iloc[0],
        "toss_decision":           df["toss_decision"].iloc[0],
        "season":                  str(df["season"].iloc[0]),
        # Numeric features — omit to use the neutral 0.5 default, or supply
        # a pre-computed value from add_historical_features for real inference.
        "team1_venue_win_rate":    0.5,
        "head_to_head_win_rate":   0.5,
    }

    result = predict_winner(model, sample)
    print(f"\nSample prediction:")
    print(f"  Match  : {sample['team1']} vs {sample['team2']} at {sample['venue']}")
    print(f"  Toss   : {sample['toss_winner']} chose to {sample['toss_decision']}")
    print(f"  Winner : {result['predicted_winner']} ({result['probability']}% confidence)")
