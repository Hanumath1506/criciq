"""
pages/3_Match_Predictor.py
--------------------------
Streamlit page that lets users predict the winner of an IPL match
using a Random Forest model trained on historical match data.
"""

import os
import streamlit as st

from utils.match_data import load_match_info
from models.predictor import train_model, predict_winner

DATA_PATH = os.path.join("data", "raw", "ipl")

# ---------------------------------------------------------------------------
# Cached helpers — these run once and stay in memory across reruns
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(path: str):
    return load_match_info(path)


@st.cache_resource
def get_model(path: str):
    """Train and cache the model. st.cache_resource avoids re-pickling the
    Pipeline on every rerun (unlike cache_data, which serialises the return
    value)."""
    df = load_match_info(path)
    return train_model(df)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.title("IPL Match Winner Predictor")
st.markdown(
    "This tool uses a **Random Forest** model trained on historical IPL match data. "
    "Select the two teams, venue, toss details, and season — then hit **Predict Winner**."
)
st.divider()

# ---------------------------------------------------------------------------
# Load data — bail early if the folder is missing or empty
# ---------------------------------------------------------------------------

if not os.path.isdir(DATA_PATH):
    st.error(f"Data folder not found: `{DATA_PATH}`. Make sure IPL data is present.")
    st.stop()

df = load_data(DATA_PATH)

if df.empty:
    st.error("No match data loaded. Check that `data/raw/ipl/` contains `*_info.csv` files.")
    st.stop()

# Derive sorted unique values for each dropdown
all_teams   = sorted(set(df["team1"].dropna()) | set(df["team2"].dropna()))
all_venues  = sorted(df["venue"].dropna().unique())
all_seasons = sorted(df["season"].dropna().unique(), reverse=True)

# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------

st.subheader("Match Details")

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", all_teams, index=0)
with col2:
    # Default team2 to the second team in the list so it's different from team1
    team2_options = [t for t in all_teams if t != team1]
    team2 = st.selectbox("Team 2", team2_options, index=0)

venue = st.selectbox("Venue", all_venues)

col3, col4 = st.columns(2)
with col3:
    # Toss winner must be one of the two selected teams
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
with col4:
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

season = st.selectbox("Season", all_seasons)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

st.divider()

if st.button("Predict Winner", type="primary", use_container_width=True):
    with st.spinner("Training model on IPL data…"):
        try:
            model = get_model(DATA_PATH)
        except ValueError as exc:
            st.error(f"Model training failed: {exc}")
            st.stop()

    match_details = {
        "venue":         venue,
        "team1":         team1,
        "team2":         team2,
        "toss_winner":   toss_winner,
        "toss_decision": toss_decision,
        "season":        str(season),
    }

    try:
        result = predict_winner(model, match_details)
    except ValueError as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.divider()
    st.subheader("Prediction Result")

    winner = result["predicted_winner"]
    prob   = result["probability"]

    st.success(f"Predicted Winner: **{winner}**")

    col5, col6 = st.columns(2)
    col5.metric("Winning Probability", f"{prob}%")
    col6.metric(
        "Opponent",
        team2 if winner == team1 else team1,
    )
