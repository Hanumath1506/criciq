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
# Cached helpers — run once and stay in memory across reruns
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
# Page banner
# ---------------------------------------------------------------------------

# Static header shown at the top of the page — matches the Head-to-Head and
# Player Form banner style for visual consistency across the app.
st.markdown("""
<div style="
    background: linear-gradient(135deg, #FFF8EE, #F5E6CC);
    border-left: 6px solid #F5A623;
    border-radius: 14px;
    padding: 28px 32px;
    text-align: center;
    margin-bottom: 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
">
    <div style="font-size: 28px; font-weight: 800; color: #1a2744;">
        🏏 IPL Match Winner Predictor
    </div>
    <div style="
        font-size: 14px;
        color: #7a6a52;
        margin-top: 10px;
        max-width: 560px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.7;
    ">
        Predict IPL match outcomes using historical match data and a
        Random Forest machine learning model.
    </div>
</div>
""", unsafe_allow_html=True)

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
# Input section — Teams & Venue
# ---------------------------------------------------------------------------

st.markdown("##### Teams & Venue")

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1", all_teams, index=0)
with col2:
    # Exclude the already-chosen team1 so both dropdowns are always different
    team2_options = [t for t in all_teams if t != team1]
    team2 = st.selectbox("Team 2", team2_options, index=0)

venue = st.selectbox("Venue", all_venues)

# ---------------------------------------------------------------------------
# Input section — Toss & Season
# ---------------------------------------------------------------------------

st.markdown("##### Toss & Season")

col3, col4 = st.columns(2)
with col3:
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
with col4:
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

season = st.selectbox("Season", all_seasons)

# ---------------------------------------------------------------------------
# Dynamic matchup preview banner
# ---------------------------------------------------------------------------

# Shown once both teams are set — gives a visual summary of the upcoming
# prediction before the user clicks the button.
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #FFF8EE, #F5E6CC);
    border-left: 6px solid #F5A623;
    border-radius: 14px;
    padding: 22px 32px;
    text-align: center;
    margin: 24px 0 20px 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
">
    <div style="
        font-size: 11px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #8B6A1F;
        font-weight: 600;
        margin-bottom: 8px;
    ">
        Match Preview
    </div>
    <div style="font-size: 22px; font-weight: 800; color: #1a2744;">
        {team1}&nbsp;
        <span style="color: #F5A623;">vs</span>
        &nbsp;{team2}
    </div>
    <div style="font-size: 13px; color: #7a6a52; margin-top: 8px;">
        📍 {venue} &nbsp;·&nbsp; {season}
        &nbsp;·&nbsp; Toss: <strong>{toss_winner}</strong> chose to <strong>{toss_decision}</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------

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

    winner   = result["predicted_winner"]
    prob     = result["probability"]
    opponent = team2 if winner == team1 else team1

    # ── Prediction result card ────────────────────────────────────────────────
    # Styled result display — replaces st.success() + st.metric().
    # Probability is shown in green to give a quick confidence read-out.
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #FFF8EE, #F5E6CC);
        border-left: 6px solid #2E8B57;
        border-radius: 14px;
        padding: 32px;
        text-align: center;
        margin: 24px 0 16px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    ">
        <div style="
            font-size: 11px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #8B6A1F;
            font-weight: 600;
            margin-bottom: 6px;
        ">
            🏆 Predicted Winner
        </div>
        <div style="font-size: 30px; font-weight: 800; color: #1a2744; margin-bottom: 20px;">
            {winner}
        </div>
        <div style="
            font-size: 11px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #8B6A1F;
            font-weight: 600;
            margin-bottom: 6px;
        ">
            📊 Winning Probability
        </div>
        <div style="font-size: 36px; font-weight: 800; color: #2E8B57;">
            {prob}%
        </div>
        <div style="font-size: 13px; color: #7a6a52; margin-top: 12px;">
            Opponent: <strong>{opponent}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Confidence progress bar ───────────────────────────────────────────────
    # Visual probability bar; st.progress() expects a value in [0.0, 1.0].
    st.caption(f"Model confidence: {prob}%")
    st.progress(prob / 100)
