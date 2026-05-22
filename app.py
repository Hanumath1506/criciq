import streamlit as st
from utils.data_loader import load_matches

st.title("🏏 CricIQ")

st.markdown(
    "CricIQ is a cricket analytics app built on IPL ball-by-ball data. "
    "It lets you explore **head-to-head** batter vs bowler matchups, track **player form** "
    "across recent innings, and predict **match winners** using a trained Random Forest model."
)

st.subheader("Features")
st.markdown("""
- **Head to Head** — Analyze batter vs bowler matchups using historical IPL data, including runs scored, dismissals, and strike rate.
- **Player Form** — Track a player's recent performances across innings to understand scoring trends and current form.
- **Match Predictor** — Predict the winner of an IPL match based on venue, teams, toss result, and season using a Random Forest model.
""")
