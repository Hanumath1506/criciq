import os
import streamlit as st
import plotly.graph_objects as go

from utils.data_loader import load_matches
from utils.stats import get_head_to_head

FORMAT_PATHS = {
    "IPL":   os.path.join("data", "raw", "ipl"),
    "T20I":  os.path.join("data", "raw", "t20s"),
    "ODI":   os.path.join("data", "raw", "odis"),
    "Test":  os.path.join("data", "raw", "tests"),
}

st.title("Head to Head")

available_formats = [fmt for fmt, path in FORMAT_PATHS.items() if os.path.isdir(path)]
if not available_formats:
    st.error("No match data found. Check that data/raw/<format> folders exist.")
    st.stop()

selected_format = st.selectbox("Format", available_formats)

@st.cache_data
def load(path: str):
    return load_matches(path)

df = load(FORMAT_PATHS[selected_format])

batters = sorted(df["striker"].dropna().unique().tolist())
bowlers = sorted(df["bowler"].dropna().unique().tolist())

col1, col2 = st.columns(2)
with col1:
    batter = st.selectbox("Batter", batters)
with col2:
    bowler = st.selectbox("Bowler", bowlers)

if batter and bowler:
    stats = get_head_to_head(df, batter, bowler)

    if stats["matches"] == 0:
        st.info(f"{batter} and {bowler} have never faced each other in this dataset.")
        st.stop()

    st.subheader(f"{batter} vs {bowler}")

    m1, m2, m3 = st.columns(3)
    m4, m5, m6 = st.columns(3)

    m1.metric("Matches", stats["matches"])
    m2.metric("Balls Faced", stats["balls_faced"])
    m3.metric("Runs Scored", stats["runs_scored"])
    m4.metric("Dismissals", stats["dismissals"])
    m5.metric("Strike Rate", stats["strike_rate"])
    m6.metric("Bowler Economy", stats["economy"])

    fig = go.Figure(data=[
        go.Bar(name="Runs Scored",  x=["Runs Scored"],  y=[stats["runs_scored"]],  marker_color="#4C78A8"),
        go.Bar(name="Balls Faced",  x=["Balls Faced"],  y=[stats["balls_faced"]],  marker_color="#F58518"),
        go.Bar(name="Dismissals",   x=["Dismissals"],   y=[stats["dismissals"]],   marker_color="#E45756"),
    ])
    fig.update_layout(
        barmode="group",
        title=f"{batter} vs {bowler} — Summary",
        yaxis_title="Count",
        legend_title="Stat",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
