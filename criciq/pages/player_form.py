import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.data_loader import load_matches
from utils.stats import get_player_form

FORMAT_PATHS = {
    "IPL":  os.path.join("data", "raw", "ipl"),
    "T20I": os.path.join("data", "raw", "t20s"),
    "ODI":  os.path.join("data", "raw", "odis"),
    "Test": os.path.join("data", "raw", "tests"),
}

st.title("Player Form")

available_formats = [fmt for fmt, path in FORMAT_PATHS.items() if os.path.isdir(path)]
if not available_formats:
    st.error("No match data found. Check that data/raw/<format> folders exist.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    selected_format = st.selectbox("Format", available_formats)
with col2:
    role = st.selectbox("Role", ["batter", "bowler"])

@st.cache_data
def load(path: str):
    return load_matches(path)

df = load(FORMAT_PATHS[selected_format])

if role == "batter":
    players = sorted(df["striker"].dropna().unique().tolist())
else:
    players = sorted(df["bowler"].dropna().unique().tolist())

col3, col4 = st.columns(2)
with col3:
    player = st.selectbox("Player", players)
with col4:
    last_n = st.slider("Last N innings", min_value=1, max_value=50, value=10)

if player:
    form = get_player_form(df, player, role, last_n)

    if not form:
        st.info(f"No innings data found for {player}.")
        st.stop()

    dates = [entry["date"] for entry in form]
    labels = [f"{entry['date']} (M{entry['match_id']} Inn{entry['innings']})" for entry in form]

    fig = go.Figure()

    if role == "batter":
        runs   = [entry["runs_scored"]  for entry in form]
        balls  = [entry["balls_faced"]  for entry in form]
        dismissed = [entry["dismissed"] for entry in form]

        fig.add_trace(go.Scatter(
            x=labels, y=runs,
            mode="lines+markers",
            name="Runs Scored",
            line=dict(color="#4C78A8", width=2),
            marker=dict(
                size=10,
                color=["#E45756" if d else "#4C78A8" for d in dismissed],
                symbol=["x" if d else "circle" for d in dismissed],
            ),
            hovertemplate="%{x}<br>Runs: %{y}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=labels, y=balls,
            mode="lines+markers",
            name="Balls Faced",
            line=dict(color="#F58518", width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="%{x}<br>Balls: %{y}<extra></extra>",
        ))
        fig.update_layout(yaxis_title="Count")

    else:
        wickets = [entry["wickets"]       for entry in form]
        runs_c  = [entry["runs_conceded"] for entry in form]

        fig.add_trace(go.Scatter(
            x=labels, y=wickets,
            mode="lines+markers",
            name="Wickets",
            line=dict(color="#54A24B", width=2),
            marker=dict(size=8),
            hovertemplate="%{x}<br>Wickets: %{y}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=labels, y=runs_c,
            mode="lines+markers",
            name="Runs Conceded",
            line=dict(color="#E45756", width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="%{x}<br>Runs Conceded: %{y}<extra></extra>",
        ))
        fig.update_layout(yaxis_title="Count")

    fig.update_layout(
        title=f"{player} — Last {len(form)} innings ({selected_format})",
        xaxis_title="Innings",
        xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
        legend_title="Stat",
        height=450,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Innings Log")
    if role == "batter":
        rows = [
            {
                "Date": entry["date"],
                "vs": entry["opponent"],
                "Score": f"{entry['runs_scored']} ({entry['balls_faced']})",
                "Dismissed": "out" if entry["dismissed"] else "not out",
            }
            for entry in form
        ]
    else:
        rows = [
            {
                "Date": entry["date"],
                "vs": entry["opponent"],
                "Figures": f"{entry['wickets']}-{entry['runs_conceded']}",
                "Overs": entry["overs"],
            }
            for entry in form
        ]

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
