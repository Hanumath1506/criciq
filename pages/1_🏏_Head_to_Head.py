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

    # ── Empty state ───────────────────────────────────────────────────────────
    # Shown instead of the default st.info() when no matchup data exists.
    if stats["matches"] == 0:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FFF8EE, #F5E6CC);
            border-radius: 16px;
            padding: 52px 32px;
            text-align: center;
            margin: 28px 0;
            box-shadow: 0 2px 14px rgba(0,0,0,0.07);
        ">
            <div style="font-size: 52px; margin-bottom: 16px;">🏏</div>
            <div style="
                font-size: 20px;
                font-weight: 700;
                color: #1a2744;
                margin-bottom: 10px;
            ">
                No historical matchup data found
            </div>
            <div style="font-size: 14px; color: #7a6a52;">
                <strong>{batter}</strong> and <strong>{bowler}</strong>
                have never faced each other in this dataset.<br>
                Try selecting another batter or bowler.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Dynamic matchup banner ─────────────────────────────────────────────────
    # Replaces the plain st.subheader — shows a cricket-themed styled header
    # with the batter's name vs the bowler's name once both are selected.
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #FFF8EE, #F5E6CC);
        border-left: 6px solid #F5A623;
        border-radius: 14px;
        padding: 26px 32px;
        text-align: center;
        margin: 16px 0 28px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    ">
        <div style="
            font-size: 12px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #8B6A1F;
            font-weight: 600;
            margin-bottom: 8px;
        ">
            Batter vs Bowler
        </div>
        <div style="font-size: 26px; font-weight: 800; color: #1a2744;">
            {batter}&nbsp;
            <span style="color: #F5A623;">vs</span>
            &nbsp;{bowler}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Color palette for stat categories:
    #   GOLD  (#F5A623) — batting stats (matches, balls, runs, strike rate)
    #   RED   (#E45756) — dismissal stats
    #   GREEN (#2E8B57) — bowling stats (economy)
    GOLD  = "#F5A623"
    RED   = "#E45756"
    GREEN = "#2E8B57"

    def stat_card(label: str, value, color: str) -> str:
        """Return an HTML card string for a single stat.

        The `color` parameter drives all accent elements:
          - left border strip (category identifier)
          - label text (reinforces the category at a glance)
        A consistent dark background keeps every card legible regardless of
        which accent color is passed in.
        """
        return f"""
        <div style="
            background-color: #12213A;
            border-left: 5px solid {color};
            border-radius: 12px;
            padding: 20px 16px;
            text-align: center;
            margin-bottom: 12px;
        ">
            <div style="
                font-size: 12px;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: {color};
                font-weight: 600;
            ">
                {label}
            </div>
            <div style="
                font-size: 30px;
                font-weight: 700;
                margin-top: 6px;
                color: white;
            ">
                {value}
            </div>
        </div>
        """

    row1 = st.columns(3)
    row2 = st.columns(3)

    row1[0].markdown(stat_card("Matches",        stats["matches"],      GOLD),  unsafe_allow_html=True)
    row1[1].markdown(stat_card("Balls Faced",    stats["balls_faced"],  GOLD),  unsafe_allow_html=True)
    row1[2].markdown(stat_card("Runs Scored",    stats["runs_scored"],  GOLD),  unsafe_allow_html=True)
    row2[0].markdown(stat_card("Dismissals",     stats["dismissals"],   RED),   unsafe_allow_html=True)
    row2[1].markdown(stat_card("Strike Rate",    stats["strike_rate"],  GOLD),  unsafe_allow_html=True)
    row2[2].markdown(stat_card("Bowler Economy", stats["economy"],      GREEN), unsafe_allow_html=True)

    fig = go.Figure(data=[
        go.Bar(name="Runs Scored",  x=["Runs Scored"],  y=[stats["runs_scored"]],  marker_color="#4C78A8"),
        go.Bar(name="Balls Faced",  x=["Balls Faced"],  y=[stats["balls_faced"]],  marker_color="#F58518"),
        go.Bar(name="Dismissals",   x=["Dismissals"],   y=[stats["dismissals"]],   marker_color="#E45756"),
    ])
    # ── Chart styling ─────────────────────────────────────────────────────────
    # Transparent backgrounds let the chart blend into the page theme.
    # Subtle gridlines keep the axes readable without a harsh white box.
    fig.update_layout(
        barmode="group",
        title=dict(text=f"{batter} vs {bowler} — Summary", font=dict(color="#1a2744", size=16)),
        yaxis_title="Count",
        legend_title="Stat",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a2744"),
        yaxis=dict(gridcolor="rgba(26,39,68,0.12)", zerolinecolor="rgba(26,39,68,0.2)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        margin=dict(t=56, b=32, l=40, r=24),
    )
    st.plotly_chart(fig, use_container_width=True)
