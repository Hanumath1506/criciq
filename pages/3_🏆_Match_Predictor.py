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

    # ── Empty state ───────────────────────────────────────────────────────────
    # Replaces the plain st.info() when no innings data exists for the player.
    if not form:
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
                No recent innings data found
            </div>
            <div style="font-size: 14px; color: #7a6a52;">
                <strong>{player}</strong> has no recorded {role} innings in this dataset.<br>
                Try selecting another player or format.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Player banner ─────────────────────────────────────────────────────────
    # Dynamic header mirroring the Head-to-Head page banner style.
    # Shows once valid data is confirmed — never displays on empty state.
    role_label = "Batter" if role == "batter" else "Bowler"
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
            Player Form Analysis &nbsp;·&nbsp; {role_label}
        </div>
        <div style="font-size: 26px; font-weight: 800; color: #1a2744;">
            {player}
        </div>
        <div style="font-size: 13px; color: #7a6a52; margin-top: 6px;">
            Last {len(form)} innings &nbsp;·&nbsp; {selected_format}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stat color palette ────────────────────────────────────────────────────
    #   GOLD  (#F5A623) — run/score-based stats
    #   GREEN (#2E8B57) — averages, consistency, wickets
    #   RED   (#E45756) — dismissals, runs conceded
    GOLD  = "#F5A623"
    GREEN = "#2E8B57"
    RED   = "#E45756"

    def stat_card(label: str, value, color: str) -> str:
        """Return an HTML stat card consistent with the Head-to-Head page style.

        The `color` parameter drives:
          - left border strip (category identifier)
          - label text color
        A fixed dark background keeps every card legible for any accent color.
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

    # ── Summary stat cards ────────────────────────────────────────────────────
    if role == "batter":
        total_runs  = sum(e["runs_scored"] for e in form)
        dismissals  = sum(1 for e in form if e["dismissed"])
        balls_total = sum(e["balls_faced"] for e in form)
        average     = round(total_runs / dismissals, 1) if dismissals > 0 else "N/A"
        strike_rate = round(total_runs / balls_total * 100, 1) if balls_total > 0 else "N/A"
        high_score  = max(e["runs_scored"] for e in form)

        r1 = st.columns(3)
        r2 = st.columns(3)
        r1[0].markdown(stat_card("Total Runs",  total_runs,   GOLD),  unsafe_allow_html=True)
        r1[1].markdown(stat_card("High Score",  high_score,   GOLD),  unsafe_allow_html=True)
        r1[2].markdown(stat_card("Strike Rate", strike_rate,  GREEN), unsafe_allow_html=True)
        r2[0].markdown(stat_card("Innings",     len(form),    GREEN), unsafe_allow_html=True)
        r2[1].markdown(stat_card("Average",     average,      GREEN), unsafe_allow_html=True)
        r2[2].markdown(stat_card("Dismissals",  dismissals,   RED),   unsafe_allow_html=True)

    else:
        total_wickets = sum(e["wickets"]       for e in form)
        total_runs_c  = sum(e["runs_conceded"] for e in form)
        total_overs   = sum(e["overs"]         for e in form)
        economy       = round(total_runs_c / total_overs, 2) if total_overs > 0 else "N/A"
        best_entry    = max(form, key=lambda e: e["wickets"])
        best_figures  = f"{best_entry['wickets']}-{best_entry['runs_conceded']}"

        r1 = st.columns(3)
        r2 = st.columns(3)
        r1[0].markdown(stat_card("Total Wickets", total_wickets,           GREEN), unsafe_allow_html=True)
        r1[1].markdown(stat_card("Best Figures",  best_figures,            GREEN), unsafe_allow_html=True)
        r1[2].markdown(stat_card("Economy",       economy,                 GREEN), unsafe_allow_html=True)
        r2[0].markdown(stat_card("Innings",       len(form),               GREEN), unsafe_allow_html=True)
        r2[1].markdown(stat_card("Runs Conceded", total_runs_c,            RED),   unsafe_allow_html=True)
        r2[2].markdown(stat_card("Total Overs",   round(total_overs, 1),   GOLD),  unsafe_allow_html=True)

    # ── Form trend chart ──────────────────────────────────────────────────────
    labels = [f"{e['date']} (M{e['match_id']} Inn{e['innings']})" for e in form]

    fig = go.Figure()

    if role == "batter":
        runs      = [e["runs_scored"] for e in form]
        balls     = [e["balls_faced"] for e in form]
        dismissed = [e["dismissed"]   for e in form]

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
        wickets = [e["wickets"]       for e in form]
        runs_c  = [e["runs_conceded"] for e in form]

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

    # ── Chart theming ─────────────────────────────────────────────────────────
    # Transparent backgrounds blend the chart into the page theme.
    # Subtle gridlines keep axes readable without a harsh white box.
    fig.update_layout(
        title=dict(
            text=f"{player} — Last {len(form)} innings ({selected_format})",
            font=dict(color="#1a2744", size=16),
        ),
        xaxis_title="Innings",
        xaxis=dict(
            tickangle=-35,
            tickfont=dict(size=10),
            gridcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            gridcolor="rgba(26,39,68,0.12)",
            zerolinecolor="rgba(26,39,68,0.2)",
        ),
        legend_title="Stat",
        height=450,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a2744"),
        margin=dict(t=56, b=48, l=40, r=24),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Innings log ───────────────────────────────────────────────────────────
    st.subheader("Innings Log")
    if role == "batter":
        rows = [
            {
                "Date":      e["date"],
                "Score":     f"{e['runs_scored']} ({e['balls_faced']})",
                "Dismissed": "out" if e["dismissed"] else "not out",
            }
            for e in form
        ]
    else:
        rows = [
            {
                "Date":    e["date"],
                "Figures": f"{e['wickets']}-{e['runs_conceded']}",
                "Overs":   e["overs"],
            }
            for e in form
        ]

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
