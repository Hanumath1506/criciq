import pandas as pd


def get_batting_stats(df: pd.DataFrame, player_name: str) -> dict:
    batting = df[df["striker"] == player_name].copy()

    # Each row is a ball; an innings is a unique (match_id, innings) pair where the batter faced a ball
    innings_groups = batting.groupby(["match_id", "innings"])

    innings_count = len(innings_groups)
    total_runs = batting["runs_off_bat"].sum()

    # Balls faced: exclude wides (wides > 0 means ball not faced by batter)
    balls_faced = batting[batting["wides"].isna() | (batting["wides"] == 0)].shape[0]

    # Dismissals: rows where player_dismissed matches the batter
    dismissals = df[df["player_dismissed"] == player_name]["match_id"].count()

    average = round(total_runs / dismissals, 2) if dismissals > 0 else float("inf")
    strike_rate = round((total_runs / balls_faced) * 100, 2) if balls_faced > 0 else 0.0

    # 50s and 100s: per innings run totals
    innings_runs = innings_groups["runs_off_bat"].sum()
    fifties = int(((innings_runs >= 50) & (innings_runs < 100)).sum())
    hundreds = int((innings_runs >= 100).sum())

    return {
        "player": player_name,
        "innings": innings_count,
        "total_runs": int(total_runs),
        "average": average,
        "strike_rate": strike_rate,
        "50s": fifties,
        "100s": hundreds,
    }


def get_bowling_stats(df: pd.DataFrame, player_name: str) -> dict:
    bowling = df[df["bowler"] == player_name].copy()

    # Legal deliveries: exclude wides and no-balls
    legal = bowling[
        (bowling["wides"].isna() | (bowling["wides"] == 0))
        & (bowling["noballs"].isna() | (bowling["noballs"] == 0))
    ]
    legal_balls = len(legal)
    overs_bowled = round(legal_balls / 6, 2)

    runs_conceded = int(bowling["runs_off_bat"].sum() + bowling["extras"].sum())

    wickets = int(
        bowling[
            bowling["wicket_type"].notna()
            & ~bowling["wicket_type"].isin(["run out", "retired hurt", "obstructing the field"])
        ].shape[0]
    )

    # Innings bowled: unique (match_id, innings) pairs where bowler delivered a ball
    innings_bowled = bowling.groupby(["match_id", "innings"]).ngroups

    economy = round(runs_conceded / overs_bowled, 2) if overs_bowled > 0 else 0.0
    average = round(runs_conceded / wickets, 2) if wickets > 0 else float("inf")

    return {
        "player": player_name,
        "innings_bowled": innings_bowled,
        "wickets": wickets,
        "runs_conceded": runs_conceded,
        "economy": economy,
        "average": average,
    }


def get_head_to_head(df: pd.DataFrame, batter: str, bowler: str) -> dict:
    h2h = df[(df["striker"] == batter) & (df["bowler"] == bowler)]

    balls_faced = int(h2h[h2h["wides"].isna() | (h2h["wides"] == 0)].shape[0])
    runs_scored = int(h2h["runs_off_bat"].sum())
    dismissals = int(h2h[h2h["player_dismissed"] == batter].shape[0])
    matches = h2h["match_id"].nunique()

    strike_rate = round((runs_scored / balls_faced) * 100, 2) if balls_faced > 0 else 0.0

    legal_balls = int(
        h2h[
            (h2h["wides"].isna() | (h2h["wides"] == 0))
            & (h2h["noballs"].isna() | (h2h["noballs"] == 0))
        ].shape[0]
    )
    runs_conceded = int(h2h["runs_off_bat"].sum() + h2h["extras"].sum())
    overs = legal_balls / 6
    economy = round(runs_conceded / overs, 2) if overs > 0 else 0.0

    return {
        "batter": batter,
        "bowler": bowler,
        "matches": matches,
        "balls_faced": balls_faced,
        "runs_scored": runs_scored,
        "dismissals": dismissals,
        "strike_rate": strike_rate,
        "economy": economy,
    }


def get_player_form(df: pd.DataFrame, player_name: str, role: str, last_n: int = 10) -> list[dict]:
    if role == "batter":
        player_df = df[df["striker"] == player_name]
    elif role == "bowler":
        player_df = df[df["bowler"] == player_name]
    else:
        raise ValueError("role must be 'batter' or 'bowler'")

    player_df = player_df.copy()
    player_df["start_date"] = pd.to_datetime(player_df["start_date"])

    innings_groups = (
        player_df.sort_values("start_date")
        .groupby(["match_id", "innings"], sort=False)
    )

    # Preserve chronological order of innings
    ordered_keys = list(dict.fromkeys(
        (row["match_id"], row["innings"])
        for _, row in player_df.sort_values("start_date").iterrows()
    ))

    result = []
    for match_id, innings in ordered_keys[-last_n:]:
        group = innings_groups.get_group((match_id, innings))
        date = group["start_date"].iloc[0].date().isoformat()

        if role == "batter":
            balls_faced = int(group[group["wides"].isna() | (group["wides"] == 0)].shape[0])
            runs_scored = int(group["runs_off_bat"].sum())
            dismissed = int(group["player_dismissed"].eq(player_name).any())
            result.append({
                "match_id": match_id,
                "innings": innings,
                "date": date,
                "runs_scored": runs_scored,
                "balls_faced": balls_faced,
                "dismissed": bool(dismissed),
            })
        else:
            legal_balls = int(group[
                (group["wides"].isna() | (group["wides"] == 0))
                & (group["noballs"].isna() | (group["noballs"] == 0))
            ].shape[0])
            runs_conceded = int(group["runs_off_bat"].sum() + group["extras"].sum())
            wickets = int(group[
                group["wicket_type"].notna()
                & ~group["wicket_type"].isin(["run out", "retired hurt", "obstructing the field"])
            ].shape[0])
            overs = round(legal_balls / 6, 1)
            result.append({
                "match_id": match_id,
                "innings": innings,
                "date": date,
                "overs": overs,
                "runs_conceded": runs_conceded,
                "wickets": wickets,
            })

    return result
