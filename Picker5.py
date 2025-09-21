# nfl_ev_picker.py
import streamlit as st
import pandas as pd
import numpy as np
import requests

# -----------------------------
# 1️⃣ Pull Fanatics NFL lines
# -----------------------------
@st.cache_data(ttl=1800)
def get_fanatics_odds():
    """
    Pull live Fanatics NFL odds.
    Replace URL with your actual Fanatics API endpoint.
    """
    url = "https://api.fanatics.com/nfl/odds"  # placeholder
    response = requests.get(url)
    data = response.json()  # expect JSON with games, odds, teams

    games = []
    for game in data["games"]:
        games.append({
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "ml_home": game["moneyline_home"],
            "ml_away": game["moneyline_away"],
            "spread_home": game["spread_home"],
            "spread_away": game["spread_away"],
            "total": game["total"],
            "over_odds": game["over_odds"],
            "under_odds": game["under_odds"]
        })
    return pd.DataFrame(games)

# -----------------------------
# 2️⃣ Pull Team Stats / Ratings
# -----------------------------
@st.cache_data(ttl=1800)
def get_team_stats():
    """
    Pull team offensive/defensive stats for probabilities.
    Example structure; replace with real stats source.
    """
    # Example stats
    data = {
        "team": ["NE", "DAL", "KC", "GB"],
        "off_avg_pts": [24, 28, 31, 26],
        "def_avg_pts_allowed": [20, 24, 21, 23],
        "std_pts": [7, 8, 6, 7]
    }
    return pd.DataFrame(data)

# -----------------------------
# 3️⃣ Simulate Game Outcomes
# -----------------------------
def simulate_game(home_team, away_team, stats_df, n_sim=10000):
    """
    Simulate using team offense/defense averages.
    """
    home_stats = stats_df.loc[stats_df["team"] == home_team].iloc[0]
    away_stats = stats_df.loc[stats_df["team"] == away_team].iloc[0]

    home_mean = (home_stats["off_avg_pts"] + away_stats["def_avg_pts_allowed"]) / 2
    away_mean = (away_stats["off_avg_pts"] + home_stats["def_avg_pts_allowed"]) / 2

    home_scores = np.random.normal(home_mean, home_stats["std_pts"], n_sim)
    away_scores = np.random.normal(away_mean, away_stats["std_pts"], n_sim)

    home_win_prob = np.mean(home_scores > away_scores)
    home_cover_prob = np.mean(home_scores - away_scores > 0)  # vs spread placeholder
    total_points = home_scores + away_scores
    over_prob = np.mean(total_points > 45)  # placeholder total line
    return {
        "home_win_prob": home_win_prob,
        "away_win_prob": 1 - home_win_prob,
        "home_cover_prob": home_cover_prob,
        "over_prob": over_prob,
        "under_prob": 1 - over_prob
    }

# -----------------------------
# 4️⃣ EV Calculation
# -----------------------------
def calculate_ev(prob, odds):
    """
    EV = probability * payout - (1 - probability) * stake
    """
    if odds > 0:
        payout = odds / 100
    else:
        payout = 100 / abs(odds)
    stake = 1
    ev = prob * payout - (1 - prob) * stake
    return ev

# -----------------------------
# 5️⃣ Streamlit UI
# -----------------------------
st.title("NFL EV Model - Fanatics Lines")

st.write("Fetching live Fanatics NFL odds...")
games_df = get_fanatics_odds()
stats_df = get_team_stats()

ev_list = []
for _, row in games_df.iterrows():
    sim = simulate_game(row["home_team"], row["away_team"], stats_df)
    
    ev_home_ml = calculate_ev(sim["home_win_prob"], row["ml_home"])
    ev_away_ml = calculate_ev(sim["away_win_prob"], row["ml_away"])
    ev_over = calculate_ev(sim["over_prob"], row["over_odds"])
    ev_under = calculate_ev(sim["under_prob"], row["under_odds"])
    
    ev_list.append({
        "matchup": f"{row['away_team']} @ {row['home_team']}",
        "EV_Home_ML": ev_home_ml,
        "EV_Away_ML": ev_away_ml,
        "EV_Over": ev_over,
        "EV_Under": ev_under
    })

ev_df = pd.DataFrame(ev_list)
ev_df = ev_df.sort_values(by=["EV_Home_ML", "EV_Away_ML", "EV_Over", "EV_Under"], ascending=False)

st.subheader("NFL Games Ranked by EV")
st.dataframe(ev_df)
