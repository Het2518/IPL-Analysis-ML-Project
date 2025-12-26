import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="IPL Analysis ML App",
    layout="wide"
)

BASE_DIR = os.path.dirname(__file__)

# -------------------- LOAD DATA & MODEL --------------------
@st.cache_resource
def load_resources():
    model = joblib.load(
        os.path.join(BASE_DIR, "saved_models", "random_forest_task1.pkl")
    )

    matches = pd.read_csv(
        os.path.join(BASE_DIR, "matches_for_task1.csv")
    )

    players = pd.read_csv(
        os.path.join(BASE_DIR, "players_for_task2.csv")
    )

    return model, matches, players


model, matches_df, players_df = load_resources()

# -------------------- SIDEBAR --------------------
st.sidebar.title("🏏 IPL ML Dashboard")
page = st.sidebar.radio(
    "Select Feature",
    ["Match Winner Prediction", "Batsman Performance", "IPL Analytics"]
)

# -------------------- PAGE 1 --------------------
if page == "Match Winner Prediction":
    st.title("🏆 Match Winner Prediction")

    teams = sorted(matches_df["team1"].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams)
    with col2:
        team2 = st.selectbox("Team 2", teams)

    toss_winner = st.selectbox("Toss Winner", [team1, team2])
    toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

    if st.button("Predict Winner"):
        input_df = pd.DataFrame({
            "team1": [team1],
            "team2": [team2],
            "toss_winner": [toss_winner],
            "toss_decision": [toss_decision]
        })

        prediction = model.predict(input_df)[0]

        st.success(f"🏆 Predicted Winner: **{prediction}**")

# -------------------- PAGE 2 --------------------
elif page == "Batsman Performance":
    st.title("🏏 Batsman Performance Analysis")

    batsmen = sorted(players_df["batter"].dropna().unique())
    batter = st.selectbox("Select Batsman", batsmen)

    batter_df = players_df[players_df["batter"] == batter]

    col1, col2, col3 = st.columns(3)
    col1.metric("Runs", batter_df["runs"].sum())
    col2.metric("Balls Faced", batter_df.shape[0])
    col3.metric(
        "Strike Rate",
        round((batter_df["runs"].sum() / batter_df.shape[0]) * 100, 2)
        if batter_df.shape[0] > 0 else 0
    )

    fig = px.histogram(
        batter_df,
        x="runs",
        nbins=20,
        title=f"Run Distribution for {batter}"
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- PAGE 3 --------------------
elif page == "IPL Analytics":
    st.title("📊 IPL Match Analytics")

    season = st.selectbox(
        "Select Season",
        sorted(matches_df["season"].dropna().unique())
    )

    season_df = matches_df[matches_df["season"] == season]

    wins = season_df["winner"].value_counts().reset_index()
    wins.columns = ["Team", "Wins"]

    fig = px.bar(
        wins,
        x="Team",
        y="Wins",
        title=f"Team Wins in {season}"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("IPL Analysis ML Project • Streamlit Cloud Ready ✅")
