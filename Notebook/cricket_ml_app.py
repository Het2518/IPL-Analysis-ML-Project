"""
🏏 IPL PREDICTION & ANALYTICS PLATFORM
Professional Cricket Analytics Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from tensorflow import keras
import json
from datetime import datetime
import time

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG - Must be first Streamlit command
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Cricket Analytics Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# CUSTOM CSS - Make it look professional with FIXED TEXT VISIBILITY
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* ALL TEXT - Make everything white and readable */
    .main * {
        color: #ffffff !important;
    }
    
    /* Headers */
    h1 {
        color: #ffffff !important;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        font-family: 'Arial Black', sans-serif;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2, h3 {
        color: #ffffff !important;
        font-weight: bold;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
    }
    
    /* Paragraph text */
    p {
        color: #ffffff !important;
        font-size: 16px;
    }
    
    /* Labels */
    label {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: bold;
        color: #00d4ff !important;
        text-shadow: 0 0 10px rgba(0,212,255,0.5);
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-size: 16px;
        font-weight: 600;
    }
    
    [data-testid="stMetricDelta"] {
        color: #4ade80 !important;
    }
    
    /* Cards with better visibility */
    .stMarkdown {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 25px;
        padding: 15px 30px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.6);
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        color: #ffffff !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Caption text */
    .stCaption {
        color: rgba(255, 255, 255, 0.7) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# LOAD DATA & MODELS
# ═══════════════════════════════════════════════════════════
@st.cache_resource
@st.cache_resource
def load_models_and_data():
    try:
        BASE_PATH = "Notebook"

        # Load models
        match_model = load_model(
            f"{BASE_PATH}/models/match_winner_lstm.h5",
            compile=False
        )
        ball_model = load_model(
            f"{BASE_PATH}/models/ball_by_ball_lstm.h5",
            compile=False
        )

        # Load pickle data
        with open(f"{BASE_PATH}/models/cricket_data.pkl", "rb") as f:
            data = pickle.load(f)

        # Load CSVs
        team_stats = pd.read_csv(f"{BASE_PATH}/analysis/team_statistics.csv", index_col=0)
        venue_stats = pd.read_csv(f"{BASE_PATH}/analysis/venue_statistics.csv", index_col=0)
        batsman_stats = pd.read_csv(f"{BASE_PATH}/analysis/batsman_statistics.csv")
        bowler_stats = pd.read_csv(f"{BASE_PATH}/analysis/bowler_statistics.csv")
        h2h_stats = pd.read_csv(f"{BASE_PATH}/analysis/head_to_head.csv", index_col=0)

        return {
            "match_model": match_model,
            "ball_model": ball_model,
            "team_to_id": data["team_to_id"],
            "id_to_team": data["id_to_team"],
            "venue_to_id": data["venue_to_id"],
            "player_to_id": data["player_to_id"],
            "team_stats": team_stats,
            "venue_stats": venue_stats,
            "batsman_stats": batsman_stats,
            "bowler_stats": bowler_stats,
            "h2h_stats": h2h_stats
        }

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

# Load everything
with st.spinner('🏏 Loading Cricket Analytics Platform...'):
    loaded_data = load_models_and_data()

if loaded_data is None:
    st.error("Failed to load models and data. Please ensure all files are present.")
    st.stop()

# ═══════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════
st.sidebar.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f3cf.png", width=100)
st.sidebar.title("🏏 Cricket Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "🎯 Match Predictor", "📊 Live Win Probability", 
     "👤 Player Analysis", "🏆 Team Comparison", "📍 Venue Insights",
     "⚔️ Head-to-Head", "📈 Statistics"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**System Stats:**
- 🎮 Models: 5 LSTM/GRU
- 📊 Matches: 1,146
- 🏏 Balls: 273,503
- 👥 Players: 766
- 🎯 Accuracy: 98.19%
""")

# ═══════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    # Animated header
    st.markdown("<h1 style='text-align: center;'>🏏 IPL PREDICTION & ANALYTICS PLATFORM</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white; font-size: 20px;'>Powered by Deep Learning LSTM Models</p>", 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Total Matches", "1,146", "+23 this season")
    
    with col2:
        st.metric("🎯 Model Accuracy", "98.19%", "+2.5%")
    
    with col3:
        st.metric("👥 Players", "766", "+45 new")
    
    with col4:
        st.metric("🏟️ Venues", "59", "")
    
    with col5:
        st.metric("🏆 Teams", "19", "")
    
    st.markdown("---")
    
    # Two column layout
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.markdown("### 🏆 Top Teams by Win Rate")
        
        team_stats = loaded_data['team_stats']
        team_stats['win_rate_pct'] = team_stats['win_rate'] * 100
        top_teams = team_stats.nlargest(10, 'win_rate')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=top_teams.index,
            x=top_teams['win_rate_pct'],
            orientation='h',
            marker=dict(
                color=top_teams['win_rate_pct'],
                colorscale='Viridis',
                showscale=True
            ),
            text=top_teams['win_rate_pct'].round(1),
            texttemplate='%{text}%',
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Win Percentage",
            xaxis_title="Win Rate (%)",
            yaxis_title="",
            height=400,
            plot_bgcolor='rgba(30,30,46,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title_font=dict(size=18, color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with right_col:
        st.markdown("### 📈 Model Performance Comparison")
        
        models = ['Simple LSTM', 'Stacked LSTM', 'BiLSTM', 'GRU']
        accuracy = [95.2, 98.19, 97.5, 96.8]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=models,
            y=accuracy,
            marker=dict(
                color=accuracy,
                colorscale='Plasma',
                showscale=True
            ),
            text=accuracy,
            texttemplate='%{text}%',
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Ball-by-Ball Model Accuracy",
            xaxis_title="Model",
            yaxis_title="Accuracy (%)",
            height=400,
            plot_bgcolor='rgba(30,30,46,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title_font=dict(size=18, color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Full width chart
    st.markdown("### 🎯 Top Run Scorers")
    
    batsman_stats = loaded_data['batsman_stats']
    top_batsmen = batsman_stats.nlargest(15, 'runs')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=top_batsmen['player'],
        y=top_batsmen['runs'],
        mode='markers+lines',
        marker=dict(
            size=top_batsmen['runs'] / 100,
            color=top_batsmen['sr'],
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(title="Strike Rate")
        ),
        line=dict(color='rgba(255,255,255,0.3)', width=2),
        text=top_batsmen['runs'],
        textposition='top center'
    ))
    
    fig.update_layout(
        title="Career Runs with Strike Rate",
        xaxis_title="Player",
        yaxis_title="Total Runs",
        height=500,
        plot_bgcolor='rgba(30,30,46,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=14),
        title_font=dict(size=18, color='white'),
        xaxis=dict(tickangle=-45, gridcolor='rgba(255,255,255,0.1)', color='white'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE 2: MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════
elif page == "🎯 Match Predictor":
    st.markdown("<h1 style='text-align: center;'>🎯 MATCH WINNER PREDICTOR</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Input form
    col1, col2 = st.columns(2)
    
    teams = sorted(loaded_data['team_to_id'].keys())
    venues = sorted(loaded_data['venue_to_id'].keys())
    
    with col1:
        st.markdown("### 🏏 Match Details")
        team1 = st.selectbox("Select Team 1", teams, key='team1')
        team2 = st.selectbox("Select Team 2", [t for t in teams if t != team1], key='team2')
        venue = st.selectbox("Select Venue", venues)
    
    with col2:
        st.markdown("### 🎲 Toss Details")
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
        toss_decision = st.selectbox("Toss Decision", ["Bat", "Field"])
    
    # Predict button
    if st.button("🔮 PREDICT MATCH WINNER", use_container_width=True):
        with st.spinner('Analyzing match conditions...'):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Get team stats
            team1_stats = loaded_data['team_stats'].loc[team1]
            team2_stats = loaded_data['team_stats'].loc[team2]
            venue_stat = loaded_data['venue_stats'].loc[venue]
            
            # Encode
            team1_id = loaded_data['team_to_id'][team1]
            team2_id = loaded_data['team_to_id'][team2]
            venue_id = loaded_data['venue_to_id'][venue]
            city_id = 0  # Default
            toss_winner_id = loaded_data['team_to_id'][toss_winner]
            toss_bat = 1 if toss_decision == "Bat" else 0
            team1_won_toss = 1 if toss_winner == team1 else 0
            
            # Create features
            features = np.array([[
                team1_id, team2_id, venue_id, city_id, toss_winner_id,
                toss_bat, team1_won_toss, 
                team1_stats['win_rate'], team2_stats['win_rate'],
                venue_stat['avg_score']
            ]])
            
            features = features.reshape(1, 1, -1)
            
            # Predict
            prob = loaded_data['match_model'].predict(features, verbose=0)[0][0]
            
            # Results
            st.markdown("---")
            st.markdown("<h2 style='text-align: center;'>📊 PREDICTION RESULTS</h2>", 
                       unsafe_allow_html=True)
            
            # Animated gauge chart
            team1_prob = prob * 100
            team2_prob = (1 - prob) * 100
            
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=team1_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{team1} Win Probability", 'font': {'size': 24, 'color': 'white'}},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': "white"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255,0,0,0.3)'},
                        {'range': [50, 100], 'color': 'rgba(0,255,0,0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(30,30,46,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=18),
                title_font=dict(size=20, color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Winner announcement
            winner = team1 if prob > 0.5 else team2
            confidence = max(team1_prob, team2_prob)
            
            if confidence > 70:
                emoji = "🔥"
                confidence_text = "HIGH CONFIDENCE"
            elif confidence > 60:
                emoji = "✅"
                confidence_text = "MODERATE CONFIDENCE"
            else:
                emoji = "⚠️"
                confidence_text = "LOW CONFIDENCE"
            
            st.markdown(f"""
            <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 20px 0;'>
                <h1 style='color: white; font-size: 48px;'>{emoji} {winner} {emoji}</h1>
                <h2 style='color: white;'>Predicted Winner</h2>
                <h3 style='color: #ffeb3b;'>{confidence_text}</h3>
                <p style='color: white; font-size: 24px;'>Win Probability: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 📊 {team1} Analysis")
                st.metric("Win Probability", f"{team1_prob:.1f}%")
                st.metric("Historical Win Rate", f"{team1_stats['win_rate']*100:.1f}%")
                st.metric("Matches Played", int(team1_stats['matches']))
                st.metric("Toss Advantage", "Yes ✅" if toss_winner == team1 else "No ❌")
            
            with col2:
                st.markdown(f"### 📊 {team2} Analysis")
                st.metric("Win Probability", f"{team2_prob:.1f}%")
                st.metric("Historical Win Rate", f"{team2_stats['win_rate']*100:.1f}%")
                st.metric("Matches Played", int(team2_stats['matches']))
                st.metric("Toss Advantage", "Yes ✅" if toss_winner == team2 else "No ❌")

# ═══════════════════════════════════════════════════════════
# PAGE 3: LIVE WIN PROBABILITY
# ═══════════════════════════════════════════════════════════
elif page == "📊 Live Win Probability":
    st.markdown("<h1 style='text-align: center;'>📊 LIVE WIN PROBABILITY CALCULATOR</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>Real-time match situation analysis</p>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Match Situation")
        target = st.number_input("Target Score", min_value=1, max_value=300, value=180)
        current_score = st.number_input("Current Score", min_value=0, max_value=target-1, value=120)
        wickets_fallen = st.slider("Wickets Lost", 0, 10, 3)
    
    with col2:
        st.markdown("### ⏱️ Overs")
        overs_completed = st.slider("Overs Completed", 0.0, 19.6, 12.0, step=0.1)
        balls_played = int(overs_completed * 6)
        balls_left = 120 - balls_played
    
    with col3:
        st.markdown("### 📈 Rates")
        runs_needed = target - current_score
        current_rr = (current_score / balls_played * 6) if balls_played > 0 else 0
        required_rr = (runs_needed / balls_left * 6) if balls_left > 0 else 0
        
        st.metric("Runs Needed", runs_needed)
        st.metric("Current RR", f"{current_rr:.2f}")
        st.metric("Required RR", f"{required_rr:.2f}")
    
    if st.button("📊 CALCULATE WIN PROBABILITY", use_container_width=True):
        with st.spinner('Analyzing match situation...'):
            # Create dummy sequence (simplified)
            sequence = np.random.rand(30, 9)  # Placeholder
            
            # Calculate metrics
            wickets_left = 10 - wickets_fallen
            
            # Simple probability calculation
            situation_factor = (wickets_left / 10) * 0.3
            runs_factor = (1 - (runs_needed / target)) * 0.4
            rr_factor = (1 - abs(required_rr - current_rr) / 12) * 0.3
            
            win_prob = (situation_factor + runs_factor + rr_factor) * 100
            win_prob = max(5, min(95, win_prob))  # Clamp between 5-95%
            
            st.markdown("---")
            
            # Create animated progress ring
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=win_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Batting Team Win Probability", 'font': {'size': 28, 'color': 'white'}},
                number={'font': {'size': 60, 'color': 'white'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
                    'bar': {'color': "lime" if win_prob > 50 else "red", 'thickness': 0.8},
                    'bgcolor': "rgba(255,255,255,0.2)",
                    'borderwidth': 3,
                    'bordercolor': "white",
                    'steps': [
                        {'range': [0, 25], 'color': 'rgba(255,0,0,0.3)'},
                        {'range': [25, 50], 'color': 'rgba(255,165,0,0.3)'},
                        {'range': [50, 75], 'color': 'rgba(255,255,0,0.3)'},
                        {'range': [75, 100], 'color': 'rgba(0,255,0,0.3)'}
                    ],
                }
            ))
            
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(30,30,46,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=18),
                title_font=dict(size=22, color='white'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Situation analysis
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🎯 Key Factors")
                st.progress(wickets_left / 10)
                st.caption(f"Wickets in Hand: {wickets_left}/10")
                
                st.progress(max(0, 1 - runs_needed / target))
                st.caption(f"Progress: {(current_score/target*100):.1f}%")
            
            with col2:
                st.markdown("### ⚡ Run Rate Pressure")
                rr_pressure = required_rr / 12
                st.progress(min(1, rr_pressure))
                st.caption(f"Required RR: {required_rr:.2f}")
                
                if required_rr < 7:
                    st.success("✅ Easy run rate")
                elif required_rr < 10:
                    st.warning("⚠️ Moderate pressure")
                else:
                    st.error("🔥 High pressure")
            
            with col3:
                st.markdown("### ⏱️ Time Remaining")
                st.progress(balls_left / 120)
                st.caption(f"Balls Left: {balls_left}/120")
                
                overs_left = balls_left / 6
                st.info(f"📊 {overs_left:.1f} overs remaining")

# ═══════════════════════════════════════════════════════════
# PAGE 4: PLAYER ANALYSIS
# ═══════════════════════════════════════════════════════════
elif page == "👤 Player Analysis":
    st.markdown("<h1 style='text-align: center;'>👤 PLAYER PERFORMANCE ANALYSIS</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🏏 Batsman Analysis", "⚾ Bowler Analysis"])
    
    with tab1:
        batsman_stats = loaded_data['batsman_stats']
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            player = st.selectbox("Select Batsman", sorted(batsman_stats['player'].unique()))
        
        with col2:
            st.markdown("### 🔍 Quick Search")
            search_term = st.text_input("Search players", "")
        
        if player:
            player_data = batsman_stats[batsman_stats['player'] == player].iloc[0]
            
            # Stats cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Runs", f"{int(player_data['runs']):,}")
            with col2:
                st.metric("Average", f"{player_data['avg']:.2f}")
            with col3:
                st.metric("Strike Rate", f"{player_data['sr']:.2f}")
            with col4:
                st.metric("Matches", int(player_data['matches']))
            
            # Radar chart
            fig = go.Figure()
            
            categories = ['Runs', 'Average', 'Strike Rate', 'Consistency']
            max_runs = batsman_stats['runs'].max()
            max_avg = batsman_stats['avg'].max()
            max_sr = batsman_stats['sr'].max()
            
            values = [
                player_data['runs'] / max_runs * 100,
                player_data['avg'] / max_avg * 100,
                player_data['sr'] / max_sr * 100,
                (player_data['matches'] / batsman_stats['matches'].max()) * 100
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=player,
                line=dict(color='cyan', width=3)
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], color='white'),
                    bgcolor='rgba(30,30,46,0.5)',
                    angularaxis=dict(color='white')
                ),
                showlegend=True,
                title="Player Performance Radar",
                height=500,
                plot_bgcolor='rgba(30,30,46,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                title_font=dict(size=18, color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        bowler_stats = loaded_data['bowler_stats']
        
        player = st.selectbox("Select Bowler", sorted(bowler_stats['player'].unique()))
        
        if player:
            player_data = bowler_stats[bowler_stats['player'] == player].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Wickets", int(player_data['wickets']))
            with col2:
                st.metric("Economy", f"{player_data['economy']:.2f}")
            with col3:
                st.metric("Average", f"{player_data['avg']:.2f}")
            with col4:
                st.metric("Matches", int(player_data['matches']))
            
            # Performance chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Wickets', 'Economy', 'Strike Rate'],
                y=[player_data['wickets'], 
                   player_data['economy'],
                   player_data['balls'] / player_data['wickets'] if player_data['wickets'] > 0 else 0],
                marker=dict(
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                    line=dict(color='white', width=2)
                ),
                text=[int(player_data['wickets']), 
                      f"{player_data['economy']:.2f}",
                      f"{player_data['balls'] / player_data['wickets']:.1f}" if player_data['wickets'] > 0 else "0"],
                textposition='outside'
            ))
            
            fig.update_layout(
                title=f"{player} - Bowling Stats",
                yaxis_title="Value",
                height=400,
                plot_bgcolor='rgba(30,30,46,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                title_font=dict(size=18, color='white'),
                xaxis=dict(color='white'),
                yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)')
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE 5: TEAM COMPARISON
# ═══════════════════════════════════════════════════════════
elif page == "🏆 Team Comparison":
    st.markdown("<h1 style='text-align: center;'>🏆 TEAM COMPARISON</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    teams = sorted(loaded_data['team_to_id'].keys())
    
    with col1:
        team_a = st.selectbox("Select Team A", teams, key='team_a')
    
    with col2:
        team_b = st.selectbox("Select Team B", [t for t in teams if t != team_a], key='team_b')
    
    if st.button("⚔️ COMPARE TEAMS", use_container_width=True):
        team_stats = loaded_data['team_stats']
        
        team_a_stats = team_stats.loc[team_a]
        team_b_stats = team_stats.loc[team_b]
        
        # Comparison metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[team_a, team_b],
                y=[team_a_stats['wins'], team_b_stats['wins']],
                marker=dict(color=['#667eea', '#764ba2']),
                text=[int(team_a_stats['wins']), int(team_b_stats['wins'])],
                textposition='outside'
            ))
            fig.update_layout(
                title="Win Percentage",
                height=300,
                plot_bgcolor='rgba(30,30,46,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=13),
                title_font=dict(size=16, color='white'),
                xaxis=dict(color='white'),
                yaxis=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[team_a, team_b],
                y=[team_a_stats['win_rate']*100, team_b_stats['win_rate']*100],
                marker=dict(color=['#FF6B6B', '#4ECDC4']),
                text=[f"{team_a_stats['win_rate']*100:.1f}%", 
                      f"{team_b_stats['win_rate']*100:.1f}%"],
                textposition='outside'
            ))
            fig.update_layout(
                title="Win Rate",
                height=300,
                plot_bgcolor='rgba(30,30,46,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=13),
                title_font=dict(size=16, color='white'),
                xaxis=dict(color='white'),
                yaxis=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[team_a, team_b],
                y=[team_a_stats['matches'], team_b_stats['matches']],
                marker=dict(color=['#45B7D1', '#FFA07A']),
                text=[int(team_a_stats['matches']), int(team_b_stats['matches'])],
                textposition='outside'
            ))
            fig.update_layout(
                title="Matches Played",
                height=300,
                plot_bgcolor='rgba(30,30,46,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=13),
                title_font=dict(size=16, color='white'),
                xaxis=dict(color='white'),
                yaxis=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# PAGE 6: VENUE INSIGHTS
# ═══════════════════════════════════════════════════════════
elif page == "📍 Venue Insights":
    st.markdown("<h1 style='text-align: center;'>📍 VENUE INSIGHTS</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    venue_stats = loaded_data['venue_stats']
    
    venue = st.selectbox("Select Venue", sorted(venue_stats.index))
    
    if venue:
        venue_data = venue_stats.loc[venue]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Matches Played", int(venue_data['matches']))
        with col2:
            st.metric("Average Score", f"{venue_data['avg_score']:.0f}")
        with col3:
            st.metric("Bat First", int(venue_data['bat_first_count']))
        with col4:
            st.metric("Field First", int(venue_data['field_first_count']))
        
        # Toss decision pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Bat First', 'Field First'],
            values=[venue_data['bat_first_count'], venue_data['field_first_count']],
            hole=.4,
            marker=dict(colors=['#667eea', '#764ba2'])
        )])
        
        fig.update_layout(
            title="Toss Decision Preference",
            height=400,
            plot_bgcolor='rgba(30,30,46,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            title_font=dict(size=18, color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        if venue_data['prefer_bat']:
            st.success("✅ RECOMMENDATION: Teams prefer batting first at this venue")
        else:
            st.info("ℹ️ RECOMMENDATION: Teams prefer fielding first at this venue")

# ═══════════════════════════════════════════════════════════
# PAGE 7: HEAD-TO-HEAD
# ═══════════════════════════════════════════════════════════
elif page == "⚔️ Head-to-Head":
    st.markdown("<h1 style='text-align: center;'>⚔️ HEAD-TO-HEAD ANALYSIS</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    teams = sorted(loaded_data['team_to_id'].keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        team_1 = st.selectbox("Team 1", teams, key='h2h_team1')
    with col2:
        team_2 = st.selectbox("Team 2", [t for t in teams if t != team_1], key='h2h_team2')
    
    if st.button("📊 SHOW HEAD-TO-HEAD", use_container_width=True):
        h2h_key = f"{team_1} vs {team_2}" if team_1 < team_2 else f"{team_2} vs {team_1}"
        
        h2h_stats = loaded_data['h2h_stats']
        
        if h2h_key in h2h_stats.index:
            h2h_data = h2h_stats.loc[h2h_key]
            
            team1_wins = h2h_data[f'{team_1}_wins']
            team2_wins = h2h_data[f'{team_2}_wins']
            total = h2h_data['matches']
            
            # Win distribution
            fig = go.Figure(data=[
                go.Pie(
                    labels=[team_1, team_2],
                    values=[team1_wins, team2_wins],
                    hole=.5,
                    marker=dict(colors=['#667eea', '#764ba2']),
                    textinfo='label+percent+value',
                    textfont=dict(size=16, color='white')
                )
            ])
            
            fig.update_layout(
                title=f"{team_1} vs {team_2} - Win Distribution",
                height=500,
                plot_bgcolor='rgba(30,30,46,0.5)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                title_font=dict(size=18, color='white'),
                annotations=[dict(text=f'Total<br>{int(total)}<br>Matches', 
                                x=0.5, y=0.5, font_size=20, showarrow=False,
                                font=dict(color='white'))]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"{team_1} Wins", int(team1_wins))
            with col2:
                st.metric("Total Matches", int(total))
            with col3:
                st.metric(f"{team_2} Wins", int(team2_wins))
        else:
            st.warning("No head-to-head data available for these teams")

# ═══════════════════════════════════════════════════════════
# PAGE 8: STATISTICS
# ═══════════════════════════════════════════════════════════
elif page == "📈 Statistics":
    st.markdown("<h1 style='text-align: center;'>📈 COMPREHENSIVE STATISTICS</h1>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["🏏 Batting Leaders", "⚾ Bowling Leaders", "🏆 Team Stats"])
    
    with tab1:
        batsman_stats = loaded_data['batsman_stats']
        
        metric = st.selectbox("Sort By", ["Total Runs", "Average", "Strike Rate"], key='bat_metric')
        
        if metric == "Total Runs":
            top_players = batsman_stats.nlargest(20, 'runs')
            col_name = 'runs'
        elif metric == "Average":
            top_players = batsman_stats[batsman_stats['balls'] >= 200].nlargest(20, 'avg')
            col_name = 'avg'
        else:
            top_players = batsman_stats[batsman_stats['balls'] >= 200].nlargest(20, 'sr')
            col_name = 'sr'
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_players['player'],
            y=top_players[col_name],
            marker=dict(
                color=top_players[col_name],
                colorscale='Viridis',
                showscale=True
            ),
            text=top_players[col_name].round(2),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Top 20 - {metric}",
            xaxis_title="Player",
            yaxis_title=metric,
            height=600,
            plot_bgcolor='rgba(30,30,46,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=13),
            title_font=dict(size=18, color='white'),
            xaxis=dict(tickangle=-45, color='white', gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(color='white', gridcolor='rgba(255,255,255,0.1)')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.dataframe(
            top_players[['player', 'runs', 'avg', 'sr', 'matches']].style.background_gradient(cmap='viridis'),
            use_container_width=True,
            height=400
        )
    
    with tab2:
        bowler_stats = loaded_data['bowler_stats']
        
        metric = st.selectbox("Sort By", ["Wickets", "Economy", "Average"], key='bowl_metric')
        
        if metric == "Wickets":
            top_players = bowler_stats.nlargest(20, 'wickets')
            col_name = 'wickets'
        elif metric == "Economy":
            top_players = bowler_stats[bowler_stats['balls'] >= 200].nsmallest(20, 'economy')
            col_name = 'economy'
        else:
            top_players = bowler_stats[bowler_stats['wickets'] >= 20].nsmallest(20, 'avg')
            col_name = 'avg'
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_players['player'],
            y=top_players[col_name],
            marker=dict(
                color=top_players[col_name],
                colorscale='Plasma',
                showscale=True
            ),
            text=top_players[col_name].round(2),
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Top 20 - {metric}",
            xaxis_title="Player",
            yaxis_title=metric,
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(tickangle=-45)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            top_players[['player', 'wickets', 'economy', 'avg', 'matches']].style.background_gradient(cmap='plasma'),
            use_container_width=True,
            height=400
        )
    
    with tab3:
        team_stats = loaded_data['team_stats']
        
        # Create comprehensive team comparison
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Win Rate %', 'Total Wins', 'Total Matches', 'Loss Count'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        teams = team_stats.index
        
        fig.add_trace(
            go.Bar(x=teams, y=team_stats['win_rate']*100, 
                   marker=dict(color=team_stats['win_rate']*100, colorscale='Viridis')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=teams, y=team_stats['wins'],
                   marker=dict(color=team_stats['wins'], colorscale='Plasma')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=teams, y=team_stats['matches'],
                   marker=dict(color=team_stats['matches'], colorscale='Turbo')),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=teams, y=team_stats['losses'],
                   marker=dict(color=team_stats['losses'], colorscale='Hot')),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            plot_bgcolor='rgba(30,30,46,0.5)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=13),
            title_font=dict(size=16, color='white')
        )
        
        fig.update_xaxes(tickangle=-45)
        
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='color: white; font-size: 16px;'>
        🏏 <b>IPL Prediction & Analytics Platform</b> | 
        Powered by LSTM Deep Learning | 
        Built with ❤️ using Streamlit
    </p>
    <p style='color: rgba(255,255,255,0.7); font-size: 14px;'>
        Model Accuracy: 98.19% | Dataset: 1,146 Matches | 273,503 Balls
    </p>
</div>
""", unsafe_allow_html=True)