import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Cricket Analytics Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(to bottom right, #1a1a2e, #16213e, #0f3460);
    }
    h1, h2, h3 {
        color: #00d4ff;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .prediction-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #00a8ff);
        color: white;
        border: none;
        padding: 10px 30px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,212,255,0.4);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
class MatchWinnerModel(nn.Module):
    def __init__(self, input_dim):
        super(MatchWinnerModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class BallByBallLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(BallByBallLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc_layers(last_output)


# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================
@st.cache_resource
def load_models_and_data():
    device = torch.device('cpu')
    
    try:
        # Load match winner model
        match_checkpoint = torch.load('models/match_winner_model.pth', 
                                      map_location=device, 
                                      weights_only=False)
        match_model = MatchWinnerModel(input_dim=10)
        match_model.load_state_dict(match_checkpoint['model_state'])
        match_model.eval()
        match_scaler = match_checkpoint['scaler']
        
        # Load ball-by-ball model
        ball_checkpoint = torch.load('models/ball_by_ball_model.pth', 
                                     map_location=device,
                                     weights_only=False)
        ball_model = BallByBallLSTM(input_dim=9)
        ball_model.load_state_dict(ball_checkpoint['model_state'])
        ball_model.eval()
        ball_scaler = ball_checkpoint['scaler']
        
        # Load metadata
        with open('models/cricket_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load advanced features
        try:
            with open('models/advanced_features.pkl', 'rb') as f:
                advanced_data = pickle.load(f)
        except:
            advanced_data = None
        
        return match_model, ball_model, match_scaler, ball_scaler, metadata, advanced_data, device
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure all model files are present in the 'models/' directory")
        return None, None, None, None, None, None, None


# Initialize
match_model, ball_model, match_scaler, ball_scaler, metadata, advanced_data, device = load_models_and_data()

if metadata:
    team_encoder = metadata['team_encoder']
    team_decoder = metadata['team_decoder']
    venue_encoder = metadata['venue_encoder']
    team_stats = metadata['team_stats']
    venue_stats = metadata['venue_stats']
    
    teams_list = sorted(team_encoder.keys())
    venues_list = sorted(venue_encoder.keys())

if advanced_data:
    powerplay_stats = advanced_data.get('powerplay_stats', {})
    death_bowler_stats = advanced_data.get('death_bowler_stats', [])
    player_history = advanced_data.get('player_history', {})
    venue_detailed_stats = advanced_data.get('venue_detailed_stats', {})
    fantasy_players = advanced_data.get('fantasy_players', [])
    toss_impact_stats = advanced_data.get('toss_impact_stats', {})


# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("🏏 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "🎯 Match Predictor", "📊 Live Win Probability", 
     "📈 Team Analytics", "🏟️ Venue Analysis", 
     "⚡ Powerplay Analysis", "💀 Death Overs Specialists",
     "🎯 Player Score Predictor", "📊 Player Form Tracker",
     "👥 Fantasy Team Builder", "🎮 Impact Analysis"]
)

st.sidebar.markdown("---")
st.sidebar.info("**Cricket Analytics Platform** v2.0\n\nPowered by PyTorch & Streamlit")


# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.title("🏏 Cricket Analytics Platform")
    st.markdown("### Your Ultimate Cricket Prediction & Analysis Tool")
    
    # Hero metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="prediction-box">
            <h2 style="text-align: center; color: #00d4ff;">🎯</h2>
            <h3 style="text-align: center; color: white;">Match Predictor</h3>
            <p style="text-align: center; color: #aaa;">Predict winner before match</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="prediction-box">
            <h2 style="text-align: center; color: #00d4ff;">📊</h2>
            <h3 style="text-align: center; color: white;">Live Probability</h3>
            <p style="text-align: center; color: #aaa;">Real-time win chances</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="prediction-box">
            <h2 style="text-align: center; color: #00d4ff;">📈</h2>
            <h3 style="text-align: center; color: white;">Team Analytics</h3>
            <p style="text-align: center; color: #aaa;">Deep statistical insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="prediction-box">
            <h2 style="text-align: center; color: #00d4ff;">🏟️</h2>
            <h3 style="text-align: center; color: white;">Venue Analysis</h3>
            <p style="text-align: center; color: #aaa;">Ground characteristics</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Key Features")
        st.markdown("""
        - ✅ **AI-Powered Predictions** using Deep Learning
        - ✅ **Real-time Win Probability** during matches
        - ✅ **Comprehensive Team Statistics**
        - ✅ **Venue Impact Analysis**
        - ✅ **Head-to-Head Records**
        - ✅ **Toss Impact Evaluation**
        """)
    
    with col2:
        st.markdown("### 🤖 Technology Stack")
        st.markdown("""
        - **PyTorch** - Deep Learning Framework
        - **LSTM Networks** - Sequential Prediction
        - **Streamlit** - Interactive Dashboard
        - **Plotly** - Dynamic Visualizations
        - **Pandas** - Data Processing
        """)
    
    st.markdown("---")
    st.success("👈 Use the sidebar to navigate between different features!")


# ============================================================================
# MATCH PREDICTOR PAGE
# ============================================================================
elif page == "🎯 Match Predictor":
    st.title("🎯 Match Winner Predictor")
    st.markdown("### Predict the match winner before the game starts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚔️ Teams")
        team1 = st.selectbox("Team 1", teams_list, key="team1")
        team2 = st.selectbox("Team 2", [t for t in teams_list if t != team1], key="team2")
    
    with col2:
        st.markdown("#### 🏟️ Match Details")
        venue = st.selectbox("Venue", venues_list)
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
    
    if st.button("🔮 Predict Winner", key="predict_match"):
        with st.spinner("Analyzing match conditions..."):
            # Prepare features
            team1_id = team_encoder.get(team1, 0)
            team2_id = team_encoder.get(team2, 0)
            venue_id = venue_encoder.get(venue, 0)
            city_id = 0
            toss_winner_id = team_encoder.get(toss_winner, 0)
            toss_bat = 1 if toss_decision == 'bat' else 0
            team1_won_toss = 1 if toss_winner == team1 else 0
            
            team1_wr = team_stats.get(team1, {}).get('win_rate', 0.5)
            team2_wr = team_stats.get(team2, {}).get('win_rate', 0.5)
            venue_avg = venue_stats.get(venue, {}).get('avg_first_innings_score', 160)
            
            features = np.array([[
                team1_id, team2_id, venue_id, city_id, toss_winner_id,
                toss_bat, team1_won_toss, team1_wr, team2_wr, venue_avg
            ]])
            
            features_scaled = match_scaler.transform(features)
            features_tensor = torch.FloatTensor(features_scaled)
            
            with torch.no_grad():
                prediction = match_model(features_tensor)
                team1_prob = prediction.item() * 100
                team2_prob = 100 - team1_prob
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Prediction Results")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Win probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=team1_prob,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{team1} Win Probability", 'font': {'size': 24}},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "lightgreen" if team1_prob > 50 else "lightcoral"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.3)'},
                            {'range': [50, 100], 'color': 'rgba(0, 255, 0, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white", 'family': "Arial"},
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color: {'#00ff00' if team1_prob > 50 else '#ff6b6b'};">{team1}</h3>
                    <h1 style="color: white; font-size: 48px;">{team1_prob:.1f}%</h1>
                    <p style="color: #aaa;">Win Probability</p>
                    <p style="color: white;">Win Rate: {team1_wr*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3 style="color: {'#00ff00' if team2_prob > 50 else '#ff6b6b'};">{team2}</h3>
                    <h1 style="color: white; font-size: 48px;">{team2_prob:.1f}%</h1>
                    <p style="color: #aaa;">Win Probability</p>
                    <p style="color: white;">Win Rate: {team2_wr*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Winner announcement
            winner = team1 if team1_prob > 50 else team2
            confidence = max(team1_prob, team2_prob)
            
            if confidence > 70:
                confidence_text = "High Confidence"
                color = "#00ff00"
            elif confidence > 60:
                confidence_text = "Moderate Confidence"
                color = "#ffaa00"
            else:
                confidence_text = "Low Confidence"
                color = "#ff6b6b"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: rgba(255,255,255,0.05); border-radius: 15px; margin-top: 20px;">
                <h2 style="color: {color};">🏆 Predicted Winner: {winner}</h2>
                <p style="color: #aaa; font-size: 18px;">{confidence_text} ({confidence:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# LIVE WIN PROBABILITY PAGE
# ============================================================================
elif page == "📊 Live Win Probability":
    st.title("📊 Live Win Probability Calculator")
    st.markdown("### Calculate real-time win probability during a chase")
    
    st.info("Enter the current match situation to calculate win probability for the batting team")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📈 Current Score")
        current_runs = st.number_input("Runs Scored", min_value=0, max_value=400, value=100)
        wickets_lost = st.number_input("Wickets Lost", min_value=0, max_value=10, value=3)
        
    with col2:
        st.markdown("#### ⏱️ Match Progress")
        overs_bowled = st.number_input("Overs Bowled", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        balls_bowled = int(overs_bowled * 6)
        
    with col3:
        st.markdown("#### 🎯 Target")
        target = st.number_input("Target", min_value=1, max_value=400, value=180)
        
    if st.button("📊 Calculate Win Probability", key="calc_live"):
        # Calculate match situation
        runs_needed = target - current_runs
        balls_remaining = 120 - balls_bowled
        wickets_remaining = 10 - wickets_lost
        
        if balls_bowled > 0:
            current_rr = (current_runs / balls_bowled) * 6
        else:
            current_rr = 0
        
        if balls_remaining > 0:
            required_rr = (runs_needed / balls_remaining) * 6
        else:
            required_rr = 0
        
        is_powerplay = 1 if overs_bowled < 6 else 0
        
        # Create dummy sequence (simplified for demo)
        # In production, you'd use actual ball-by-ball data
        sequence = []
        for i in range(30):
            ball_data = [
                np.random.randint(0, 7),  # runs_scored
                np.random.randint(0, 2),  # is_wicket
                current_runs * (i+1) / 30,  # cumulative_runs
                wickets_lost * (i+1) / 30,  # cumulative_wickets
                current_rr,
                required_rr,
                is_powerplay,
                balls_remaining,
                wickets_remaining
            ]
            sequence.append(ball_data)
        
        sequence = np.array([sequence], dtype=np.float32)
        sequence_reshaped = sequence.reshape(-1, sequence.shape[-1])
        sequence_scaled = ball_scaler.transform(sequence_reshaped)
        sequence_scaled = sequence_scaled.reshape(sequence.shape)
        sequence_tensor = torch.FloatTensor(sequence_scaled)
        
        with torch.no_grad():
            prediction = ball_model(sequence_tensor)
            win_prob = prediction.item() * 100
        
        # Display results
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Win probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=win_prob,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Batting Team Win Probability", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightcoral"},
                        {'range': [30, 70], 'color': "lightyellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Match Situation")
            st.metric("Runs Needed", runs_needed)
            st.metric("Balls Remaining", balls_remaining)
            st.metric("Wickets Left", wickets_remaining)
            st.metric("Current RR", f"{current_rr:.2f}")
            st.metric("Required RR", f"{required_rr:.2f}")
        
        # Comparison chart
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=['Batting Team', 'Bowling Team'],
            y=[win_prob, 100-win_prob],
            marker_color=['#00d4ff', '#ff6b6b'],
            text=[f'{win_prob:.1f}%', f'{100-win_prob:.1f}%'],
            textposition='auto',
        ))
        
        fig2.update_layout(
            title="Win Probability Comparison",
            yaxis_title="Probability (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# TEAM ANALYTICS PAGE
# ============================================================================
elif page == "📈 Team Analytics":
    st.title("📈 Team Analytics Dashboard")
    st.markdown("### Comprehensive team performance analysis")
    
    selected_team = st.selectbox("Select Team", teams_list)
    
    if selected_team and selected_team in team_stats:
        stats = team_stats[selected_team]
        
        # Team metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Matches Played", stats['matches_played'])
        with col2:
            st.metric("Wins", stats['wins'])
        with col3:
            st.metric("Losses", stats['matches_played'] - stats['wins'])
        with col4:
            st.metric("Win Rate", f"{stats['win_rate']*100:.1f}%")
        
        st.markdown("---")
        
        # Win rate visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stats['win_rate'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Team Win Rate", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightcoral"},
                    {'range': [40, 60], 'color': "lightyellow"},
                    {'range': [60, 100], 'color': "lightgreen"}
                ]
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Wins vs Losses pie chart
        fig2 = go.Figure(data=[go.Pie(
            labels=['Wins', 'Losses'],
            values=[stats['wins'], stats['matches_played'] - stats['wins']],
            hole=.3,
            marker_colors=['#00ff00', '#ff6b6b']
        )])
        
        fig2.update_layout(
            title="Win/Loss Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# VENUE ANALYSIS PAGE
# ============================================================================
elif page == "🏟️ Venue Analysis":
    st.title("🏟️ Venue Analysis")
    st.markdown("### Understand how different venues impact match outcomes")
    
    selected_venue = st.selectbox("Select Venue", venues_list)
    
    if selected_venue and selected_venue in venue_stats:
        stats = venue_stats[selected_venue]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Matches Played", stats['matches'])
        with col2:
            st.metric("Avg First Innings Score", f"{stats['avg_first_innings_score']:.0f}")
        with col3:
            preference = "Bat First" if stats['bat_first_preference'] else "Field First"
            st.metric("Toss Preference", preference)
        
        st.markdown("---")
        
        # Average score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stats['avg_first_innings_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average First Innings Score", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [120, 220]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [120, 150], 'color': "lightcoral"},
                    {'range': [150, 180], 'color': "lightyellow"},
                    {'range': [180, 220], 'color': "lightgreen"}
                ]
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# ADVANCED FEATURES PAGE
# ============================================================================
elif page == "🎮 Advanced Features":
    st.title("🎮 Advanced Analytics Features")
    st.markdown("### Coming Soon!")
    
    st.info("These advanced features are currently in development:")
    
    features = [
        ("⚡ Powerplay Analysis", "Analyze powerplay performances"),
        ("💀 Death Overs Specialists", "Identify clutch players"),
        ("🎯 Player Score Prediction", "Predict individual scores"),
        ("📊 Player Form Trends", "Time series form analysis"),
        ("🌤️ Weather Impact", "Weather condition effects"),
        ("🏏 Pitch Reports", "Pitch type analysis"),
        ("👥 Fantasy Team Builder", "Optimal team selection"),
        ("💰 Betting Odds Analysis", "Value bet identification")
    ]
    
    col1, col2 = st.columns(2)
    
    for i, (title, desc) in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="prediction-box">
                <h3>{title}</h3>
                <p style="color: #aaa;">{desc}</p>
                <p style="color: #ffaa00;">🔨 Under Development</p>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# POWERPLAY ANALYSIS PAGE
# ============================================================================
elif page == "⚡ Powerplay Analysis":
    st.title("⚡ Powerplay Analysis")
    st.markdown("### First 6 overs performance breakdown")
    
    if advanced_data and powerplay_stats:
        # Load powerplay data
        try:
            powerplay_df = pd.read_csv('analysis/powerplay_analysis.csv', index_col=0)
            
            st.markdown("### 🏆 Top Powerplay Performers")
            
            col1, col2, col3 = st.columns(3)
            
            top_team = powerplay_df.index[0]
            with col1:
                st.metric("Best Team", top_team)
            with col2:
                st.metric("Avg PP Score", f"{powerplay_df.iloc[0]['avg_powerplay_score']:.1f}")
            with col3:
                st.metric("PP Run Rate", f"{powerplay_df.iloc[0]['powerplay_run_rate']:.2f}")
            
            # Powerplay comparison chart
            fig = px.bar(powerplay_df.head(10).reset_index(), 
                        x='index', 
                        y='powerplay_run_rate',
                        title="Top 10 Teams - Powerplay Run Rate",
                        labels={'index': 'Team', 'powerplay_run_rate': 'Run Rate'},
                        color='powerplay_run_rate',
                        color_continuous_scale='blues')
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.markdown("### 📊 Detailed Powerplay Statistics")
            st.dataframe(powerplay_df.head(15).style.background_gradient(cmap='Blues'))
            
        except:
            st.warning("Powerplay analysis data not found. Please run the advanced features notebook.")
    else:
        st.warning("Advanced features not loaded. Please run the advanced features analysis first.")


# ============================================================================
# DEATH OVERS SPECIALISTS PAGE
# ============================================================================
elif page == "💀 Death Overs Specialists":
    st.title("💀 Death Overs Specialists")
    st.markdown("### Best bowlers in overs 16-20")
    
    if advanced_data and death_bowler_stats:
        try:
            death_df = pd.read_csv('analysis/death_overs_specialists.csv')
            
            st.markdown("### 🎯 Top Death Bowlers")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best Bowler", death_df.iloc[0]['player'])
            with col2:
                st.metric("Death Economy", f"{death_df.iloc[0]['death_economy']:.2f}")
            with col3:
                st.metric("Wickets", int(death_df.iloc[0]['wickets']))
            
            # Death bowling comparison
            fig = go.Figure()
            
            top_10 = death_df.head(10)
            fig.add_trace(go.Bar(
                x=top_10['player'],
                y=top_10['death_economy'],
                name='Economy',
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title="Top 10 Death Bowlers - Economy Rate",
                xaxis_title="Bowler",
                yaxis_title="Economy Rate",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Wickets vs Economy scatter
            fig2 = px.scatter(death_df.head(20), 
                            x='death_economy', 
                            y='wickets',
                            size='balls',
                            hover_data=['player'],
                            title="Death Bowling: Economy vs Wickets",
                            labels={'death_economy': 'Economy Rate', 'wickets': 'Wickets'},
                            color='death_economy',
                            color_continuous_scale='RdYlGn_r')
            
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"}
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Table
            st.markdown("### 📊 Detailed Statistics")
            st.dataframe(death_df.head(20))
            
        except:
            st.warning("Death overs data not found. Please run the advanced features notebook.")
    else:
        st.warning("Advanced features not loaded.")


# ============================================================================
# PLAYER SCORE PREDICTOR PAGE
# ============================================================================
elif page == "🎯 Player Score Predictor":
    st.title("🎯 Player Score Predictor")
    st.markdown("### Predict individual player performance")
    
    if advanced_data and player_history:
        players_list = sorted(player_history.keys())
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_player = st.selectbox("Select Player", players_list)
        
        with col2:
            selected_venue = st.selectbox("Select Venue", venues_list, key="player_venue")
        
        if st.button("🔮 Predict Player Score"):
            if selected_player in player_history:
                player_data = player_history[selected_player]
                
                # Simple prediction
                base_score = player_data['avg_runs']
                predicted_runs = int(base_score * 0.9)
                predicted_balls = int(player_data['avg_balls'])
                predicted_sr = round((predicted_runs / predicted_balls * 100), 1) if predicted_balls > 0 else 0
                
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Predicted Runs", predicted_runs)
                with col2:
                    st.metric("Predicted Balls", predicted_balls)
                with col3:
                    st.metric("Strike Rate", f"{predicted_sr}")
                with col4:
                    st.metric("Historical Avg", f"{player_data['avg_runs']:.1f}")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_runs,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{selected_player} - Expected Runs"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 20], 'color': "lightgray"},
                            {'range': [20, 40], 'color': "lightyellow"},
                            {'range': [40, 100], 'color': "lightgreen"}
                        ]
                    }
                ))
                
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "white"},
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Performance breakdown
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>📈 Performance Analysis</h3>
                    <p><strong>Matches Played:</strong> {player_data['matches']}</p>
                    <p><strong>Average Runs:</strong> {player_data['avg_runs']:.1f}</p>
                    <p><strong>Average Balls:</strong> {player_data['avg_balls']:.1f}</p>
                    <p><strong>Consistency:</strong> {'High' if player_data['matches'] > 20 else 'Moderate' if player_data['matches'] > 10 else 'Low'}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Player data not loaded.")


# ============================================================================
# PLAYER FORM TRACKER PAGE
# ============================================================================
elif page == "📊 Player Form Tracker":
    st.title("📊 Player Form Tracker")
    st.markdown("### Track player performance trends")
    
    try:
        form_df = pd.read_csv('analysis/player_form_analysis.csv', index_col=0)
        
        players_with_form = list(form_df.index)
        selected_player = st.selectbox("Select Player", players_with_form)
        
        if selected_player:
            player_form = form_df.loc[selected_player]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Form", player_form['form'])
            with col2:
                st.metric("Last 5 Avg", f"{player_form['last_5_avg']:.1f}")
            with col3:
                st.metric("Last 10 Avg", f"{player_form['last_10_avg']:.1f}")
            with col4:
                trend_icon = "📈" if player_form['trend'] == 'Up' else "📉"
                st.metric("Trend", f"{trend_icon} {player_form['trend']}")
            
            # Form indicator
            form_color = {
                'Excellent': '#00ff00',
                'Good': '#90EE90',
                'Average': '#FFD700',
                'Poor': '#ff6b6b'
            }
            
            color = form_color.get(player_form['form'], '#aaa')
            
            st.markdown(f"""
            <div class="prediction-box" style="border-left: 5px solid {color};">
                <h3 style="color: {color};">Form Analysis: {player_form['form']}</h3>
                <p>This player is currently in <strong>{player_form['form'].lower()}</strong> form.</p>
                <p>Recent performance shows a <strong>{player_form['trend'].lower()}ward</strong> trend.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Form comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Last 5', 'Last 10', 'Overall'],
                y=[player_form['last_5_avg'], player_form['last_10_avg'], player_form['overall_avg']],
                marker_color=['#00d4ff', '#764ba2', '#667eea']
            ))
            
            fig.update_layout(
                title=f"{selected_player} - Performance Comparison",
                yaxis_title="Average Runs",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
    except:
        st.warning("Player form data not found. Please run the advanced features notebook.")


# ============================================================================
# FANTASY TEAM BUILDER PAGE
# ============================================================================
elif page == "👥 Fantasy Team Builder":
    st.title("👥 Fantasy Team Builder")
    st.markdown("### Build your optimal fantasy cricket team")
    
    try:
        fantasy_df = pd.read_csv('analysis/fantasy_recommendations.csv')
        
        st.markdown("### 🌟 Top Fantasy Picks")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            player_type = st.selectbox("Player Type", ["All", "Batsman", "All-Rounder"])
        
        with col2:
            min_matches = st.slider("Minimum Matches", 0, 50, 10)
        
        # Filter data
        filtered_df = fantasy_df[fantasy_df['matches'] >= min_matches]
        if player_type != "All":
            filtered_df = filtered_df[filtered_df['type'] == player_type]
        
        # Top picks
        top_picks = filtered_df.head(11)
        
        # Display cards
        cols = st.columns(3)
        for idx, (i, player) in enumerate(top_picks.iterrows()):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>{player['player']}</h4>
                    <p style="color: #00d4ff;">{player['type']}</p>
                    <h2 style="color: #00ff00;">{player['expected_points']:.1f}</h2>
                    <p style="color: #aaa;">Expected Points</p>
                    <p>Avg Runs: {player['avg_runs']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Points distribution
        fig = px.bar(filtered_df.head(15), 
                    x='player', 
                    y='expected_points',
                    color='type',
                    title="Top 15 Fantasy Players - Expected Points",
                    labels={'expected_points': 'Expected Fantasy Points'})
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Full table
        st.markdown("### 📊 Complete Fantasy Rankings")
        st.dataframe(filtered_df.head(50))
        
    except:
        st.warning("Fantasy data not found. Please run the advanced features notebook.")


# ============================================================================
# IMPACT ANALYSIS PAGE
# ============================================================================
elif page == "🎮 Impact Analysis":
    st.title("🎮 Match Impact Analysis")
    st.markdown("### Understand key factors affecting match outcomes")
    
    try:
        toss_impact_df = pd.read_csv('analysis/toss_impact_analysis.csv', index_col=0)
        venue_detailed_df = pd.read_csv('analysis/venue_detailed_analysis.csv', index_col=0)
        
        # Toss Impact
        st.markdown("### 🎯 Toss Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Toss advantage chart
            fig = px.bar(toss_impact_df.head(10).reset_index(),
                        x='index',
                        y='toss_advantage',
                        title="Top 10 Teams - Toss Win to Match Win %",
                        labels={'index': 'Team', 'toss_advantage': 'Win % After Toss Win'},
                        color='toss_advantage',
                        color_continuous_scale='RdYlGn')
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average toss advantage
            avg_advantage = toss_impact_df['toss_advantage'].mean()
            
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_advantage,
                title={'text': "Average Toss Advantage %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "white"},
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        # Venue Impact
        st.markdown("### 🏟️ Venue Impact Analysis")
        
        # Chase success by venue
        fig3 = px.bar(venue_detailed_df.head(10).reset_index(),
                     x='index',
                     y='chase_success_rate',
                     title="Top 10 Venues - Chase Success Rate",
                     labels={'index': 'Venue', 'chase_success_rate': 'Chase Success %'},
                     color='nature',
                     color_discrete_map={
                         'Batting Friendly': '#00ff00',
                         'Balanced': '#FFD700',
                         'Bowling Friendly': '#ff6b6b'
                     })
        
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Venue nature distribution
        nature_counts = venue_detailed_df['nature'].value_counts()
        
        fig4 = go.Figure(data=[go.Pie(
            labels=nature_counts.index,
            values=nature_counts.values,
            hole=.3,
            marker_colors=['#00ff00', '#FFD700', '#ff6b6b']
        )])
        
        fig4.update_layout(
            title="Venue Nature Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )
        
        st.plotly_chart(fig4, use_container_width=True)
        
    except:
        st.warning("Impact analysis data not found. Please run the advanced features notebook.")


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #aaa; padding: 20px;">
    <p>🏏 Cricket Analytics Platform | Powered by PyTorch & Streamlit</p>
    <p>© 2024 | Made with ❤️ for Cricket Fans</p>
</div>
""", unsafe_allow_html=True)