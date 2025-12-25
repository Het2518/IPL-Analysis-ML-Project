import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🏏 IPL ML Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    h1, h2, h3, h4 {
        color: white;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin: 25px 0;
        background: linear-gradient(45deg, #ff6b35, #f7931e);
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        animation: fadeIn 0.5s ease-in;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #ff6b35, #f7931e);
        color: white;
        border: none;
        padding: 15px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LSTM MODEL DEFINITION
# ============================================================
class WinLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sig(self.fc(out[:, -1]))

# ============================================================
# LOAD ALL RESOURCES
# ============================================================
@st.cache_resource
def load_resources():
    try:
        # Load datasets
        matches = pd.read_csv("matches_for_task1.csv")
        players = pd.read_csv("players_for_task2.csv")
        balls = pd.read_csv("balls_for_task3.csv")

        # Load Task 1 model (Random Forest)
        rf_model = joblib.load("saved_models/random_forest_task1.pkl")
        
        # Load Task 3 components
        scaler = joblib.load("saved_models/task3_scaler.pkl")

        # Create and fit Label Encoder for teams
        le_team = LabelEncoder()
        all_teams = pd.unique(matches[['team1', 'team2']].values.ravel())
        le_team.fit(all_teams)

        # Calculate batsman statistics
        batsman_runs = players.groupby(['match_id', 'batter'])['runs'].sum()
        batsman_avg = batsman_runs.groupby('batter').agg(['mean', 'std', 'count']).round(1)
        batsman_avg.columns = ['avg_runs', 'std_runs', 'matches_played']
        batsman_avg = batsman_avg.sort_values('avg_runs', ascending=False)

        # Load LSTM model
        lstm_model = WinLSTM()
        lstm_model.load_state_dict(
            torch.load("saved_models/winlstm_task3_state.pt", map_location="cpu")
        )
        lstm_model.eval()

        return rf_model, le_team, batsman_avg, lstm_model, scaler, list(all_teams), matches, players

    except Exception as e:
        st.error(f"❌ Error loading resources: {str(e)}")
        st.info("Please ensure all model files and datasets are in the correct directories.")
        st.stop()

# Load all resources
rf_model, le_team, batsman_avg, lstm_model, scaler, teams, matches_df, players_df = load_resources()

# ============================================================
# SIDEBAR - APP INFO
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/cricket.png", width=100)
    st.title("🏏 IPL ML Predictor")
    st.markdown("---")
    
    st.markdown("### 📊 Model Information")
    st.info(f"""
    **Loaded Models:**
    - 🎯 Match Winner (RF)
    - 🏏 Batsman Performance
    - 📈 Win Probability (LSTM)
    
    **Dataset Size:**
    - Matches: {len(matches_df)}
    - Players: {len(players_df['batter'].unique())}
    - Teams: {len(teams)}
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Features")
    st.markdown("""
    - Real-time predictions
    - Interactive visualizations
    - Historical statistics
    - Deep learning models
    """)

# ============================================================
# HEADER
# ============================================================
st.title("🏏 IPL Machine Learning Predictor")
st.markdown("### 🤖 Powered by Machine Learning | 📊 Real IPL Data | ⚡ Instant Predictions")
st.markdown("---")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Match Winner",
    "🏏 Batsman Performance",
    "📈 Live Win Probability",
    "📊 Analytics Dashboard"
])

# ============================================================
# TAB 1: MATCH WINNER PREDICTION
# ============================================================
with tab1:
    st.header("🏆 Match Winner Prediction")
    st.markdown("Predict the winner based on teams and toss information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("🔵 Team 1", teams, key="team1_select")
        
    with col2:
        available_teams = [t for t in teams if t != team1]
        team2 = st.selectbox("🔴 Team 2", available_teams, key="team2_select")

    col3, col4 = st.columns(2)
    
    with col3:
        toss_winner = st.selectbox("🪙 Toss Winner", [team1, team2])
    
    with col4:
        toss_decision = st.radio("⚡ Toss Decision", ["bat", "field"], horizontal=True)

    st.markdown("---")
    
    if st.button("🔮 Predict Match Winner", use_container_width=True, key="predict_winner"):
        with st.spinner("Analyzing match conditions..."):
            # Encode teams
            t1_encoded = le_team.transform([team1])[0]
            t2_encoded = le_team.transform([team2])[0]
            toss_encoded = le_team.transform([toss_winner])[0]

            # Create input dataframe
            input_df = pd.DataFrame(
                [[t1_encoded, t2_encoded, toss_encoded, toss_decision]],
                columns=['t1', 't2', 'toss', 'toss_decision']
            )
            input_df = pd.get_dummies(input_df, columns=['toss_decision'])

            # Align columns with training data
            for col in rf_model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[rf_model.feature_names_in_]

            # Make prediction
            prob = rf_model.predict_proba(input_df)[0]
            winner = team1 if prob[1] > prob[0] else team2
            confidence = max(prob) * 100

            # Display result
            st.markdown(f"""
            <div class="prediction-box">
                🏆 PREDICTED WINNER<br>
                <span style="font-size: 36px;">{winner}</span><br>
                <span style="font-size: 20px;">Confidence: {confidence:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

            # Show probability breakdown
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(f"{team1} Win Probability", f"{prob[1]*100:.1f}%")
            with col_b:
                st.metric(f"{team2} Win Probability", f"{prob[0]*100:.1f}%")

            # Visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=[team1, team2],
                    y=[prob[1]*100, prob[0]*100],
                    marker=dict(
                        color=['#ff6b35', '#f7931e'],
                        line=dict(color='white', width=2)
                    ),
                    text=[f"{prob[1]*100:.1f}%", f"{prob[0]*100:.1f}%"],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Win Probability Comparison",
                xaxis_title="Teams",
                yaxis_title="Win Probability (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2: BATSMAN PERFORMANCE
# ============================================================
with tab2:
    st.header("🏏 Batsman Performance Predictor")
    st.markdown("Predict expected runs based on historical performance")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        batter = st.selectbox(
            "Select Batsman",
            batsman_avg.index,
            key="batter_select"
        )
    
    with col2:
        st.markdown("### 📊 Quick Stats")

    if st.button("🔮 Predict Expected Runs", use_container_width=True, key="predict_runs"):
        with st.spinner("Analyzing batsman performance..."):
            avg_runs = batsman_avg.loc[batter, 'avg_runs']
            std_runs = batsman_avg.loc[batter, 'std_runs']
            matches = int(batsman_avg.loc[batter, 'matches_played'])
            
            # Generate prediction with some randomness
            predicted_runs = max(0, int(avg_runs + np.random.normal(0, min(std_runs/2, 10))))

            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                🏏 {batter}<br>
                <span style="font-size: 40px;">{predicted_runs}</span><br>
                <span style="font-size: 18px;">Expected Runs</span>
            </div>
            """, unsafe_allow_html=True)

            # Show statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Career Average", f"{avg_runs:.1f}")
            with col2:
                st.metric("📈 Matches Played", f"{matches}")
            with col3:
                st.metric("📉 Std Deviation", f"{std_runs:.1f}")

            # Performance visualization
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_runs,
                delta={'reference': avg_runs},
                title={'text': "Predicted vs Average"},
                gauge={
                    'axis': {'range': [None, avg_runs * 2]},
                    'bar': {'color': "#ff6b35"},
                    'steps': [
                        {'range': [0, avg_runs * 0.5], 'color': "lightgray"},
                        {'range': [avg_runs * 0.5, avg_runs * 1.5], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': avg_runs
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Arial"},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3: LIVE WIN PROBABILITY
# ============================================================
with tab3:
    st.header("📈 Live Match Win Probability")
    st.markdown("Calculate real-time win probability during a chase")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        runs_needed = st.slider(
            "🎯 Runs Needed",
            min_value=1,
            max_value=300,
            value=100,
            help="Runs required to win"
        )
    
    with col2:
        balls_left = st.slider(
            "⏱️ Balls Remaining",
            min_value=1,
            max_value=120,
            value=60,
            help="Legal deliveries left"
        )
    
    with col3:
        wickets_left = st.slider(
            "🎯 Wickets Remaining",
            min_value=1,
            max_value=10,
            value=7,
            help="Wickets in hand"
        )

    # Calculate additional metrics
    required_rrr = (runs_needed / balls_left) * 6 if balls_left > 0 else 0
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(f"**Required Run Rate:** {required_rrr:.2f}")
    with col_b:
        st.info(f"**Overs Left:** {balls_left/6:.1f}")
    with col_c:
        st.info(f"**Runs per Wicket:** {runs_needed/wickets_left:.1f}")

    st.markdown("---")
    
    if st.button("📊 Calculate Win Probability", use_container_width=True, key="calc_prob"):
        with st.spinner("Processing match state..."):
            # Prepare input for LSTM
            state = np.array([[runs_needed, balls_left, wickets_left]])
            state_scaled = scaler.transform(state)

            # Create sequence
            seq = np.repeat(state_scaled, 120, axis=0)
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

            # Predict
            with torch.no_grad():
                win_prob = lstm_model(seq_tensor).item() * 100

            # Display result with color coding
            if win_prob >= 70:
                color = "#4CAF50"
                status = "Strong Favorite"
            elif win_prob >= 50:
                color = "#FFC107"
                status = "Slight Favorite"
            elif win_prob >= 30:
                color = "#FF9800"
                status = "Underdog"
            else:
                color = "#F44336"
                status = "Unlikely"

            st.markdown(f"""
            <div class="prediction-box" style="background: linear-gradient(45deg, {color}, {color}dd);">
                📈 CHASING TEAM WIN PROBABILITY<br>
                <span style="font-size: 48px;">{win_prob:.1f}%</span><br>
                <span style="font-size: 20px;">{status}</span>
            </div>
            """, unsafe_allow_html=True)

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=win_prob,
                title={'text': "Win Probability", 'font': {'color': 'white', 'size': 24}},
                number={'font': {'color': 'white', 'size': 40}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(244, 67, 54, 0.3)'},
                        {'range': [30, 50], 'color': 'rgba(255, 152, 0, 0.3)'},
                        {'range': [50, 70], 'color': 'rgba(255, 193, 7, 0.3)'},
                        {'range': [70, 100], 'color': 'rgba(76, 175, 80, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': 'white', 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Match situation analysis
            st.markdown("### 🎯 Match Situation Analysis")
            
            situation_col1, situation_col2 = st.columns(2)
            
            with situation_col1:
                st.markdown(f"""
                <div class="info-card">
                    <h4>📊 Current State</h4>
                    <p>• Runs Required: {runs_needed}</p>
                    <p>• Required RRR: {required_rrr:.2f}</p>
                    <p>• Wickets Available: {wickets_left}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with situation_col2:
                st.markdown(f"""
                <div class="info-card">
                    <h4>⚡ Key Factors</h4>
                    <p>• Overs Remaining: {balls_left/6:.1f}</p>
                    <p>• Runs per Wicket: {runs_needed/wickets_left:.1f}</p>
                    <p>• Pressure Index: {"High" if required_rrr > 10 else "Medium" if required_rrr > 7 else "Low"}</p>
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# TAB 4: ANALYTICS DASHBOARD
# ============================================================
with tab4:
    st.header("📊 Analytics Dashboard")
    st.markdown("Explore historical data and team statistics")
    
    # Team performance analysis
    st.subheader("🏆 Team Performance Analysis")
    
    team_wins = matches_df['winner'].value_counts().head(10)
    
    fig_wins = px.bar(
        x=team_wins.index,
        y=team_wins.values,
        labels={'x': 'Team', 'y': 'Total Wins'},
        title='Top 10 Teams by Wins',
        color=team_wins.values,
        color_continuous_scale='Viridis'
    )
    
    fig_wins.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False
    )
    
    st.plotly_chart(fig_wins, use_container_width=True)
    
    # Top batsmen
    st.subheader("🏏 Top Batsmen by Average")
    
    top_batsmen = batsman_avg.nlargest(15, 'avg_runs')
    
    fig_batsmen = px.bar(
        x=top_batsmen.index,
        y=top_batsmen['avg_runs'],
        labels={'x': 'Batsman', 'y': 'Average Runs'},
        title='Top 15 Batsmen by Career Average',
        color=top_batsmen['avg_runs'],
        color_continuous_scale='Oranges'
    )
    
    fig_batsmen.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    st.plotly_chart(fig_batsmen, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 20px;'>
    <h4>🏏 IPL Machine Learning Predictor</h4>
    <p>Built with ❤️ using Streamlit, PyTorch & Scikit-learn</p>
    <p style='font-size: 12px;'>Data from IPL matches | Models trained on historical data</p>
</div>
""", unsafe_allow_html=True)