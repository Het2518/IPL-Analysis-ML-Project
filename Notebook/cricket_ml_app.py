import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import os

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
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
    }
    
    h1, h2, h3 {
        color: white !important;
        font-family: 'Inter', sans-serif;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin: 20px 0;
        animation: slideIn 0.5s ease-out;
    }
    
    .prediction-box h2 {
        color: white !important;
        font-size: 2.5rem;
        margin: 10px 0;
    }
    
    .prediction-box p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
    }
    
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        color: white;
    }
    
    [data-testid="stMetricValue"] {
        color: white;
        font-size: 2rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.8);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD RESOURCES
# ============================================================
BASE_DIR = os.path.dirname(__file__) if os.path.dirname(__file__) else "."

@st.cache_resource
def load_resources():
    try:
        # Load Random Forest model
        rf_model = joblib.load(os.path.join(BASE_DIR, "saved_models", "random_forest_task1.pkl"))
        
        # Load datasets
        matches = pd.read_csv(os.path.join(BASE_DIR, "matches_for_task1.csv"))
        players = pd.read_csv(os.path.join(BASE_DIR, "players_for_task2.csv"))
        
        # Create Label Encoder for teams
        le_team = LabelEncoder()
        all_teams = pd.unique(matches[['team1', 'team2']].values.ravel())
        le_team.fit(all_teams)
        
        # Calculate batsman statistics
        batsman_runs = players.groupby(['match_id', 'batter'])['runs'].sum()
        batsman_avg = batsman_runs.groupby('batter').agg(['mean', 'std', 'count']).round(1)
        batsman_avg.columns = ['avg_runs', 'std_runs', 'matches_played']
        batsman_avg = batsman_avg[batsman_avg['matches_played'] >= 5].sort_values('avg_runs', ascending=False)
        
        return rf_model, le_team, batsman_avg, list(all_teams), matches, players
        
    except Exception as e:
        st.error(f"❌ Error loading resources: {str(e)}")
        st.info("Please ensure all files are in the correct location.")
        st.stop()

# Load all resources
rf_model, le_team, batsman_avg, teams, matches_df, players_df = load_resources()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🏏 IPL ML Predictor")
    st.markdown("---")
    
    page = st.radio(
        "📍 Navigation",
        ["🏆 Match Winner", "🏏 Batsman Performance", "📊 Analytics Dashboard"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📈 Dataset Info")
    st.info(f"""
    **Matches**: {len(matches_df):,}  
    **Players**: {len(players_df['batter'].unique()):,}  
    **Teams**: {len(teams)}
    """)
    
    st.markdown("---")
    st.markdown("### 🤖 Models")
    st.success("""
    ✅ Random Forest  
    ✅ Statistical Model  
    ✅ Predictive Analytics
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <p style='font-size: 0.8rem;'>Built with ❤️</p>
        <p style='font-size: 0.8rem;'>Streamlit • Scikit-learn</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='font-size: 3rem; margin-bottom: 0;'>🏏 IPL Machine Learning Predictor</h1>
    <p style='color: rgba(255,255,255,0.8); font-size: 1.2rem;'>Powered by AI • Real IPL Data • Instant Predictions</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# PAGE 1: MATCH WINNER PREDICTION
# ============================================================
if page == "🏆 Match Winner":
    st.markdown("<h2 style='text-align: center;'>🏆 Match Winner Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7);'>Predict the winner based on teams and toss information</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 Team Selection")
        team1 = st.selectbox("Select Team 1", teams, key="team1")
        
    with col2:
        st.markdown("### 🔴 Team Selection")
        available_teams = [t for t in teams if t != team1]
        team2 = st.selectbox("Select Team 2", available_teams, key="team2")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 🪙 Toss Information")
        toss_winner = st.selectbox("Toss Winner", [team1, team2])
        
    with col4:
        st.markdown("### ⚡ Decision")
        toss_decision = st.radio("Toss Decision", ["bat", "field"], horizontal=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔮 PREDICT WINNER", use_container_width=True):
        with st.spinner("🔄 Analyzing match conditions..."):
            # Encode teams
            t1_encoded = le_team.transform([team1])[0]
            t2_encoded = le_team.transform([team2])[0]
            toss_encoded = le_team.transform([toss_winner])[0]
            
            # Create input
            input_df = pd.DataFrame(
                [[t1_encoded, t2_encoded, toss_encoded, toss_decision]],
                columns=['t1', 't2', 'toss', 'toss_decision']
            )
            input_df = pd.get_dummies(input_df, columns=['toss_decision'])
            
            # Align columns
            for col in rf_model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[rf_model.feature_names_in_]
            
            # Predict
            prob = rf_model.predict_proba(input_df)[0]
            winner = team1 if prob[1] > prob[0] else team2
            confidence = max(prob) * 100
            
            # Display result
            st.markdown(f"""
            <div class="prediction-box">
                <p style='font-size: 1.2rem; margin: 0;'>🏆 PREDICTED WINNER</p>
                <h2 style='font-size: 3rem; margin: 15px 0;'>{winner}</h2>
                <p style='font-size: 1.3rem; margin: 0;'>Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability breakdown
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(f"🔵 {team1}", f"{prob[1]*100:.1f}%", delta=None)
            with col_b:
                st.metric(f"🔴 {team2}", f"{prob[0]*100:.1f}%", delta=None)
            
            # Visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=[team1, team2],
                    y=[prob[1]*100, prob[0]*100],
                    marker=dict(
                        color=['#667eea', '#764ba2'],
                        line=dict(color='white', width=2)
                    ),
                    text=[f"{prob[1]*100:.1f}%", f"{prob[0]*100:.1f}%"],
                    textposition='auto',
                    textfont=dict(size=16, color='white')
                )
            ])
            
            fig.update_layout(
                title={
                    'text': "Win Probability Comparison",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'color': 'white'}
                },
                xaxis_title="Teams",
                yaxis_title="Win Probability (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2: BATSMAN PERFORMANCE
# ============================================================
elif page == "🏏 Batsman Performance":
    st.markdown("<h2 style='text-align: center;'>🏏 Batsman Performance Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7);'>Analyze and predict batsman performance based on historical data</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        batter = st.selectbox("🏏 Select Batsman", batsman_avg.index)
    
    with col2:
        st.markdown("### ")
        predict_btn = st.button("🔮 PREDICT", use_container_width=True)
    
    if predict_btn:
        with st.spinner("🔄 Analyzing batsman performance..."):
            avg_runs = batsman_avg.loc[batter, 'avg_runs']
            std_runs = batsman_avg.loc[batter, 'std_runs']
            matches = int(batsman_avg.loc[batter, 'matches_played'])
            
            # Generate prediction
            predicted_runs = max(0, int(avg_runs + np.random.normal(0, min(std_runs/2, 8))))
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <p style='font-size: 1.2rem; margin: 0;'>🏏 {batter}</p>
                <h2 style='font-size: 3.5rem; margin: 15px 0;'>{predicted_runs}</h2>
                <p style='font-size: 1.3rem; margin: 0;'>Expected Runs</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📊 Career Average", f"{avg_runs:.1f}")
            with col2:
                st.metric("🎯 Matches Played", f"{matches}")
            with col3:
                st.metric("📈 Consistency", f"{std_runs:.1f}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Performance gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_runs,
                delta={'reference': avg_runs, 'increasing': {'color': "green"}},
                title={'text': "Predicted vs Average", 'font': {'color': 'white', 'size': 20}},
                number={'font': {'color': 'white', 'size': 40}},
                gauge={
                    'axis': {'range': [None, avg_runs * 2], 'tickcolor': 'white'},
                    'bar': {'color': "#f093fb"},
                    'steps': [
                        {'range': [0, avg_runs * 0.5], 'color': "rgba(255,255,255,0.1)"},
                        {'range': [avg_runs * 0.5, avg_runs * 1.5], 'color': "rgba(255,255,255,0.2)"}
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
                font={'color': "white"},
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Historical performance
            batter_data = players_df[players_df['batter'] == batter]
            match_performance = batter_data.groupby('match_id')['runs'].sum().reset_index()
            
            fig2 = px.line(
                match_performance,
                y='runs',
                title=f"Match-by-Match Performance",
                labels={'runs': 'Runs Scored', 'index': 'Match Number'}
            )
            
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                showlegend=False,
                xaxis_title="Match Number",
                yaxis_title="Runs"
            )
            
            fig2.update_traces(line_color='#f093fb', line_width=3)
            
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PAGE 3: ANALYTICS DASHBOARD
# ============================================================
elif page == "📊 Analytics Dashboard":
    st.markdown("<h2 style='text-align: center;'>📊 IPL Analytics Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7);'>Explore comprehensive IPL statistics and insights</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Season selector
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        season = st.selectbox("📅 Select Season", sorted(matches_df["season"].dropna().unique(), reverse=True))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Filter data by season
    season_df = matches_df[matches_df["season"] == season]
    
    # Team wins
    st.markdown("### 🏆 Team Performance")
    wins = season_df["winner"].value_counts().reset_index()
    wins.columns = ["Team", "Wins"]
    
    fig1 = px.bar(
        wins,
        x="Team",
        y="Wins",
        title=f"Team Wins in {season}",
        color="Wins",
        color_continuous_scale="Viridis"
    )
    
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis_title="",
        yaxis_title="Total Wins",
        showlegend=False
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Toss stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🪙 Toss Decisions")
        toss_decisions = season_df['toss_decision'].value_counts()
        
        fig2 = px.pie(
            values=toss_decisions.values,
            names=toss_decisions.index,
            title="Bat vs Field Decision",
            color_discrete_sequence=['#667eea', '#764ba2']
        )
        
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'}
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.markdown("### 🏟️ Venue Statistics")
        venue_matches = season_df['venue'].value_counts().head(10)
        
        fig3 = px.bar(
            x=venue_matches.values,
            y=venue_matches.index,
            orientation='h',
            title="Top 10 Venues",
            color=venue_matches.values,
            color_continuous_scale="Plasma"
        )
        
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            xaxis_title="Matches",
            yaxis_title="",
            showlegend=False
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # Top batsmen
    st.markdown("### 🏏 Top Run Scorers")
    
    season_players = players_df[players_df['match_id'].isin(season_df['match_id'])]
    top_scorers = season_players.groupby('batter')['runs'].sum().nlargest(15).reset_index()
    top_scorers.columns = ['Batsman', 'Total Runs']
    
    fig4 = px.bar(
        top_scorers,
        x='Batsman',
        y='Total Runs',
        title=f"Top 15 Run Scorers in {season}",
        color='Total Runs',
        color_continuous_scale='Oranges'
    )
    
    fig4.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Key metrics
    st.markdown("### 📈 Season Statistics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Matches", len(season_df))
    with metric_col2:
        st.metric("Unique Teams", season_df[['team1', 'team2']].stack().nunique())
    with metric_col3:
        st.metric("Total Venues", season_df['venue'].nunique())
    with metric_col4:
        total_runs = season_players['runs'].sum()
        st.metric("Total Runs", f"{total_runs:,}")

# ============================================================
# FOOTER
# ============================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h4 style='color: white;'>🏏 IPL Machine Learning Predictor</h4>
    <p style='color: rgba(255,255,255,0.7);'>Built with ❤️ using Streamlit & Scikit-learn</p>
    <p style='color: rgba(255,255,255,0.5); font-size: 0.9rem;'>Trained on Real IPL Data • Accurate ML Predictions</p>
</div>
""", unsafe_allow_html=True)