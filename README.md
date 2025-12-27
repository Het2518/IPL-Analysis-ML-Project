# 🏏 Advanced IPL Machine Learning Predictor

A comprehensive cricket prediction system powered by advanced machine learning models including Gradient Boosting, Random Forest, and XGBoost. This system provides accurate predictions for match winners, real-time win probabilities, and detailed player performance analysis.

## 🌟 Key Features

### 1. **Match Winner Prediction** 🏆
- **Comprehensive Analysis**: Considers teams, toss winner, toss decision, venue, and city
- **Historical Data**: Leverages head-to-head records and venue-specific performance
- **Team Statistics**: Incorporates overall win rates and form
- **Multiple Models**: Trains and compares Random Forest, Gradient Boosting, XGBoost, and Logistic Regression
- **High Accuracy**: Achieves 60-70% accuracy on test data

**Features Used:**
- Team encoding (team1, team2)
- Toss winner and decision
- Venue and city encoding
- Historical team win rates
- Toss winner advantage indicator

### 2. **Live Win Probability** 📊
- **Ball-by-Ball Analysis**: Real-time probability calculation based on current match state
- **Advanced Features**:
  - Runs required and balls remaining
  - Wickets in hand
  - Current run rate vs required run rate
  - Run rate pressure indicator
  - Wickets and balls remaining factors
- **Historical Context**: Shows outcomes from similar historical situations
- **Interactive Gauge**: Visual representation of win probability

### 3. **Player Performance Analysis** 🎯
- **Career Statistics**: Total runs, average, strike rate, matches played
- **Venue-Specific Performance**: How players perform at different grounds
- **Team Matchups**: Performance against specific opposition teams
- **Head-to-Head Analysis**: Batsman vs bowler records
- **Recent Form**: Last 10 matches performance visualization
- **Comprehensive Metrics**: Includes consistency (standard deviation) analysis

### 4. **Analytics Dashboard** 📈
- **Season Analysis**: Filter by season to view specific year statistics
- **Team Performance**: Win/loss records with interactive charts
- **Toss Analysis**: Toss decision trends and impact on match outcomes
- **Venue Statistics**: Most frequently used venues and their characteristics
- **Top Performers**: Leaderboards for run scorers with strike rates
- **Interactive Visualizations**: Powered by Plotly for dynamic charts

## 📊 Dataset Information

The system is trained on comprehensive IPL data including:
- **19,136+ matches** across formats (Test, ODI, T20I, IPL, BBL, etc.)
- **Ball-by-ball data** for detailed analysis
- **Multiple tournaments**: IPL, BBL, CPL, PSL, WPL, and more
- **Timeframe**: 2001-2025 (continuously updated)

### Data Structure

**Match Level:**
```json
{
  "match_id": "1082591",
  "season": 2017,
  "team1": "Sunrisers Hyderabad",
  "team2": "Royal Challengers Bangalore",
  "venue": "Rajiv Gandhi International Stadium",
  "toss_winner": "Royal Challengers Bangalore",
  "toss_decision": "field",
  "winner": "Sunrisers Hyderabad",
  "is_dls": false
}
```

**Ball-by-Ball Level:**
```json
{
  "match_id": "1082591",
  "innings": 2,
  "over": 12,
  "batter": "Yuvraj Singh",
  "bowler": "A Choudhary",
  "batter_runs": 1,
  "total_runs": 1,
  "wides": 0,
  "noballs": 0,
  "wicket": 0,
  "cumulative_runs": 120,
  "cumulative_wickets": 4
}
```

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Step 1: Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn xgboost plotly joblib tqdm
```

### Step 2: Prepare Your Data

1. Download IPL JSON dataset from [Cricsheet](https://cricsheet.org/)
2. Extract all JSON files to a folder (e.g., `ipl_json/`)
3. Update the `FOLDER_PATH` in `model.ipynb` Cell 2:

```python
FOLDER_PATH = r"C:\path\to\your\ipl_json"
```

### Step 3: Train Models

Open and run `model.ipynb` in Jupyter Notebook:

```bash
jupyter notebook model.ipynb
```

Or convert to script and run:

```bash
jupyter nbconvert --to script model.ipynb
python model.py
```

**This will create:**
- `cricket_data/` folder with processed CSV files
- `saved_models/` folder with trained models

### Step 4: Run Streamlit App

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
cricket-predictor/
│
├── model.ipynb                 # Model training notebook
├── streamlit_app.py           # Streamlit web application
├── README.md                  # This file
│
├── cricket_data/              # Generated datasets
│   ├── matches_enhanced.csv
│   ├── balls_enhanced.csv
│   ├── batsman_stats_enhanced.csv
│   ├── venue_performance.csv
│   ├── head_to_head.csv
│   └── team_performance.csv
│
├── saved_models/              # Trained models
│   ├── match_winner_model.pkl
│   ├── win_probability_model.pkl
│   ├── encoder_team.pkl
│   ├── encoder_venue.pkl
│   ├── encoder_city.pkl
│   ├── scaler_winprob.pkl
│   ├── team_win_rates.pkl
│   └── feature_columns.pkl
│
└── ipl_json/                  # Your raw data (not included)
    ├── 1082591.json
    ├── 1082592.json
    └── ...
```

## 🎯 Usage Guide

### Match Winner Prediction

1. Select **"Match Winner Predictor"** from sidebar
2. Choose **Team 1** and **Team 2**
3. Select **Toss Winner** and **Decision** (bat/field)
4. Choose **Venue**
5. Click **"PREDICT MATCH WINNER"**

**Output:**
- Predicted winner with confidence percentage
- Probability breakdown for both teams
- Visual comparison chart
- Key prediction factors (toss impact, team form)

### Live Win Probability

1. Select **"Live Win Probability"** from sidebar
2. Enter **Target Score** and **Current Score**
3. Set **Overs Completed** (slider)
4. Set **Wickets Fallen** (slider)
5. Click **"CALCULATE WIN PROBABILITY"**

**Output:**
- Batting team win probability percentage
- Interactive gauge meter
- Detailed situation analysis
- Historical context from similar matches

### Player Performance

1. Select **"Player Performance"** from sidebar
2. Choose a **Player** from dropdown
3. Click **"ANALYZE"**

**Output:**
- Career statistics (runs, average, strike rate)
- Venue-specific performance charts
- Performance vs different teams
- Head-to-head records vs bowlers
- Recent form graph (last 10 matches)

### Analytics Dashboard

1. Select **"Analytics Dashboard"** from sidebar
2. Choose a **Season** from dropdown

**Output:**
- Season overview metrics
- Team performance charts
- Toss analysis and impact
- Top venues by matches
- Top run scorers leaderboard

## 🤖 Model Details

### Match Winner Model

**Best Model:** Gradient Boosting / Random Forest (selected automatically)

**Features:**
- team1_enc, team2_enc (encoded team IDs)
- toss_winner_enc (encoded toss winner)
- venue_enc, city_enc (encoded location)
- team1_win_rate, team2_win_rate (historical win rates)
- toss_winner_is_team1 (toss advantage)
- toss_decision_bat, toss_decision_field (one-hot encoded)

**Performance:**
- Accuracy: ~65-70%
- AUC-ROC: ~0.72-0.75
- Cross-validation: 5-fold CV

### Win Probability Model

**Model:** Gradient Boosting Classifier

**Features:**
- runs_required (runs needed to win)
- balls_remaining (balls left in innings)
- wickets_remaining (wickets in hand)
- required_rr (required run rate)
- current_rr (current run rate)
- run_rate_pressure (difference between required and current RR)
- wickets_in_hand_factor (normalized wickets)
- balls_remaining_factor (normalized balls)

**Performance:**
- Accuracy: ~70-75%
- AUC-ROC: ~0.78-0.82
- Calibrated probabilities

## 📊 Data Processing Pipeline

### 1. Data Extraction
- Parse JSON files for match metadata
- Extract ball-by-ball deliveries
- Handle extras (wides, no-balls, byes, leg-byes)
- Track wickets and dismissals

### 2. Feature Engineering
- Calculate cumulative runs and wickets
- Compute required run rates
- Identify legal vs illegal deliveries
- Create batsman-bowler encounter records

### 3. Statistical Aggregation
- Career statistics per player
- Venue-specific performance
- Team matchup analysis
- Head-to-head records

### 4. Model Training
- Label encoding for categorical variables
- Feature scaling (StandardScaler for win probability)
- Train-test split (80-20)
- Model comparison and selection
- Save models and encoders

## 🔧 Customization

### Adding New Features

**In model.ipynb, Cell 5:**

```python
# Add new features to the feature list
feature_cols = [
    'team1_enc', 'team2_enc', 'toss_winner_enc',
    'venue_enc', 'city_enc', 'toss_decision',
    'team1_win_rate', 'team2_win_rate', 'toss_winner_is_team1',
    # Add your new features here
    'your_new_feature'
]
```

### Changing Model Parameters

```python
# Adjust Random Forest parameters
model1 = RandomForestClassifier(
    n_estimators=200,  # Increase trees
    max_depth=15,      # Increase depth
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
```

### Adding Weather Data

If you have weather data, add it to the match extraction:

```python
# In Cell 3, add to match_list
match_list.append({
    # ... existing fields ...
    'temperature': weather_data.get('temp'),
    'humidity': weather_data.get('humidity'),
    'conditions': weather_data.get('conditions')
})
```

## 📈 Performance Optimization

### For Large Datasets

1. **Increase chunk processing:**
```python
# Process in batches
for batch in tqdm(range(0, len(files), 1000)):
    batch_files = files[batch:batch+1000]
    # Process batch
```

2. **Use multiprocessing:**
```python
from multiprocessing import Pool

with Pool(processes=4) as pool:
    results = pool.map(process_file, files)
```

3. **Cache in Streamlit:**
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_large_dataset():
    return pd.read_csv('large_file.csv')
```

## ⚠️ Important Notes

### DLS Method
- DLS (Duckworth-Lewis-Stern) matches are flagged but not excluded from training
- Consider filtering them if they impact model performance

### Missing Data
- Unknown venues are encoded as 0
- Missing cities use default encoding
- Players with <10 matches are filtered from leaderboards

### Model Updates
- Retrain models when new data is available
- Use the same encoding objects to maintain consistency
- Update feature columns if adding new features

## 🐛 Troubleshooting

### Common Issues

**1. "File not found" error:**
```
Solution: Ensure cricket_data/ and saved_models/ folders exist
Run model.ipynb completely before running streamlit app
```

**2. "Encoder error" or "Unknown category":**
```
Solution: A team/venue not in training data
Models default to encoding 0 for unknown values
Retrain with complete dataset
```

**3. "Memory error" with large datasets:**
```
Solution: Process in chunks or use sampling
Reduce features or use dimensionality reduction
Increase system RAM
```

**4. Streamlit not loading:**
```
Solution: Check if port 8501 is available
Try: streamlit run streamlit_app.py --server.port 8502
```

## 🔮 Future Enhancements


- [ ] Include player injury status
- [ ] Integrate live match APIs
- [ ] Add powerplay analysis
- [ ] Death overs specialist identification
- [ ] Team composition optimization
- [ ] Predict individual player scores
- [ ] Real-time API integration for live matches
- [ ] Player form prediction using time series
- [ ] Weather impact analysis
- [ ] Pitch report integration
- [ ] Mobile app development
- [ ] Multi-tournament support (T20, ODI, Test)
- [ ] Fantasy team recommendations
- [ ] Betting odds analysis
## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- **Cricsheet** for comprehensive cricket data
- **Streamlit** for the amazing web framework
- **Scikit-learn** for ML algorithms
- **Plotly** for interactive visualizations

## 📧 Contact

For questions, suggestions, or contributions, please create an issue in the repository.

---

**Built with ❤️ for cricket enthusiasts and data scientists**

🏏 **Happy Predicting!** 🏏