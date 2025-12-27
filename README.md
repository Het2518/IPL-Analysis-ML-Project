# 🏏 IPL Prediction & Analytics Platform

<div align="center">

![Cricket](https://img.shields.io/badge/Sport-Cricket-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Professional Cricket Analytics Platform powered by Deep Learning LSTM Models**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Models](#-models) • [Screenshots](#-screenshots)

</div>

---

## 📊 Project Overview

A comprehensive cricket prediction and analytics system built with **LSTM Deep Learning models** that provides:
- **Match Winner Prediction** (56.52% accuracy)
- **Ball-by-Ball Win Probability** (98.19% accuracy)
- **Player Performance Analysis**
- **Team Comparison & Statistics**
- **Venue Insights**
- **Head-to-Head Analysis**

### 📈 Dataset Statistics

| Metric | Count |
|--------|-------|
| **Total Matches** | 1,146 |
| **Total Balls** | 273,503 |
| **Teams** | 19 |
| **Venues** | 59 |
| **Players** | 766 |
| **Batsmen Analyzed** | 703 |
| **Bowlers Analyzed** | 548 |
| **H2H Matchups** | 171 |
| **Batsman vs Bowler** | 8,429 |

---

## 🎯 Features

### 1. **Match Winner Predictor** 🏆
- Predict match outcomes before the game starts
- Considers: Teams, Toss, Venue, City, Historical Performance
- Visual confidence meters and probability gauges
- Detailed team analysis breakdown

### 2. **Live Win Probability Calculator** 📊
- Real-time match situation analysis
- Ball-by-ball probability updates
- Factors: Current score, wickets, run rates, overs remaining
- Interactive situation breakdown

### 3. **Player Performance Analysis** 👤
- **Batting Statistics**: Runs, Average, Strike Rate
- **Bowling Statistics**: Wickets, Economy, Average
- Radar charts for visual comparison
- Performance trends and consistency metrics

### 4. **Team Comparison** ⚔️
- Head-to-head records
- Win rates and form analysis
- Side-by-side statistical comparison
- Historical performance trends

### 5. **Venue Insights** 📍
- Ground-specific statistics
- Average scores and pitch behavior
- Toss decision preferences
- Venue recommendations

### 6. **Comprehensive Statistics** 📈
- Top run scorers and wicket takers
- Best strike rates and economy rates
- League tables and rankings
- Interactive leaderboards

---

## 🚀 Installation

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/cricket-prediction-system.git
cd cricket-prediction-system
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
Place your cricket JSON files in the following directory:
```
C:\Users\YOUR_USERNAME\OneDrive\Desktop\Cricket\Dataset\ipl_json\
```

Or update the path in the code:
```python
JSON_PATH = r"YOUR_PATH_HERE"
```

---

## 📦 Dependencies

Create a `requirements.txt` file with:

```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
tensorflow==2.13.0
scikit-learn==1.3.0
plotly==5.17.0
seaborn==0.12.2
matplotlib==3.7.2
tqdm==4.66.1
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🎮 Usage

### Training the Models

Run the complete notebook to train all models:

```bash
jupyter notebook cricket_complete_system.ipynb
```

This will:
1. Load and process all match data (1-5 minutes)
2. Extract ball-by-ball sequences (2-8 minutes)
3. Train 5 different LSTM models (10-30 minutes)
4. Generate analysis reports and visualizations
5. Save all models and data files

### Running the Streamlit App

After training, launch the web application:

```bash
streamlit run cricket_app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🧠 Models

### 1. **Match Winner LSTM**
- **Architecture**: Single LSTM layer (64 units) + Dense layers
- **Input Features**: 10 (teams, venue, toss, win rates)
- **Accuracy**: 56.52%
- **Use Case**: Pre-match winner prediction

### 2. **Ball-by-Ball Models** (4 variants)

#### Simple LSTM
- **Architecture**: LSTM(64) + Dense
- **Accuracy**: ~95%

#### Stacked LSTM ⭐ (Best Model)
- **Architecture**: LSTM(128) + LSTM(64) + Dense
- **Accuracy**: **98.19%**
- **Features**: 9 (runs, wickets, rates, extras)

#### Bidirectional LSTM
- **Architecture**: BiLSTM(128) + BiLSTM(64) + Dense
- **Accuracy**: ~97.5%

#### GRU
- **Architecture**: GRU(128) + GRU(64) + Dense
- **Accuracy**: ~96.8%

### Model Selection
The system automatically selects the **Stacked LSTM** as it achieved the highest accuracy during training.

---

## 📁 Project Structure

```
cricket-prediction-system/
│
├── cricket_app.py                 # Streamlit web application
├── cricket_complete_system.ipynb  # Complete training notebook
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── data/                          # Data directory
│   └── (generated during training)
│
├── models/                        # Saved models
│   ├── match_winner_lstm.h5
│   ├── ball_by_ball_lstm.h5
│   ├── ball_model_1_SimpleLSTM.h5
│   ├── ball_model_2_StackedLSTM.h5
│   ├── ball_model_3_BiLSTM.h5
│   ├── ball_model_4_GRU.h5
│   └── cricket_data.pkl
│
├── analysis/                      # Analysis reports (CSV)
│   ├── team_statistics.csv
│   ├── venue_statistics.csv
│   ├── batsman_statistics.csv
│   ├── bowler_statistics.csv
│   ├── head_to_head.csv
│   └── batsman_vs_bowler.csv
│
└── plots/                         # Visualizations
    ├── team_analysis.png
    ├── model_performance.png
    ├── player_analysis.png
    └── confusion_matrix.png
```

---

## 📸 Screenshots

### Dashboard
![Dashboard](https://via.placeholder.com/800x400/667eea/ffffff?text=Dashboard+Overview)

### Match Predictor
![Predictor](https://via.placeholder.com/800x400/764ba2/ffffff?text=Match+Winner+Predictor)

### Live Win Probability
![Live](https://via.placeholder.com/800x400/45B7D1/ffffff?text=Live+Win+Probability)

### Player Analysis
![Player](https://via.placeholder.com/800x400/FF6B6B/ffffff?text=Player+Performance)

---

## 🎨 Key Features of the UI

### Visual Design
- **Glassmorphism UI** with blur effects
- **Dark theme** with gradient backgrounds
- **Animated charts** using Plotly
- **Interactive elements** with hover effects
- **Responsive layout** for all screen sizes

### Charts & Visualizations
- 📊 Bar charts for comparisons
- 📈 Line charts for trends
- 🎯 Gauge charts for probabilities
- 🕸️ Radar charts for player analysis
- 🥧 Pie charts for distributions
- 📉 Heatmaps for correlations

---

## 🔧 Configuration

### Update File Paths

In `cricket_app.py`, update the JSON path:
```python
JSON_PATH = r"YOUR_PATH_TO_JSON_FILES"
```

### Customize Models

In the notebook, you can adjust:
- **Sequence Length**: Change `seq_length = 30`
- **Epochs**: Modify `epochs=50`
- **Batch Size**: Adjust `batch_size=64`
- **Learning Rate**: Tune in optimizer

---

## 📊 API Usage (Optional)

### Match Prediction
```python
from cricket_predictor import predict_match_winner

result = predict_match_winner(
    team1="Mumbai Indians",
    team2="Chennai Super Kings",
    venue="Wankhede Stadium",
    toss_winner="Mumbai Indians",
    toss_decision="bat"
)

print(f"Winner: {result['predicted_winner']}")
print(f"Probability: {result['team1_win_prob']}%")
```

### Win Probability
```python
from cricket_predictor import calculate_win_probability

# Last 30 balls data
sequence_data = np.array([...])  # Shape: (30, 9)

win_prob = calculate_win_probability(sequence_data)
print(f"Batting team win probability: {win_prob}%")
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{cricket_prediction_2025,
  title={IPL Prediction & Analytics Platform},
  author={Het Monpara},
  year={2025},
  url={https://github.com/het2518/IPL-Analysis-ML-Project}
}
```

---

## 🐛 Known Issues

- Dataset must be in JSON format (Cricsheet compatible)
- Large datasets (>1GB) may require 8GB+ RAM
- Training time varies based on hardware (10-60 minutes)
- First-time loading takes 30-60 seconds

---

## 🔮 Future Enhancements

- [ ] Real-time API integration for live matches
- [ ] Player form prediction using time series
- [ ] Weather impact analysis
- [ ] Pitch report integration
- [ ] Mobile app development
- [ ] Multi-tournament support (T20, ODI, Test)
- [ ] Fantasy team recommendations
- [ ] Betting odds analysis

---

## 📞 Support

For issues and questions:
- **GitHub Issues**: [Create an issue](https://github.com/het2518/IPL-Analysis-ML-Project/issues)
- **Email**: hetmonpara2022@gmail.com
- **Discord**: [Join our server](https://discord.gg/het2518)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

- **Dataset**: [Cricsheet](https://cricsheet.org/) for match data
- **Deep Learning**: TensorFlow & Keras teams
- **Visualization**: Plotly and Seaborn communities
- **Web Framework**: Streamlit team
- **Inspiration**: Cricket analytics platforms like CricViz and Cricinfo

---

## 📊 Performance Metrics

| Model | Accuracy | Loss | Training Time |
|-------|----------|------|---------------|
| Match Winner | 56.52% | 0.68 | ~5 min |
| Simple LSTM | 95.20% | 0.12 | ~8 min |
| **Stacked LSTM** | **98.19%** | **0.05** | **~15 min** |
| Bidirectional | 97.50% | 0.07 | ~20 min |
| GRU | 96.80% | 0.09 | ~10 min |

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=het2518/IPL-Analysis-ML-Project&type=Date)](https://star-history.com/#het2518/IPL-Analysis-ML-Project&Date)

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

**Made with ❤️ by Cricket Analytics Enthusiasts**

</div>

---



---

**Last Updated**: December 2025 
**Version**: 1.0.0  
**Status**: ✅ Production Ready