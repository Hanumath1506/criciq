# CricIQ 🏏
An IPL cricket analytics app for exploring player matchups, tracking player form, and predicting match winners using machine learning.

## Live Demo
[criciq-app.streamlit.app](https://criciq-app.streamlit.app/)

## Features
- **Head to Head** — Analyze historical batter vs bowler matchups with stats like runs, dismissals, and strike rate.
- **Player Form** — Track a player's recent performances and scoring trends across IPL matches.
- **Match Predictor** — Predict IPL match winners using venue, toss details, team history, and a Random Forest ML model (53.5% accuracy).

## Tech Stack
- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly

## How to Run Locally

1. Clone the repository
```bash
git clone https://github.com/Hanumath1506/criciq.git
cd criciq
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

## Data
Historical IPL ball-by-ball and match metadata from [cricsheet.org](https://cricsheet.org).
