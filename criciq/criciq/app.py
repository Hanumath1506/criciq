import streamlit as st
from utils.data_loader import load_matches

st.title("CricIQ")
st.write("Cricket Analytics and Insights Platform")

st.sidebar.title("Navigation")
format_choice = st.sidebar.selectbox(
    "Select Format",
    ["IPL","ODI", "T20I", "Test"]
)

format_paths = {
    "IPL": "data/raw/ipl",
    "ODI": "data/raw/odis",
    "T20I": "data/raw/t20s",
    "Test": "data/raw/tests"
}

st.write(f"Loading {format_choice} data...")
df = load_matches(format_paths[format_choice])
st.write(f"Loaded {len(df)} deliveries")
st.dataframe(df.head())
