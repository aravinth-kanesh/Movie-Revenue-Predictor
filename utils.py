import streamlit as st
import pandas as pd
import joblib
from src.feature_engineering import create_basic_features

def load_css():
    """Inject custom CSS for styling."""
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 0.5rem 0;
        }
        .prediction-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .model-performance {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        .shap-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        .feature-importance {
            background: linear-gradient(135deg, #FC466B 0%, #3F5EFB 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all pre-trained models and feature columns."""
    try:
        models = {
            "Random Forest": joblib.load("data/models/random_forest.pkl"),
            "LightGBM": joblib.load("data/models/lightgbm.pkl"),
            "XGBoost": joblib.load("data/models/xgb.pkl"),
            "Gradient Boost": joblib.load("data/models/gradient_boost.pkl"),
            "Neural Network (MLP)": joblib.load("data/models/mlp.pkl")
        }
        feature_cols = joblib.load("data/models/feature_cols.pkl")
        return models, feature_cols
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Please ensure your trained models are in the 'data/models/' directory")
        return None, None

@st.cache_data
def load_sample_data():
    """Load sample data for analysis."""
    try:
        df = pd.read_csv("data/tmdb_movie_dataset.csv", nrows=1000)
        return df
    except FileNotFoundError:
        st.warning("Sample dataset not found. Please check the file path.")

def create_user_input(feature_cols):
    """Create user input interface in the sidebar."""
    st.sidebar.header("🎬 Movie Features Input")

    vote_average = st.sidebar.slider("Vote Average", 0.0, 10.0, 7.0, step=0.1)
    vote_count = st.sidebar.slider("Vote Count", 0, 5000, 1000)
    runtime = st.sidebar.slider("Runtime (minutes)", 30, 300, 120)
    release_year = st.sidebar.slider("Release Year", 1950, 2025, 2023)
    release_month = st.sidebar.slider("Release Month", 1, 12, 6)

    top_genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Thriller", "Adventure", "Crime"]
    selected_genres = st.sidebar.multiselect("Genres", top_genres, default=["Action"])

    input_data = pd.DataFrame({
        "vote_average": [vote_average], "vote_count": [vote_count], "runtime": [runtime],
        "release_year": [release_year], "release_month": [release_month],
    })
    input_data = create_basic_features(input_data)
    for genre in top_genres:
        input_data[f"genres_{genre}"] = int(genre in selected_genres)
    for col in feature_cols:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[feature_cols]

    return input_data, {
        'vote_average': vote_average, 'vote_count': vote_count, 'runtime': runtime,
        'release_year': release_year, 'release_month': release_month, 'genres': selected_genres
    }

def model_selector(models):
    """Render the model selection sidebar widget and return the choice."""
    st.sidebar.header("🤖 Model Selection")
    model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
    return model_choice