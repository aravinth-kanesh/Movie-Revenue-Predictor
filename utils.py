import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add src directory to path for imports
if 'src' not in sys.path:
    sys.path.append('src')

try:
    from src.preprocess import preprocess
    from src.feature_engineering import expand_features

    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False
    st.warning("⚠️ Preprocessing modules not found. Using basic feature engineering only.")

# Custom CSS for styling
def load_css():
    """Inject custom CSS for Streamlit app styling."""
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
        .metric-card, .prediction-box, .model-performance, .shap-container, .feature-importance {
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .prediction-box { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
        .model-performance { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
        .shap-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .feature-importance { background: linear-gradient(135deg, #FC466B 0%, #3F5EFB 100%); }
        .warning { background: #FFA726; padding: 1rem; border-radius: 5px; color: white; margin: 1rem 0; }
    </style>
    """, unsafe_allow_html=True)

# Load pre-trained models
@st.cache_resource
def load_models():
    """Load Random Forest and LightGBM models and feature columns."""
    models_dir = "data/models"
    try:
        if not os.path.exists(models_dir):
            st.error(f"Models directory '{models_dir}' not found!")
            return None, None

        models = {}
        model_files = {
            "Random Forest": "random_forest.pkl",
            "LightGBM": "lightgbm.pkl"
        }

        for name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                models[name] = joblib.load(filepath)
            else:
                st.warning(f"Model file {filename} not found")

        feature_cols_path = os.path.join(models_dir, "feature_cols.pkl")
        if os.path.exists(feature_cols_path):
            feature_cols = joblib.load(feature_cols_path)
        else:
            st.error("Feature columns file not found!")
            return None, None

        if not models:
            st.error("No models loaded successfully!")
            return None, None

        return models, feature_cols
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load and preprocess sample dataset
@st.cache_data
def load_sample_data(nrows=1000000):
    """Load and preprocess sample TMDB dataset for analysis."""
    try:
        data_path = "data/tmdb_movie_dataset.csv"
        if not os.path.exists(data_path):
            st.warning("Sample dataset not found. Please check the file path.")
            return None

        if PREPROCESSING_AVAILABLE:
            # Use the same preprocessing pipeline as training
            df = preprocess(data_path, nrows=nrows)
            df, _ = expand_features(df)
            return df
        else:
            # Fallback to basic loading
            df = pd.read_csv(data_path, nrows=nrows)
            return df
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

# Basic feature engineering for user input (fallback)
def create_basic_features(df):
    """Compute weighted votes, log runtime, and release-related features."""
    df = df.copy()

    # Weighted vote calculation
    if 'vote_average' in df.columns and 'vote_count' in df.columns:
        v = df['vote_count'].fillna(0)
        R = df['vote_average'].fillna(0)
        C = 7.0  # global average
        m = 1000  # vote threshold
        df['weighted_vote'] = (v / (v + m) * R) + (m / (v + m) * C)
    else:
        df['weighted_vote'] = 0

    # Log runtime
    if 'runtime' in df.columns:
        df['log_runtime'] = np.log1p(df['runtime'].fillna(0).clip(lower=0))
    else:
        df['log_runtime'] = 0

    # Release date features
    if 'release_month' in df.columns:
        df['release_quarter'] = ((df['release_month'].fillna(1) - 1) // 3 + 1).astype(int)
        df['release_season'] = df['release_quarter']
    else:
        df['release_quarter'] = df['release_season'] = 1

    # Fill any remaining NaN values
    df = df.fillna(0)
    return df

# Sidebar: user input
def create_user_input(feature_cols):
    """Create Streamlit sidebar input and return DataFrame and params."""
    st.sidebar.header("🎬 Movie Features Input")

    # Input sliders - including budget and popularity
    vote_average = st.sidebar.slider("Vote Average", 0.0, 10.0, 8.5, step=0.1)
    vote_count = st.sidebar.slider("Vote Count", 2000, 35000, 20000)
    runtime = st.sidebar.slider("Runtime (minutes)", 30, 300, 120)
    release_year = st.sidebar.slider("Release Year", 1950, 2025, 2023)
    release_month = st.sidebar.slider("Release Month", 1, 12, 6)

    # Budget slider - CRITICAL feature
    budget_millions = st.sidebar.slider("Budget ($ millions)", 0.5, 400.0, 50.0, step=0.5)
    budget = budget_millions * 1_000_000

    # Popularity slider
    popularity = st.sidebar.slider("Popularity", 10.0, 500.0, 100.0, step=5.0)

    # Get actual genres from your trained model
    available_genres = []
    for col in feature_cols:
        if col.startswith('genres_'):
            genre_name = col.replace('genres_', '')
            # Only keep strings that are not digits
            if not genre_name.isdigit():
                available_genres.append(genre_name)

    available_genres = sorted(list(set(available_genres)))

    selected_genres = st.sidebar.multiselect(
        "Genres",
        available_genres if available_genres else ["Action", "Comedy", "Drama"],
        default=["Action"],
        help="Select genres for your movie"
    )

    # Create input DataFrame with ALL required features
    input_data = pd.DataFrame({
        "vote_average": [vote_average],
        "vote_count": [vote_count],
        "runtime": [runtime],
        "release_year": [release_year],
        "release_month": [release_month],
        "budget": [budget],
        "popularity": [popularity]
    })

    # Apply feature engineering
    input_data = create_basic_features(input_data)

    # Add budget features that models expect
    input_data['log_budget'] = np.log1p(input_data['budget'])

    # Add genre features
    for genre in available_genres:
        input_data[f"genres_{genre}"] = int(genre in selected_genres)

    # Remove raw features if log versions exist
    if "runtime" in input_data.columns and "log_runtime" in feature_cols:
        input_data = input_data.drop(columns=["runtime"])
    if "budget" in input_data.columns and "log_budget" in feature_cols:
        input_data = input_data.drop(columns=["budget"])

    # Ensure all required features exist
    for col in feature_cols:
        if col not in input_data.columns:
            input_data[col] = 0

    # Select only model features
    input_data = input_data[feature_cols]

    return input_data, {
        "vote_average": vote_average,
        "vote_count": vote_count,
        "runtime": runtime,
        "release_year": release_year,
        "release_month": release_month,
        "budget_millions": budget_millions,
        "popularity": popularity,
        "genres": selected_genres
    }

# Prepare sample data for model evaluation
def prepare_sample_data(sample_data, feature_cols):
    """Prepare sample data with proper preprocessing for model evaluation."""
    if sample_data is None or sample_data.empty:
        return None, None

    try:
        # If we have preprocessing available, assume data is already processed
        if PREPROCESSING_AVAILABLE:
            sample_X = sample_data.copy()
        else:
            # Apply basic feature engineering
            sample_X = create_basic_features(sample_data)

        # Ensure all required features exist
        for col in feature_cols:
            if col not in sample_X.columns:
                sample_X[col] = 0

        # Select only model features
        sample_X = sample_X[feature_cols].fillna(0)

        # Prepare target variable (log revenue)
        if 'revenue' in sample_data.columns:
            sample_y = np.log1p(sample_data['revenue'].fillna(0).clip(lower=0))
        else:
            st.warning("No 'revenue' column found in sample data")
            sample_y = pd.Series([0] * len(sample_X))

        return sample_X, sample_y

    except Exception as e:
        st.error(f"Error preparing sample data: {e}")
        return None, None

# Sidebar: model selection
def model_selector(models):
    """Render sidebar model selection and return choice."""
    if not models:
        st.sidebar.error("No models available")
        return None

    st.sidebar.header("🤖 Model Selection")
    model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))
    return model_choice

# Debug function
def debug_features(input_df, feature_cols, show_debug=False):
    """Debug feature alignment issues."""
    if show_debug:
        st.write("**Debug Info:**")
        st.write(f"Input shape: {input_df.shape}")
        st.write(f"Required features: {len(feature_cols)}")

        missing_features = [col for col in feature_cols if col not in input_df.columns]
        if missing_features:
            st.write(f"Missing features: {missing_features[:10]}...")

        extra_features = [col for col in input_df.columns if col not in feature_cols]
        if extra_features:
            st.write(f"Extra features: {extra_features[:10]}...")