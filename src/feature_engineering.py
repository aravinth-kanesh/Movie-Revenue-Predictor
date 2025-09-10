import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# --- Core numeric + basic features --- #
def create_basic_features(df):
    """Numeric and temporal features for both training and app input."""

    # Runtime
    if 'runtime' in df:
        df['log_runtime'] = np.log1p(df['runtime'].fillna(0))
    else:
        df['log_runtime'] = 0

    # Weighted vote
    if 'vote_average' in df and 'vote_count' in df:
        df['weighted_vote'] = df['vote_average'] * np.log1p(df['vote_count'].fillna(0))
    else:
        df['weighted_vote'] = 0

    # Release date handling
    if 'release_date' in df:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year.fillna(0)
        df['release_month'] = df['release_date'].dt.month.fillna(0)
        df['release_quarter'] = df['release_date'].dt.quarter.fillna(0)
        df['release_season'] = df['release_month'].apply(lambda x: (x % 12 + 3) // 3 if x > 0 else 0)
    else:
        df['release_year'] = 0
        df['release_month'] = 0
        df['release_quarter'] = 0
        df['release_season'] = 0

    df.fillna(0, inplace=True)
    return df

# --- Multi-hot categorical features --- #
def add_topN_multihot(df, col, top_n=10):
    """Keep only top-N categories for multi-label column and one-hot encode."""
    if col not in df:
        return df

    df[col] = df[col].fillna('').astype(str)

    # Split multi-labels
    all_labels = [l.strip() for sublist in df[col].str.split(',') for l in sublist if l]
    top_labels = pd.Series(all_labels).value_counts().head(top_n).index

    for label in top_labels:
        safe_name = re.sub(r'[^A-Za-z0-9_]+', '_', f'{col}_{label}')
        df[safe_name] = df[col].apply(lambda x: int(label in x))
    return df

# --- Text features (TF-IDF + SVD) --- #
def add_text_svd(df, col, svd_dim=10):
    """Add low-dimensional TF-IDF SVD features for text column."""
    if col not in df:
        return df

    df[col] = df[col].fillna('').astype(str)
    vectoriser = TfidfVectorizer(max_features=500, stop_words='english')
    X_tfidf = vectoriser.fit_transform(df[col])
    n_components = min(svd_dim, X_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)

    for i in range(X_svd.shape[1]):
        df[f'{col}_svd_{i}'] = X_svd[:, i]
    return df

# --- Full fast feature expansion --- #
def expand_features_fast(df):
    df = create_basic_features(df)

    # Multi-hot top-N
    df = add_topN_multihot(df, 'genres', top_n=10)
    df = add_topN_multihot(df, 'production_companies', top_n=20)
    if 'spoken_languages' in df.columns:
        df = add_topN_multihot(df, 'spoken_languages', top_n=5)

    # Text features
    df = add_text_svd(df, 'overview', svd_dim=10)
    df = add_text_svd(df, 'tagline', svd_dim=5)

    # Collect new feature names
    new_features = [
        c for c in df.columns
        if re.search(r'_svd_|genres_|production_companies_|spoken_languages_', c)
    ]

    return df, new_features