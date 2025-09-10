import pandas as pd
import numpy as np

def preprocess(path, nrows=None):
    """Load TMDB dataset, clean, and add basic engineered features."""
    df = pd.read_csv(path, nrows=nrows)

    # Drop rows without revenue
    df = df[df['revenue'].notnull() & (df['revenue'] > 0)]

    # Basic cleaning
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        df['release_quarter'] = df['release_date'].dt.quarter
        df['release_season'] = df['release_date'].dt.month % 12 // 3 + 1

    # Handle runtime
    if 'runtime' in df.columns:
        df['runtime'] = df['runtime'].fillna(df['runtime'].median())
        df['log_runtime'] = np.log1p(df['runtime'])

    # Weighted vote average
    if 'vote_count' in df.columns and 'vote_average' in df.columns:
        v = df['vote_count']
        R = df['vote_average']
        C = df['vote_average'].mean()
        m = df['vote_count'].quantile(0.7)
        df['weighted_vote'] = (v/(v+m) * R) + (m/(m+v) * C)

    # Log-transform target
    df['log_revenue'] = np.log1p(df['revenue'])

    return df