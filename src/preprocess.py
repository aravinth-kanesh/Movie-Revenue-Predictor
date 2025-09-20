import pandas as pd
import numpy as np

def preprocess(path, nrows=None):
    df = pd.read_csv(path, nrows=nrows)  # Load CSV, limit rows
    df = df[df['revenue'].notnull() & (df['revenue'] > 0)]  # Keep only movies with positive revenue

    # Extract date-based features
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year.fillna(0).astype(int)  # Year of release
        df['release_month'] = df['release_date'].dt.month.fillna(0).astype(int)  # Month of release
        df['release_quarter'] = df['release_date'].dt.quarter.fillna(0).astype(int)  # Quarter of release
        df['release_season'] = (df['release_month'] % 12 // 3 + 1).fillna(0).astype(int)  # Season (1-4)
    else:
        df['release_year'] = df['release_month'] = df['release_quarter'] = df['release_season'] = 0

    # Handle runtime
    if 'runtime' in df.columns:
        df['runtime'] = df['runtime'].fillna(df['runtime'].median())  # Fill missing with median
        df['log_runtime'] = np.log1p(df['runtime'].clip(lower=0))  # Log-transform runtime

    # Compute weighted vote
    if 'vote_count' in df.columns and 'vote_average' in df.columns:
        v = df['vote_count']
        R = df['vote_average']
        C = df['vote_average'].mean()
        m = df['vote_count'].quantile(0.7)
        df['weighted_vote'] = (v/(v+m) * R) + (m/(m+v) * C)  # Bayesian weighted rating
    else:
        df['weighted_vote'] = 0

    df['log_revenue'] = np.log1p(df['revenue'])  # Log-transform target
    df.fillna(0, inplace=True)  # Fill any remaining NaNs
    return df