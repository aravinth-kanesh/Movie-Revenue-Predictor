import pandas as pd
import numpy as np

def preprocess(path, nrows=None):
    df = pd.read_csv(path, nrows=nrows)  # Load CSV, limit rows
    df = df[df['revenue'].notnull() & (df['revenue'] > 0)]  # Keep only movies with positive revenue

    # Extract date-based features
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year.fillna(0).astype(int)
        df['release_month'] = df['release_date'].dt.month.fillna(0).astype(int)
        df['release_quarter'] = df['release_date'].dt.quarter.fillna(0).astype(int)
        df['release_season'] = (df['release_month'] % 12 // 3 + 1).fillna(0).astype(int)
    else:
        df['release_year'] = df['release_month'] = df['release_quarter'] = df['release_season'] = 0

    # Handle runtime
    if 'runtime' in df.columns:
        df['runtime'] = df['runtime'].fillna(df['runtime'].median())
        df['log_runtime'] = np.log1p(df['runtime'].clip(lower=0))
    else:
        df['log_runtime'] = 0

    # Budget
    if 'budget' in df.columns:
        df['budget'] = df['budget'].fillna(0)
        df['log_budget'] = np.log1p(df['budget'].clip(lower=0))
    else:
        df['log_budget'] = 0

    # Popularity
    if 'popularity' in df.columns:
        df['popularity'] = df['popularity'].fillna(0)
    else:
        df['popularity'] = 0

    # Target
    df['log_revenue'] = np.log1p(df['revenue'])

    df.fillna(0, inplace=True)
    return df