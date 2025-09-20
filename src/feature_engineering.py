import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def create_basic_features(df):
    # Numeric features
    df['log_runtime'] = np.log1p(df['runtime'].fillna(0)) if 'runtime' in df else 0
    df['weighted_vote'] = df['vote_average'].fillna(0) * np.log1p(df['vote_count'].fillna(0)) \
        if 'vote_average' in df and 'vote_count' in df else 0

    # Release date features
    if 'release_date' in df:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['release_year'] = df['release_date'].dt.year.fillna(0).astype(int)
        df['release_month'] = df['release_date'].dt.month.fillna(0).astype(int)
    else:
        df['release_year'] = df['release_month'] = 0

    df.fillna(0, inplace=True)
    return df

def add_topN_multihot(df, col, top_n=5):
    if col not in df:
        return df

    df[col] = df[col].fillna('').astype(str)
    all_labels = [l.strip() for sublist in df[col].str.split(',') for l in sublist if l]
    top_labels = list(pd.Series(all_labels).value_counts().head(top_n).index)

    for label in top_labels:
        safe_name = re.sub(r'[^A-Za-z0-9_]+', '_', f'{col}_{label}')
        df[safe_name] = df[col].apply(lambda x: int(label in x))

    return df

def add_text_svd(df, col, svd_dim=5, max_features=200):
    if col not in df:
        return df

    df[col] = df[col].fillna('').astype(str)
    vectoriser = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = vectoriser.fit_transform(df[col])
    if X_tfidf.shape[1] < 1:
        return df  # skip if empty
    n_components = min(svd_dim, X_tfidf.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)

    for i in range(X_svd.shape[1]):
        df[f'{col}_svd_{i}'] = X_svd[:, i]

    return df

def expand_features(df):
    df = create_basic_features(df)
    df = add_topN_multihot(df, 'genres', top_n=5)
    df = add_topN_multihot(df, 'production_companies', top_n=10)
    df = add_text_svd(df, 'overview', svd_dim=5)
    df = add_text_svd(df, 'tagline', svd_dim=3)

    new_features = [c for c in df.columns if re.search(r'_svd_|genres_|production_companies_', c)]
    return df, new_features