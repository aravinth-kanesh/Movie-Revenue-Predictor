import os
import joblib
import numpy as np
import gc
import time
from src.preprocess import preprocess
from src.feature_engineering import expand_features
from src.train_model import train_random_forest, train_lightgbm

def ensure_numeric_features(df, feature_list):
    present = [c for c in feature_list if c in df.columns]
    numeric = df[present].select_dtypes(include=[np.number]).columns.tolist()
    return numeric

def optimise_for_8gb_ram(df):
    """Memory optimisation for 8GB systems"""
    # Convert to smaller dtypes
    for col in df.select_dtypes(['int64']):
        if df[col].max() < 2147483647:
            df[col] = df[col].astype('int32')

    for col in df.select_dtypes(['float64']):
        if col != 'log_revenue':  # Keep target at full precision
            df[col] = df[col].astype('float32')

    return df

if __name__ == "__main__":
    # Optimal settings for 8GB RAM
    data_path = "data/tmdb_movie_dataset.csv"
    nrows = 1000000
    models_dir = "data/models"
    os.makedirs(models_dir, exist_ok=True)

    print(f"8GB RAM Optimised Training:")
    print(f"   Sample size: {nrows:,} records")
    print(f"   Expected time: 3-5 minutes")

    # Load and optimise data
    start_time = time.time()
    df = preprocess(data_path, nrows=nrows)
    df = optimise_for_8gb_ram(df)
    print(f"   Loaded {len(df):,} records in {time.time() - start_time:.1f}s")

    # Feature engineering
    print("Creating features...")
    df, new_features = expand_features(df)

    # Force garbage collection
    gc.collect()

    # Core features (remove budget_millions to reduce budget dominance)
    core_features = [
        "vote_average", "vote_count", "log_runtime",
        "release_year", "release_month", "release_quarter", "release_season",
        "log_budget", "popularity"  # Keep log_budget but remove budget_millions
    ]

    candidate_features = list(dict.fromkeys(core_features + new_features))
    feature_cols = ensure_numeric_features(df, candidate_features)

    # Feature selection to improve balance
    if len(feature_cols) > 20:
        # Limit features to prevent overfitting and improve speed
        correlations = df[feature_cols].corrwith(df['log_revenue']).abs()

        # Force diversity: take top budget feature + diverse others
        budget_features = [f for f in feature_cols if 'budget' in f.lower()]
        non_budget = [f for f in feature_cols if 'budget' not in f.lower()]

        # Select balanced feature set
        top_budget = correlations[budget_features].nlargest(1).index.tolist()
        top_others = correlations[non_budget].nlargest(15).index.tolist()
        feature_cols = top_budget + top_others

        print(f"Selected {len(feature_cols)} balanced features")

    print(f"Training features: {feature_cols}")

    # Save features
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_cols.pkl"))

    # Train models with memory management
    models = {"random_forest": train_random_forest, "lightgbm": train_lightgbm}

    for name, func in models.items():
        try:
            print(f"\nTraining {name}...")
            start_time = time.time()

            # Train without full evaluation to save time/memory
            model, X, y, mse_mean, mse_std, r2_mean, r2_std = func(
                df, feature_cols, full_eval=False, save_dir=None
            )

            # Save model
            joblib.dump(model, os.path.join(models_dir, f"{name}.pkl"))

            training_time = time.time() - start_time
            print(f"   Completed in {training_time:.1f}s")
            print(f"   R² = {r2_mean:.3f} ± {r2_std:.3f}")

            # Memory cleanup
            del model
            gc.collect()

        except Exception as e:
            print(f"Failed {name}: {e}")

    print(f"\nTraining complete! Check your Streamlit app.")