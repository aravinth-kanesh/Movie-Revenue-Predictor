import os
import joblib
import numpy as np
from src.preprocess import preprocess
from src.feature_engineering import expand_features
from src.train_model import train_random_forest, train_lightgbm

def ensure_numeric_features(df, feature_list):
    present = [c for c in feature_list if c in df.columns]
    numeric = df[present].select_dtypes(include=[np.number]).columns.tolist()
    return numeric

if __name__ == "__main__":
    data_path = "data/tmdb_movie_dataset.csv"
    nrows = 5000
    models_dir = "data/models"
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading and preprocessing data from: {data_path} (nrows={nrows})")
    df = preprocess(data_path, nrows=nrows)

    print("Expanding features...")
    df, new_features = expand_features(df)

    core_features = [
        "vote_average",
        "vote_count",
        "log_runtime",
        "weighted_vote",
        "release_year",
        "release_month",
        "release_quarter",
        "release_season"
    ]

    # Candidate = raw + engineered + expanded
    candidate_features = list(dict.fromkeys(core_features + new_features))
    feature_cols = ensure_numeric_features(df, candidate_features)

    if not feature_cols:
        raise RuntimeError("No numeric features found for training.")

    print(f"Feature columns used for training ({len(feature_cols)}): {feature_cols[:20]}{'...' if len(feature_cols)>20 else ''}")

    feature_cols_path = os.path.join(models_dir, "feature_cols.pkl")
    joblib.dump(feature_cols, feature_cols_path)
    print(f"✅ Saved feature columns -> {feature_cols_path}")

    model_funcs = {
        "random_forest": train_random_forest,
        "lightgbm": train_lightgbm
    }

    for name, func in model_funcs.items():
        try:
            print(f"\n🔄 Training {name} ...")
            model, X_used, y_used, mse_mean, mse_std, r2_mean, r2_std = func(df, feature_cols)
            model_path = os.path.join(models_dir, f"{name}.pkl")
            joblib.dump(model, model_path)
            print(f"✅ Saved {name} -> {model_path}")
            print(f"   CV metrics: MSE = {mse_mean:.4f} ± {mse_std:.4f}, R² = {r2_mean:.4f} ± {r2_std:.4f}")
        except Exception as e:
            print(f"❌ Failed training {name}: {e}")
            continue

    print("\n🎉 All done. Models and feature columns saved to:", os.path.abspath(models_dir))