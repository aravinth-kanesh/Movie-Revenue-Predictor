from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

def evaluate_model_cv(model, X, y, cv=3):
    """Fast CV evaluation"""
    mse = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=2)
    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=2)
    return mse.mean(), mse.std(), r2.mean(), r2.std()

def train_random_forest(df, feature_cols, full_eval=False, save_dir=None):
    X, y = df[feature_cols], df["log_revenue"]

    # Fast, balanced Random Forest
    model = RandomForestRegressor(
        n_estimators=100,  # Reduced for speed
        max_depth=6,  # Balanced depth
        min_samples_split=30,
        min_samples_leaf=15,
        max_features=0.4,  # Prevent budget dominance
        n_jobs=2,  # Limit for 8GB RAM
        random_state=42
    )

    mse_mean, mse_std, r2_mean, r2_std = evaluate_model_cv(model, X, y, cv=3)
    model.fit(X, y)
    return model, X, y, mse_mean, mse_std, r2_mean, r2_std

def train_lightgbm(df, feature_cols, full_eval=False, save_dir=None):
    X, y = df[feature_cols], df["log_revenue"]

    # Fast, well-regularized LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=200,  # Moderate number
        learning_rate=0.08,
        max_depth=6,
        num_leaves=30,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=1  # Single thread for memory
    )

    mse_mean, mse_std, r2_mean, r2_std = evaluate_model_cv(model, X, y, cv=3)
    model.fit(X, y)
    return model, X, y, mse_mean, mse_std, r2_mean, r2_std