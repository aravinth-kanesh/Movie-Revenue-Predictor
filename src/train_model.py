from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

def evaluate_model(model, X, y, cv=3):
    """Evaluate a regression model using CV for MSE and R²."""
    mse = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    return mse.mean(), mse.std(), r2.mean(), r2.std()

def train_random_forest(df, feature_cols):
    X, y = df[feature_cols], df["log_revenue"]
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    mse_mean, mse_std, r2_mean, r2_std = evaluate_model(model, X, y)
    model.fit(X, y)
    return model, X, y, mse_mean, mse_std, r2_mean, r2_std

def train_lightgbm(df, feature_cols):
    X, y = df[feature_cols], df["log_revenue"]
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, random_state=42, verbose=-1)
    mse_mean, mse_std, r2_mean, r2_std = evaluate_model(model, X, y)
    model.fit(X, y)
    return model, X, y, mse_mean, mse_std, r2_mean, r2_std