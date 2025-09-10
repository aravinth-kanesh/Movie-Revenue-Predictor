from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Evaluate a model using cross-validation
def evaluate_model(model, X, y, cv=3):
    mse = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1)
    r2 = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)

    return mse.mean(), mse.std(), r2.mean(), r2.std()

# Train Random Forest
def train_random_forest(df, feature_cols):
    X, y = df[feature_cols], df["log_revenue"]
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    mse_mean, mse_std, r2_mean, r2_std = evaluate_model(model, X, y)  # CV metrics
    model.fit(X, y)  # Fit full model

    return model, X, y, mse_mean, mse_std, r2_mean, r2_std

# Train LightGBM
def train_lightgbm(df, feature_cols):
    X, y = df[feature_cols], df["log_revenue"]
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.1, random_state=42, verbose=-1)
    mse_mean, mse_std, r2_mean, r2_std = evaluate_model(model, X, y)
    model.fit(X, y)

    return model, X, y, mse_mean, mse_std, r2_mean, r2_std

# Train Gradient Boosting
def train_gradient_boost(df, feature_cols):
    X, y = df[feature_cols], df["log_revenue"]
    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    mse_mean, mse_std, r2_mean, r2_std = evaluate_model(model, X, y)
    model.fit(X, y)

    return model, X, y, mse_mean, mse_std, r2_mean, r2_std

# Train XGBoost
def train_xgb(df, feature_cols):
    X, y = df[feature_cols], df["log_revenue"]
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                             n_jobs=-1, objective='reg:squarederror', random_state=42)
    mse_mean, mse_std, r2_mean, r2_std = evaluate_model(model, X, y)
    model.fit(X, y)

    return model, X, y, mse_mean, mse_std, r2_mean, r2_std

# Train MLP with scaling
def train_mlp(df, feature_cols):
    X, y = df[feature_cols], df["log_revenue"]
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale features
        ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
    ])
    mse_mean, mse_std, r2_mean, r2_std = evaluate_model(pipeline, X, y)
    pipeline.fit(X, y)

    return pipeline, X, y, mse_mean, mse_std, r2_mean, r2_std