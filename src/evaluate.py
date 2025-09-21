import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import numpy as np
import os

def evaluate_model(y_test, y_pred, log_scale=True):
    """Evaluate regression model with MSE, RMSE, MAE, R², MAPE; optionally convert log to actual."""
    if log_scale:
        y_test_act, y_pred_act = np.expm1(y_test), np.expm1(y_pred)
    else:
        y_test_act, y_pred_act = y_test, y_pred

    mse = mean_squared_error(y_test_act, y_pred_act)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_act, y_pred_act)
    r2 = r2_score(y_test_act, y_pred_act)
    mape = mean_absolute_percentage_error(y_test_act, y_pred_act)

    # Print metrics neatly
    print(f"MSE: {mse:.2f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f} | MAPE: {mape:.2%}")
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot top_n feature importances for tree-based models; optionally save to file."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-top_n:]

        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in idx], importances[idx])
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
        plt.show()
    else:
        print("Feature importance not available for this model type.")

def plot_predictions(y_test, y_pred, log_scale=True, save_path=None):
    """Plot actual vs predicted with diagonal reference; supports log->actual conversion."""
    if log_scale:
        y_test, y_pred = np.expm1(y_test), np.expm1(y_pred)

    plt.style.use('dark_background')
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    plt.xlabel("Actual Revenue ($)" if log_scale else "Actual")
    plt.ylabel("Predicted Revenue ($)" if log_scale else "Predicted")
    plt.title("Actual vs Predicted Revenue")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()

def full_evaluation(model, X, y, feature_names=None, top_n=20, log_scale=True, save_dir=None):
    """
    Run evaluation metrics, feature importance, and prediction plots in one go.

    Parameters:
    - model: trained regression model
    - X: features
    - y: target (log or raw)
    - feature_names: list of feature names
    - top_n: top features to show
    - log_scale: if True, convert log1p to actual dollars
    - save_dir: if provided, saves plots to this folder
    """
    y_pred = model.predict(X)
    metrics = evaluate_model(y, y_pred, log_scale=log_scale)

    if feature_names is not None:
        fi_path = f"{save_dir}/feature_importance.png" if save_dir else None
        plot_feature_importance(model, feature_names, top_n=top_n, save_path=fi_path)

    pred_path = f"{save_dir}/predictions.png" if save_dir else None
    plot_predictions(y, y_pred, log_scale=log_scale, save_path=pred_path)

    return metrics