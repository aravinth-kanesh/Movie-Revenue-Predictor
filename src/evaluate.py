import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np

def evaluate_model(y_test, y_pred, log_scale=True):
    """Evaluate regression model with MSE, RMSE, R², MAPE; optionally convert log to actual."""
    if log_scale:
        y_test_act, y_pred_act = np.expm1(y_test), np.expm1(y_pred)
    else:
        y_test_act, y_pred_act = y_test, y_pred

    mse = mean_squared_error(y_test_act, y_pred_act)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_act, y_pred_act)
    mape = mean_absolute_percentage_error(y_test_act, y_pred_act)
    return {"mse": mse, "rmse": rmse, "r2": r2, "mape": mape}

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot top_n feature importances for tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-top_n:]
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in idx], importances[idx])
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()
    else:
        print("Feature importance not available for this model type.")

def plot_predictions(y_test, y_pred, log_scale=True):
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
    plt.show()

def full_evaluation(model, X, y, feature_names=None, top_n=20, log_scale=True):
    """Run evaluation metrics, feature importance, and prediction plots in one go."""
    y_pred = model.predict(X)
    metrics = evaluate_model(y, y_pred, log_scale=log_scale)

    if feature_names is not None:
        plot_feature_importance(model, feature_names, top_n=top_n)
    plot_predictions(y, y_pred, log_scale=log_scale)

    return metrics