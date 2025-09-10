import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np

def evaluate_model(y_test, y_pred):
    """Evaluate regression model performance with multiple metrics."""
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.2f}")
    print(f"MAPE: {mape:.2%}")

    return {"mse": mse, "rmse": rmse, "r2": r2, "mape": mape}

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot feature importances for tree-based models, sorted by importance.
    Shows top_n most important features.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-top_n:]  # top N features
        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in idx], importances[idx])
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()
    else:
        print("Feature importance not available for this model type.")

def plot_predictions(y_test, y_pred):
    """Plot actual vs predicted values with diagonal reference line."""
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    min_val, max_val = y_test.min(), y_test.max()
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    plt.xlabel("Actual log_revenue")
    plt.ylabel("Predicted log_revenue")
    plt.title("Actual vs Predicted Revenue")
    plt.tight_layout()
    plt.show()

def full_evaluation(model, X, y, feature_names=None, top_n=20):
    """Run evaluation metrics, feature importance (if available), and prediction plots in one go."""
    y_pred = model.predict(X)
    metrics = evaluate_model(y, y_pred)

    if feature_names is not None:
        plot_feature_importance(model, feature_names, top_n=top_n)
    plot_predictions(y, y_pred)

    return metrics