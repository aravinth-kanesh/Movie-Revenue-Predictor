import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
from utils import load_models, load_sample_data, load_css, prepare_sample_data

# Add src to path for train_model import
import sys

if 'src' not in sys.path:
    sys.path.append('src')

try:
    from src.train_model import evaluate_model_cv
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False
    st.warning("⚠️ Evaluation module not available. Using basic metrics only.")

warnings.filterwarnings('ignore')

# Load custom CSS
load_css()

# Load models and data
models, feature_cols = load_models()
if models is None or feature_cols is None:
    st.error("Models not loaded. Please check the file paths.")
    st.stop()

sample_data = load_sample_data()

# --- Page Header ---
st.header("📊 Model Performance Comparison")

# Prepare sample data for evaluation
sample_X, sample_y = prepare_sample_data(sample_data, feature_cols)

if sample_X is None or sample_y is None:
    st.error("Could not prepare sample data for evaluation.")
    st.stop()

# --- Cross-validation metrics ---
st.subheader("⚡ Model Performance Evaluation")

if EVALUATION_AVAILABLE:
    model_metrics = {}
    with st.spinner("🔥 Evaluating models..."):
        for name, model in models.items():
            try:
                mse_mean, mse_std, r2_mean, r2_std = evaluate_model_cv(model, sample_X, sample_y, cv=3)
                model_metrics[name] = {
                    'MSE': mse_mean,
                    'MSE_std': mse_std,
                    'R2': r2_mean,
                    'R2_std': r2_std,
                    'RMSE': np.sqrt(mse_mean)
                }
            except Exception as e:
                st.warning(f"Evaluation failed for {name}: {e}")
                model_metrics[name] = None
else:
    # Fallback: simple train-test evaluation
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    model_metrics = {}

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            sample_X, sample_y, test_size=0.2, random_state=42
        )

        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                model_metrics[name] = {
                    'MSE': mse,
                    'MSE_std': 0,  # No std for single evaluation
                    'R2': r2,
                    'R2_std': 0,
                    'RMSE': np.sqrt(mse)
                }
            except Exception as e:
                st.warning(f"Evaluation failed for {name}: {e}")
                model_metrics[name] = None

    except Exception as e:
        st.error(f"Could not evaluate models: {e}")
        model_metrics = {}

# Display metrics in cards
valid_metrics = {k: v for k, v in model_metrics.items() if v is not None}

if valid_metrics:
    cols = st.columns(len(valid_metrics))
    for i, (name, metrics) in enumerate(valid_metrics.items()):
        with cols[i]:
            std_text = f" ± {metrics['R2_std']:.3f}" if metrics['R2_std'] > 0 else ""
            st.markdown(f"""
            <div class="model-performance">
                <h4>{name}</h4>
                <p><strong>R²:</strong> {metrics['R2']:.3f}{std_text}</p>
                <p><strong>RMSE:</strong> {metrics['RMSE']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("No valid model metrics available.")

# --- Performance Visualisation ---
if valid_metrics:
    st.subheader("📈 Performance Visualisation")

    # R² with error bars (if available)
    fig1 = go.Figure()

    if any(m['R2_std'] > 0 for m in valid_metrics.values()):
        # With error bars
        fig1.add_trace(go.Bar(
            x=list(valid_metrics.keys()),
            y=[m['R2'] for m in valid_metrics.values()],
            error_y=dict(array=[m['R2_std'] for m in valid_metrics.values()]),
            marker_color='rgb(55, 83, 109)'
        ))
    else:
        # Without error bars
        fig1.add_trace(go.Bar(
            x=list(valid_metrics.keys()),
            y=[m['R2'] for m in valid_metrics.values()],
            marker_color='rgb(55, 83, 109)'
        ))

    fig1.update_layout(
        title="Model R² Score Comparison",
        xaxis_title="Model",
        yaxis_title="R² Score",
        showlegend=False
    )
    st.plotly_chart(fig1, use_container_width=True)

    # RMSE bar chart
    fig2 = px.bar(
        x=list(valid_metrics.keys()),
        y=[m['RMSE'] for m in valid_metrics.values()],
        title="Root Mean Square Error (Lower is Better)",
        labels={'x': 'Model', 'y': 'RMSE'},
        color=[m['RMSE'] for m in valid_metrics.values()],
        color_continuous_scale='Reds_r'
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- Feature importance for all models ---
st.subheader("🔍 Feature Importance Analysis")

# Show feature importance for all models that support it
for model_name, model in models.items():
    try:
        if hasattr(model, "feature_importances_"):
            st.subheader(f"📊 {model_name} - Top 15 Features")

            importances = model.feature_importances_
            importance_df = (
                pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
                .sort_values('Importance', ascending=False)
                .head(15)
            )

            fig = px.bar(
                importance_df,
                x='Importance', y='Feature', orientation='h',
                color='Importance',
                color_continuous_scale='Viridis',
                title=f"Feature Importances - {model_name}"
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Feature importance not available for {model_name}")

    except Exception as e:
        st.warning(f"Feature importance plot failed for {model_name}: {e}")