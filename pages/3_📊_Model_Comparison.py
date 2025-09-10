import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

# Import project modules
from utils import load_models, load_sample_data, model_selector, load_css
from src.train_model import evaluate_model

warnings.filterwarnings('ignore')

# Load custom CSS
load_css()

# Load models and data
models, feature_cols = load_models()
if models is None or feature_cols is None:
    st.error("Models not loaded. Please check the file paths.")
    st.stop()

sample_data = load_sample_data()

# --- Page Content ---
st.header("📊 Model Performance Comparison")

# Sidebar for model selection
model_choice = model_selector(models)
selected_model = models[model_choice]

# Prepare sample data for evaluation
sample_X = sample_data.copy()
for col in feature_cols:
    if col not in sample_X.columns:
        sample_X[col] = 0
sample_X = sample_X[feature_cols]
sample_y = np.log1p(sample_data['revenue'].fillna(0))

# Evaluate all models
st.subheader("⚡ Cross-Validation Performance")
model_metrics = {}
with st.spinner("🔄 Evaluating models..."):
    for name, model in models.items():
        try:
            mse_mean, mse_std, r2_mean, r2_std = evaluate_model(model, sample_X, sample_y, cv=3)
            model_metrics[name] = {
                'MSE': mse_mean, 'MSE_std': mse_std, 'R2': r2_mean,
                'R2_std': r2_std, 'RMSE': np.sqrt(mse_mean)
            }
        except Exception as e:
            st.error(f"Evaluation failed for {name}: {e}")
            model_metrics[name] = None

# Display metrics in cards
valid_metrics_list = [m for m in model_metrics.values() if m is not None]
if valid_metrics_list:
    cols = st.columns(len(valid_metrics_list))
    for i, (name, metrics) in enumerate(model_metrics.items()):
        if metrics is not None:
            with cols[i]:
                st.markdown(f"""
                <div class="model-performance">
                    <h4>{name}</h4>
                    <p><strong>R²:</strong> {metrics['R2']:.3f} ± {metrics['R2_std']:.3f}</p>
                    <p><strong>RMSE:</strong> {metrics['RMSE']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

# Visualization of model performance
st.subheader("📈 Performance Visualization")
valid_metrics = {k: v for k, v in model_metrics.items() if v is not None}
if valid_metrics:
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=list(valid_metrics.keys()), y=[m['R2'] for m in valid_metrics.values()],
        error_y=dict(array=[m['R2_std'] for m in valid_metrics.values()]),
        name='R² Score', marker_color='rgb(55, 83, 109)'
    ))
    fig1.update_layout(title="Model Performance Comparison (R² Score with std)", xaxis_title="Model", yaxis_title="R² Score", showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        x=list(valid_metrics.keys()), y=[m['RMSE'] for m in valid_metrics.values()],
        title="Root Mean Square Error (Lower is Better)", labels={'x': 'Model', 'y': 'RMSE'},
        color=[m['RMSE'] for m in valid_metrics.values()], color_continuous_scale='Reds_r'
    )
    st.plotly_chart(fig2, use_container_width=True)

# Feature importance for selected model
st.subheader("🔍 Feature Importance Analysis")
if hasattr(selected_model, "feature_importances_"):
    st.markdown(f'<div class="feature-importance">', unsafe_allow_html=True)
    st.write(f"**{model_choice} - Top Feature Importances**")
    importances = selected_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False).head(15)
    fig3 = px.bar(
        importance_df, x='Importance', y='Feature', orientation='h',
        title=f"Top 15 Feature Importances - {model_choice}",
        color='Importance', color_continuous_scale='Viridis'
    )
    fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info(f"Feature importance not available for {model_choice}")