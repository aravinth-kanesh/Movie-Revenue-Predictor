import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings

# Import project modules
from utils import load_models, load_sample_data, model_selector, load_css

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
st.header("📈 Sample Data Analysis & Predictions")

# Sidebar for model selection
model_choice = model_selector(models)
selected_model = models[model_choice]

# Prepare data
sample_X = sample_data.copy()
for col in feature_cols:
    if col not in sample_X.columns:
        sample_X[col] = 0
sample_X = sample_X[feature_cols]
sample_y = np.log1p(sample_data['revenue'].fillna(0))

# Make predictions
sample_pred = selected_model.predict(sample_X)

# Actual vs Predicted scatter plot
st.subheader("🎯 Model Predictions on Sample Data")
col1, col2 = st.columns(2)
with col1:
    fig1 = px.scatter(
        x=sample_y, y=sample_pred, title=f"Actual vs Predicted Revenue - {model_choice}",
        labels={'x': 'Actual Revenue (log)', 'y': 'Predicted Revenue (log)'}, opacity=0.6
    )
    min_val, max_val = min(sample_y.min(), sample_pred.min()), max(sample_y.max(), sample_pred.max())
    fig1.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    residuals = sample_y - sample_pred
    fig2 = px.scatter(x=sample_pred, y=residuals, title="Residuals Plot", labels={'x': 'Predicted Revenue (log)', 'y': 'Residuals'}, opacity=0.6)
    fig2.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig2, use_container_width=True)

# Distribution plots
st.subheader("📊 Revenue Distribution Analysis")
col1, col2 = st.columns(2)
with col1:
    fig3 = px.histogram(sample_data, x='revenue', nbins=30, title="Actual Revenue Distribution", labels={'revenue': 'Revenue ($)', 'count': 'Count'})
    st.plotly_chart(fig3, use_container_width=True)
with col2:
    pred_revenue_actual = np.expm1(sample_pred)
    fig4 = px.histogram(x=pred_revenue_actual, nbins=30, title=f"Predicted Revenue Distribution - {model_choice}", labels={'x': 'Predicted Revenue ($)', 'y': 'Count'})
    st.plotly_chart(fig4, use_container_width=True)

# Performance metrics
st.subheader("📈 Performance Summary")
mse = mean_squared_error(sample_y, sample_pred)
rmse = np.sqrt(mse)
r2 = r2_score(sample_y, sample_pred)
mape = mean_absolute_percentage_error(np.expm1(sample_y), np.expm1(sample_pred))
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f'<div class="metric-card"><h3>{mse:.3f}</h3><p>MSE</p></div>', unsafe_allow_html=True)
col2.markdown(f'<div class="metric-card"><h3>{rmse:.3f}</h3><p>RMSE</p></div>', unsafe_allow_html=True)
col3.markdown(f'<div class="metric-card"><h3>{r2:.3f}</h3><p>R² Score</p></div>', unsafe_allow_html=True)
col4.markdown(f'<div class="metric-card"><h3>{mape:.1%}</h3><p>MAPE</p></div>', unsafe_allow_html=True)