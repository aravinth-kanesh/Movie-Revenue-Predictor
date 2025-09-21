import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings

from utils import load_models, load_sample_data, model_selector, load_css, prepare_sample_data

warnings.filterwarnings('ignore')

# --- Load Custom CSS ---
load_css()

# --- Load Models and Sample Data ---
models, feature_cols = load_models()
if models is None or feature_cols is None:
    st.error("Models not loaded. Please check the file paths.")
    st.stop()

sample_data = load_sample_data()

# --- Page Header ---
st.header("📈 Sample Data Analysis & Predictions")

# Sidebar: model selection
model_choice = model_selector(models)
if model_choice is None:
    st.stop()

selected_model = models[model_choice]

# --- Prepare Data ---
sample_X, sample_y = prepare_sample_data(sample_data, feature_cols)

if sample_X is None or sample_y is None:
    st.error("Could not prepare sample data for analysis.")
    st.stop()

# Check if we have revenue data for meaningful analysis
has_revenue_data = 'revenue' in sample_data.columns and sample_data['revenue'].notna().any()

if not has_revenue_data:
    st.warning("⚠️ No revenue data found in sample dataset. Analysis will be limited.")

# --- Model Predictions ---
try:
    sample_pred = selected_model.predict(sample_X)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# --- Actual vs Predicted Analysis (only if we have revenue data) ---
if has_revenue_data:
    st.subheader("🎯 Model Predictions vs Actual Revenue")
    col1, col2 = st.columns(2)

    with col1:
        # Filter out zero revenues for better visualization
        non_zero_mask = sample_y > 0
        if non_zero_mask.sum() > 10:  # Only plot if we have enough non-zero data
            sample_y_filtered = sample_y[non_zero_mask]
            sample_pred_filtered = sample_pred[non_zero_mask]

            fig1 = px.scatter(
                x=sample_y_filtered, y=sample_pred_filtered,
                title=f"Actual vs Predicted Revenue (log) - {model_choice}",
                labels={'x': 'Actual Revenue (log)', 'y': 'Predicted Revenue (log)'},
                opacity=0.6
            )

            # Perfect prediction line
            min_val = min(sample_y_filtered.min(), sample_pred_filtered.min())
            max_val = max(sample_y_filtered.max(), sample_pred_filtered.max())
            fig1.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Not enough non-zero revenue data for scatter plot")

    with col2:
        if has_revenue_data and non_zero_mask.sum() > 10:
            residuals = sample_y_filtered - sample_pred_filtered
            fig2 = px.scatter(
                x=sample_pred_filtered, y=residuals,
                title="Residuals Plot",
                labels={'x': 'Predicted Revenue (log)', 'y': 'Residuals'},
                opacity=0.6
            )
            fig2.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Residuals plot not available")

# --- Revenue Distribution ---
st.subheader("📊 Revenue Distribution Analysis")
col1, col2 = st.columns(2)

with col1:
    if has_revenue_data:
        # Filter out very low revenues for better visualisation
        revenue_data = sample_data['revenue'][sample_data['revenue'] > 1000]
        if len(revenue_data) > 0:
            fig3 = px.histogram(
                x=revenue_data, nbins=30,
                title="Actual Revenue Distribution (>$1K)",
                labels={'x': 'Revenue ($)', 'y': 'Count'}
            )
            fig3.update_layout(xaxis=dict(tickformat='$,.0f'))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No significant revenue data for histogram")
    else:
        st.info("No actual revenue data available")

with col2:
    pred_revenue_actual = np.expm1(sample_pred)
    # Filter predictions for better visualisation
    pred_filtered = pred_revenue_actual[pred_revenue_actual > 1000]

    if len(pred_filtered) > 0:
        fig4 = px.histogram(
            x=pred_filtered, nbins=30,
            title=f"Predicted Revenue Distribution - {model_choice}",
            labels={'x': 'Predicted Revenue ($)', 'y': 'Count'}
        )
        fig4.update_layout(xaxis=dict(tickformat='$,.0f'))
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No significant predicted revenue data")

# --- Performance Metrics ---
st.subheader("📈 Performance Summary")

if has_revenue_data and non_zero_mask.sum() > 10:
    try:
        # Use filtered data for metrics
        mse = mean_squared_error(sample_y_filtered, sample_pred_filtered)
        rmse = np.sqrt(mse)
        r2 = r2_score(sample_y_filtered, sample_pred_filtered)

        # Convert to actual revenue for MAPE calculation
        actual_revenue = np.expm1(sample_y_filtered)
        pred_revenue = np.expm1(sample_pred_filtered)

        # Only calculate MAPE where actual revenue > 0
        valid_mape_mask = actual_revenue > 0
        if valid_mape_mask.sum() > 0:
            mape = mean_absolute_percentage_error(
                actual_revenue[valid_mape_mask],
                pred_revenue[valid_mape_mask]
            )
        else:
            mape = 0

    except Exception as e:
        st.warning(f"Metric computation failed: {e}")
        mse = rmse = r2 = mape = 0

    cols = st.columns(4)
    cols[0].markdown(f'<div class="metric-card"><h3>{mse:.3f}</h3><p>MSE (log)</p></div>', unsafe_allow_html=True)
    cols[1].markdown(f'<div class="metric-card"><h3>{rmse:.3f}</h3><p>RMSE (log)</p></div>', unsafe_allow_html=True)
    cols[2].markdown(f'<div class="metric-card"><h3>{r2:.3f}</h3><p>R² Score</p></div>', unsafe_allow_html=True)
    cols[3].markdown(f'<div class="metric-card"><h3>{mape:.1%}</h3><p>MAPE</p></div>', unsafe_allow_html=True)

else:
    st.info("Performance metrics not available - insufficient revenue data")

    # Show basic prediction statistics instead
    cols = st.columns(3)
    cols[0].markdown(f'<div class="metric-card"><h3>{len(sample_pred)}</h3><p>Predictions Made</p></div>',
                     unsafe_allow_html=True)
    cols[1].markdown(
        f'<div class="metric-card"><h3>${np.expm1(sample_pred).mean():,.0f}</h3><p>Avg Predicted Revenue</p></div>',
        unsafe_allow_html=True)
    cols[2].markdown(
        f'<div class="metric-card"><h3>${np.expm1(sample_pred).std():,.0f}</h3><p>Prediction Std Dev</p></div>',
        unsafe_allow_html=True)

# --- Additional Data Insights ---
if sample_data is not None:
    st.subheader("💡 Dataset Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Movies", len(sample_data))

    with col2:
        if has_revenue_data:
            avg_revenue = sample_data['revenue'][sample_data['revenue'] > 0].mean()
            st.metric("Avg Revenue", f"${avg_revenue:,.0f}")
        else:
            st.metric("Revenue Data", "Not Available")