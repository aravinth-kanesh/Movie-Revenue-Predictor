import streamlit as st
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings

# Import project modules
from utils import load_models, create_user_input, model_selector, load_css

warnings.filterwarnings('ignore')

# Load custom CSS
load_css()

# Load models and data
models, feature_cols = load_models()
if models is None or feature_cols is None:
    st.error("Models not loaded. Please check the file paths.")
    st.stop()

# --- Page Content ---
st.header("🧠 SHAP Feature Analysis")

# Sidebar for model selection
model_choice = model_selector(models)
selected_model = models[model_choice]

# Get user input
input_df, _ = create_user_input(feature_cols)

# Make prediction first
prediction_log = selected_model.predict(input_df)[0]
prediction_usd = np.expm1(prediction_log)

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"""
    <div class="prediction-box">
        <h4>Current Prediction</h4>
        <h2>${prediction_usd:,.0f}</h2>
        <p>{model_choice}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("🔍 Feature Impact Analysis")

    @st.cache_data
    def compute_shap_explanation(_model, X, model_name):
        if "Neural Network" in model_name or "MLP" in model_name:
            st.warning("⚠️ SHAP analysis not available for Neural Network models")
            return None
        try:
            explainer = shap.TreeExplainer(_model)
            return explainer(X)
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")
            return None

    shap_explanation = compute_shap_explanation(selected_model, input_df, model_choice)

    if shap_explanation is not None:
        st.markdown('<div class="shap-container">', unsafe_allow_html=True)
        st.subheader("📊 Feature Contribution (Top 15)")
        fig1, _ = plt.subplots(figsize=(10, 6))
        plt.style.use('dark_background')
        shap.summary_plot(shap_explanation, plot_type="bar", show=False, max_display=15)
        plt.tight_layout()
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="shap-container">', unsafe_allow_html=True)
        st.subheader("💧 Waterfall Analysis")
        st.info("Shows how each feature pushes the prediction above or below the expected value")
        fig2, _ = plt.subplots(figsize=(10, 8))
        plt.style.use('dark_background')
        try:
            shap.plots.waterfall(shap_explanation[0], show=False, max_display=15)
            plt.tight_layout()
            st.pyplot(fig2)
        except Exception:
            st.error("Waterfall plot failed.")
        st.markdown('</div>', unsafe_allow_html=True)