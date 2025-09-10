import streamlit as st
import warnings
from utils import load_css

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Revenue Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
load_css()

# Title and description
st.markdown('<h1 class="main-header">🎬 Movie Revenue Predictor</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
    Predict movie revenues with pre-trained ML models and visualise feature impacts with SHAP.
    </p>
</div>
""", unsafe_allow_html=True)

st.info("👈 Select an analysis mode from the sidebar to get started!")