import streamlit as st
import numpy as np
import warnings

# Import project modules
from src.app_utils import load_models, create_user_input, model_selector, load_css

warnings.filterwarnings('ignore')

# Load custom CSS
load_css()

# Load models and data
models, feature_cols = load_models()
if models is None or feature_cols is None:
    st.error("Models not loaded. Please check the file paths.")
    st.stop()

# --- Page Content ---
st.header("🎯 Movie Revenue Prediction")

# Sidebar for model selection
model_choice = model_selector(models)
selected_model = models[model_choice]

# Get user input
input_df, input_params = create_user_input(feature_cols)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Features")
    st.markdown("**Movie Details:**")
    st.write(f"🗳️ Vote Average: **{input_params['vote_average']}/10**")
    st.write(f"📊 Vote Count: **{input_params['vote_count']:,}**")
    st.write(f"⏱️ Runtime: **{input_params['runtime']} minutes**")
    st.write(f"📅 Release: **{input_params['release_month']}/{input_params['release_year']}**")
    st.write(f"🎭 Genres: **{', '.join(input_params['genres'])}**")

    with st.expander("🔍 View Full Feature Vector"):
        st.dataframe(input_df.T, use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Results")
    prediction_log = selected_model.predict(input_df)[0]
    prediction_usd = np.expm1(prediction_log)

    st.markdown(f"""
    <div class="prediction-box">
        <h3>💰 Predicted Revenue</h3>
        <h1>${prediction_usd:,.0f}</h1>
        <p>Log Revenue: {prediction_log:.2f}</p>
        <p>Model: {model_choice}</p>
    </div>
    """, unsafe_allow_html=True)

    if prediction_usd < 10_000_000:
        category, color = "🎬 Independent Film", "#FFA726"
    elif prediction_usd < 100_000_000:
        category, color = "🎭 Commercial Success", "#66BB6A"
    else:
        category, color = "🚀 Blockbuster", "#EF5350"

    st.markdown(f"""
    <div style="background: {color}; padding: 1rem; border-radius: 10px; color: white; text-align: center; margin: 1rem 0;">
        <h4>{category}</h4>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("📊 All Model Predictions")
    all_predictions = {}
    for name, model in models.items():
        try:
            pred_log = model.predict(input_df)[0]
            pred_usd = np.expm1(pred_log)
            all_predictions[name] = pred_usd
        except:
            all_predictions[name] = None

    cols = st.columns(2)
    for i, (name, pred) in enumerate(all_predictions.items()):
        if pred is not None:
            with cols[i % 2]:
                st.metric(
                    name, f"${pred:,.0f}",
                    delta=f"{((pred - prediction_usd) / prediction_usd * 100):+.1f}%" if name != model_choice else None
                )