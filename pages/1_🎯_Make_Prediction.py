import streamlit as st
import numpy as np
import warnings
from utils import load_models, create_user_input, model_selector, load_css

warnings.filterwarnings('ignore')

# Load custom CSS
load_css()

# Load models and feature columns
models, feature_cols = load_models()
if models is None or feature_cols is None:
    st.error("Models not loaded. Please check the file paths.")
    st.stop()

# --- Page Content ---
st.header("🎯 Movie Revenue Prediction")

# Sidebar: select model
model_choice = model_selector(models)

# Add check for model_choice (in case no models are available)
if model_choice is None:
    st.error("No models available for selection.")
    st.stop()

selected_model = models[model_choice]

# Get user input
try:
    input_df, input_params = create_user_input(feature_cols)
except Exception as e:
    st.error(f"Error creating user input: {e}")
    st.stop()

# Layout columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input Features")
    st.markdown("**Movie Details:**")
    st.write(f"🗳️ Vote Average: **{input_params['vote_average']}/10**")
    st.write(f"📊 Vote Count: **{input_params['vote_count']:,}**")
    st.write(f"⏱️ Runtime: **{input_params['runtime']} minutes**")
    st.write(f"📅 Release: **{input_params['release_month']}/{input_params['release_year']}**")
    st.write(f"🎭 Genres: **{', '.join(input_params['genres'])}**")
    st.write(f"💰 Budget: **${input_params['budget_millions']:.1f}M**")
    st.write(f"⭐ Popularity: **{input_params['popularity']:.1f}**")

    with st.expander("🔍 View Full Feature Vector"):
        st.dataframe(input_df.T, use_container_width=True)

with col2:
    st.subheader("🎯 Prediction Results")

    # Make prediction with error handling
    try:
        prediction_log = selected_model.predict(input_df)[0]
        prediction_usd = np.expm1(prediction_log)

        # Sanity check for prediction
        if prediction_usd < 0 or np.isnan(prediction_usd) or np.isinf(prediction_usd):
            st.error("Invalid prediction result. Please check your input features.")
            st.stop()

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("**Debug Info:**")
        st.write(f"Input shape: {input_df.shape}")
        st.write(f"Model type: {type(selected_model)}")
        st.stop()

    st.markdown(f"""
    <div class="prediction-box">
        <h3>💰 Predicted Revenue</h3>
        <h1>${prediction_usd:,.0f}</h1>
        <p>Log Revenue: {prediction_log:.2f}</p>
        <p>Model: {model_choice}</p>
    </div>
    """, unsafe_allow_html=True)

    # Categorise movie based on predicted revenue
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

    # Display predictions for all models
    st.subheader("📊 All Model Predictions")

    try:
        all_predictions = {}
        for name, model in models.items():
            try:
                pred_log = model.predict(input_df)[0]
                pred_usd = np.expm1(pred_log)
                # Sanity check
                if pred_usd >= 0 and not np.isnan(pred_usd) and not np.isinf(pred_usd):
                    all_predictions[name] = pred_usd
                else:
                    st.warning(f"Invalid prediction from {name} model")
            except Exception as e:
                st.warning(f"Prediction failed for {name}: {e}")

        if all_predictions:
            cols = st.columns(2)
            for i, (name, pred_usd) in enumerate(all_predictions.items()):
                with cols[i % 2]:
                    # Calculate delta only if current model prediction is valid
                    delta = None
                    if name != model_choice and prediction_usd > 0:
                        delta = f"{((pred_usd - prediction_usd) / prediction_usd * 100):+.1f}%"

                    st.metric(
                        name,
                        f"${pred_usd:,.0f}",
                        delta=delta
                    )
        else:
            st.error("No valid predictions from any model")

    except Exception as e:
        st.error(f"Error generating all model predictions: {e}")