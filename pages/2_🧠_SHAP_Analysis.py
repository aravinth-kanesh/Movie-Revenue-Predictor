import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
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
st.header("🧠 SHAP Feature Analysis")

# Sidebar: select model
model_choice = model_selector(models)
if model_choice is None:
    st.stop()

selected_model = models[model_choice]

# Get user input
input_df, _ = create_user_input(feature_cols)

# Prediction
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
    # Compute SHAP values
    with st.spinner("🔄 Computing SHAP explanations..."):
        try:
            explainer = shap.TreeExplainer(selected_model)
            shap_values = explainer(input_df)

            # Feature Contribution (Bar Plot)
            st.markdown("### 📊 Feature Contribution (Top 15)")

            # Create manual bar plot (more reliable than shap.summary_plot in Streamlit)
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'SHAP_Value': shap_values.values[0],
                'Abs_SHAP': np.abs(shap_values.values[0])
            }).sort_values('Abs_SHAP', ascending=False).head(15)

            # Create horizontal bar chart
            fig1, ax1 = plt.subplots(figsize=(10, 8))

            # Color bars based on positive/negative impact
            colors = ['green' if x > 0 else 'red' for x in feature_importance['SHAP_Value']]

            bars = ax1.barh(range(len(feature_importance)), feature_importance['SHAP_Value'], color=colors)
            ax1.set_yticks(range(len(feature_importance)))
            ax1.set_yticklabels(feature_importance['Feature'])
            ax1.set_xlabel('SHAP Value (Impact on Prediction)')
            ax1.set_title('Feature Contributions to Prediction')
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax1.text(width, bar.get_y() + bar.get_height() / 2,
                         f'{width:.3f}', ha='left' if width > 0 else 'right',
                         va='center', fontsize=8)

            plt.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)

            # Waterfall Analysis
            st.markdown("### 💧 Waterfall Analysis")
            st.info("Shows how each feature pushes the prediction above or below the expected value")

            try:
                # Try modern SHAP waterfall first
                fig2, ax2 = plt.subplots(figsize=(12, 8))
                shap.plots.waterfall(shap_values[0], show=False, max_display=15)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

            except Exception as e:
                st.warning(f"Modern waterfall plot failed, using alternative visualization: {e}")

                # Alternative: Cumulative impact chart
                fig3, ax3 = plt.subplots(figsize=(12, 6))

                # Get top features for waterfall
                top_features = feature_importance.head(10)
                baseline = shap_values.base_values[0]

                # Create cumulative values
                cumulative = [baseline]
                labels = ['Baseline']

                for _, row in top_features.iterrows():
                    cumulative.append(cumulative[-1] + row['SHAP_Value'])
                    labels.append(row['Feature'][:20] + '...' if len(row['Feature']) > 20 else row['Feature'])

                # Plot
                x_pos = range(len(cumulative))
                ax3.plot(x_pos, cumulative, marker='o', linewidth=2, markersize=6)

                # Add bars showing individual contributions
                for i in range(1, len(cumulative)):
                    contribution = cumulative[i] - cumulative[i - 1]
                    color = 'green' if contribution > 0 else 'red'
                    ax3.bar(i, abs(contribution), bottom=min(cumulative[i - 1], cumulative[i]),
                            alpha=0.3, color=color, width=0.6)

                ax3.set_xticks(x_pos)
                ax3.set_xticklabels(labels, rotation=45, ha='right')
                ax3.set_ylabel('Log Revenue Prediction')
                ax3.set_title('Cumulative Feature Impact (Waterfall Alternative)')
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)

            # Feature Values Table
            st.markdown("### 📋 Detailed Feature Analysis")

            detailed_table = pd.DataFrame({
                'Feature': feature_importance['Feature'],
                'Feature_Value': [input_df.iloc[0][feat] for feat in feature_importance['Feature']],
                'SHAP_Value': feature_importance['SHAP_Value'],
                'Impact': ['Increases' if x > 0 else 'Decreases' for x in feature_importance['SHAP_Value']]
            })

            st.dataframe(detailed_table, use_container_width=True)

            # Summary info
            st.markdown("### 📈 SHAP Summary")
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.metric("Baseline Prediction", f"{shap_values.base_values[0]:.3f}")

            with col_b:
                total_shap = shap_values.values[0].sum()
                st.metric("Total SHAP Impact", f"{total_shap:.3f}")

            with col_c:
                final_pred = shap_values.base_values[0] + total_shap
                st.metric("Final Prediction", f"{final_pred:.3f}")

        except Exception as e:
            st.error(f"SHAP analysis failed: {e}")
            st.write("**Possible solutions:**")
            st.write("1. Ensure your model is tree-based (Random Forest, LightGBM)")
            st.write("2. Check SHAP installation: `pip install shap`")
            st.write("3. Try a different model type")