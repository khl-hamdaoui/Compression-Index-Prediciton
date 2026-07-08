import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ================================================================================
# PAGE CONFIGURATION & STYLING
# ================================================================================
st.set_page_config(
    layout="wide",
    page_title="ML-Based Compression Index Prediction",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    :root {
        --primary: #1e3a5f;
        --accent: #0066cc;
        --success: #059669;
    }
    .stApp { background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 100%); }
    h1 { color: var(--primary); font-weight: 700; border-bottom: 3px solid var(--accent); padding-bottom: 1rem; }
    h2 { color: var(--primary); font-weight: 600; border-left: 4px solid var(--accent); padding-left: 1rem; }
    .info-box { background: #eff6ff; border-left: 4px solid var(--accent); padding: 1rem; border-radius: 6px; margin: 1rem 0; }
    .result-box { background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border: 2px solid var(--success); padding: 2rem; border-radius: 8px; text-align: center; }
    .citation-box { background: #f5f3ff; border-left: 4px solid #8b5cf6; padding: 1.5rem; border-radius: 6px; margin: 1.5rem 0; font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)

# ================================================================================
# HEADER
# ================================================================================
st.title("🔬 Compression Index Prediction Tool")

st.markdown("""
<div class="info-box">
Predicts the compression index (Cc) of clay soils using XGBoost, LightGBM, and CatBoost based on a database of 1,243 samples.
</div>
""", unsafe_allow_html=True)

# ================================================================================
# RESOURCE LOADING
# ================================================================================
@st.cache_resource
def load_resources():
    try:
        models = {
            "XGBoost (Recommended)": joblib.load("xg_reg_model.pkl"),
            "LightGBM": joblib.load("lgbm_reg_model.pkl"),
            "CatBoost": joblib.load("cat_reg_model.pkl")
        }
        
        df = pd.read_excel("Compression index.xlsx")
        df = df.dropna()
        
        feature_cols = ['PL (%)', 'PI (%)', 'e0', 'w (%)']
        
        feature_bounds = {
            'PL (%)': {'min': 0.00, 'max': 56.00, 'mean': 24.03},
            'PI (%)': {'min': 1.00, 'max': 153.70, 'mean': 26.65},
            'e0': {'min': 0.28, 'max': 7.11, 'mean': 1.06},
            'w (%)': {'min': 8.00, 'max': 178.20, 'mean': 38.37}
        }
        
        scaler = StandardScaler()
        scaler.fit(df[feature_cols].values)
        
        return models, scaler, feature_cols, feature_bounds
        
    except Exception as e:
        st.error(f"❌ Error loading resources: {e}")
        return None, None, None, None

models, scaler, feature_names, feature_bounds = load_resources()
if models is None:
    st.stop()

# ================================================================================
# SIDEBAR
# ================================================================================
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("""
**Model Performance (Test Set)**
- **XGBoost:** R²=0.913, RMSE=0.197 ⭐
- **LightGBM:** R²=0.913, RMSE=0.198
- **CatBoost:** R²=0.905, RMSE=0.207
""")

selected_model_name = st.sidebar.radio("Select Model", 
    ["XGBoost (Recommended)", "LightGBM", "CatBoost"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset (n=1,243)")
st.sidebar.markdown(f"""
| Feature | Min | Mean | Max |
|---------|-----|------|-----|
| PL (%) | 0.00 | 24.03 | 56.00 |
| PI (%) | 1.00 | 26.65 | 153.70 |
| e₀ | 0.28 | 1.06 | 7.11 |
| w (%) | 8.00 | 38.37 | 178.20 |
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Feature Importance (SHAP)")
st.sidebar.markdown("""
1. e₀ (Void Ratio): 0.20
2. w (Water Content): 0.10
3. PL (Plastic Limit): 0.02
4. PI (Plasticity Index): 0.01
""")

# ================================================================================
# MAIN PREDICTION
# ================================================================================
st.markdown("## 📝 Prediction")

col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.markdown("### Input Parameters")
    
    with st.form("prediction_form"):
        pl = st.number_input("Plastic Limit (PL %)", 
            min_value=0.00, max_value=56.00, value=24.03, step=0.1, format="%.2f")
        
        pi = st.number_input("Plasticity Index (PI %)", 
            min_value=1.00, max_value=153.70, value=26.65, step=0.1, format="%.2f")
        
        e0 = st.number_input("Initial Void Ratio (e₀)", 
            min_value=0.28, max_value=7.11, value=1.06, step=0.01, format="%.2f")
        
        w = st.number_input("Water Content (w %)", 
            min_value=8.00, max_value=178.20, value=38.37, step=0.1, format="%.2f")
        
        submit_btn = st.form_submit_button("🔬 Predict", use_container_width=True)

with col2:
    if submit_btn:
        input_data = np.array([[pl, pi, e0, w]])
        input_scaled = scaler.transform(input_data)
        model = models[selected_model_name]
        
        try:
            prediction = model.predict(input_scaled)[0]

            # RMSE (test set) per model, used as the ± uncertainty margin
            rmse_by_model = {
                "XGBoost (Recommended)": 0.197,
                "LightGBM": 0.198,
                "CatBoost": 0.207
            }
            model_rmse = rmse_by_model[selected_model_name]
            
            st.markdown(f"""
            <div class="result-box">
                <h3 style="margin: 0; color: var(--primary);">Compression Index (Cc)</h3>
                <p style="font-size: 2.8rem; font-weight: 700; color: var(--success); margin: 1rem 0;">{prediction:.2f} <span style="font-size: 1.5rem; color: #6b7280;">± {model_rmse:.3f}</span></p>
                <p style="color: #6b7280;">Model: {selected_model_name} (± RMSE, test set)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Input Summary Table
            st.markdown("### Input Summary")
            summary_data = {
                'Parameter': ['PL (%)', 'PI (%)', 'e₀', 'w (%)'],
                'Input': [f'{pl:.2f}', f'{pi:.2f}', f'{e0:.2f}', f'{w:.2f}'],
                'Dataset Min': ['0.00', '1.00', '0.28', '8.00'],
                'Dataset Max': ['56.00', '153.70', '7.11', '178.20']
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            # Model Performance Comparison (Test Set)
            st.markdown("### Model Performance (Test Set)")
            perf_data = {
                'Model': ['XGBoost', 'LightGBM', 'CatBoost'],
                'R²': [0.913, 0.913, 0.905],
                'RMSE': [0.197, 0.198, 0.207],
                'MAE': [0.100, 0.107, 0.104],
                'MAPE (%)': [25.47, 28.03, 25.67]
            }
            st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

            # Model Performance Comparison (Training Set)
            st.markdown("### Model Performance (Training Set)")
            train_perf_data = {
                'Model': ['XGBoost', 'LightGBM', 'CatBoost'],
                'R²': [0.915, 0.912, 0.922],
                'RMSE': [0.167, 0.170, 0.160],
                'MAE': [0.074, 0.073, 0.079],
                'MAPE (%)': [26.114, 22.647, 26.270]
            }
            train_perf_df = pd.DataFrame(train_perf_data)
            st.dataframe(train_perf_df, use_container_width=True, hide_index=True)

            # Training Error Metrics Bar Chart (RMSE, MAE, MAPE)
            st.markdown("### Training Error Metrics (RMSE, MAE, MAPE)")
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            x = np.arange(len(train_perf_df['Model']))
            width = 0.35

            # RMSE & MAE on left axis (same units as Cc)
            ax1.bar(x - width/2, train_perf_df['RMSE'], width, label='RMSE', color='#0ea5e9')
            ax1.bar(x + width/2, train_perf_df['MAE'], width, label='MAE', color='#059669')
            ax1.set_xticks(x)
            ax1.set_xticklabels(train_perf_df['Model'])
            ax1.set_ylabel('Error Value', fontsize=10, fontweight='bold')
            ax1.set_title('Training RMSE & MAE by Model', fontsize=11, fontweight='bold')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)

            # MAPE on right subplot (different units - %)
            ax2.bar(train_perf_df['Model'], train_perf_df['MAPE (%)'], color='#f59e0b')
            ax2.set_ylabel('MAPE (%)', fontsize=10, fontweight='bold')
            ax2.set_title('Training MAPE by Model', fontsize=11, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()
            
            # Feature Importance Chart
            st.markdown("### Feature Importance (SHAP Mean Values)")
            importance_data = {
                'Feature': ['e₀ (Void Ratio)', 'w (Water Content)', 'PL (Plastic Limit)', 'PI (Plasticity Index)'],
                'Mean SHAP': [0.20, 0.10, 0.02, 0.01]
            }
            importance_df = pd.DataFrame(importance_data)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#059669', '#0ea5e9', '#f59e0b', '#ef4444']
            ax.barh(importance_df['Feature'], importance_df['Mean SHAP'], color=colors)
            ax.set_xlabel('Mean |SHAP| Value', fontsize=11, fontweight='bold')
            ax.set_title('Feature Importance in Compression Index Prediction', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.markdown("""
        <div style="background: #f3f4f6; padding: 3rem; border-radius: 8px; text-align: center;">
            <p style="font-size: 1rem; color: #6b7280;">👈 Enter soil parameters and click <strong>Predict</strong></p>
        </div>
        """, unsafe_allow_html=True)

# ================================================================================
# RESEARCH INFO
# ================================================================================
st.markdown("---")

with st.expander("📚 **Research Information**", expanded=False):
    st.markdown("""
    ### Dataset
    - **Samples:** 1,243 clay specimens
    - **Sources:** 13 peer-reviewed studies
    - **Split:** 70% training (870) / 30% testing (373)
    - **Validation:** 5-fold cross-validation
    
    ### Input Features
    - Plastic Limit (PL): 0.00 - 56.00%
    - Plasticity Index (PI): 1.00 - 153.70%
    - Initial Void Ratio (e₀): 0.28 - 7.11
    - Water Content (w): 8.00 - 178.20%
    
    ### Output
    - Compression Index (Cc): 0.01 - 4.22
    
    ### Key Correlations
    - e₀ vs Cc: r = 0.89
    - w vs e₀: r = 0.97
    - w vs Cc: r = 0.89
    
    ### Methodology
    - **Hyperparameter Optimization:** Optuna (1,000 iterations)
    - **Feature Scaling:** StandardScaler
    - **Models:** XGBoost, LightGBM, CatBoost
    """)

with st.expander("⚠️ **Limitations**", expanded=False):
    st.markdown("""
    **Applicability:**
    - Fine-grained cohesive soils (CH, OH, CL, OL, ML)
    - Limited to input variable ranges (see dataset bounds)
    - Site-specific validation recommended
    
    **Best Used For:**
    - Preliminary geotechnical assessment
    - Rapid multi-scenario screening
    - Guiding laboratory testing strategy
    
    **Not Suitable For:**
    - Final design (lab testing required)
    - Extrapolation beyond dataset ranges
    """)

# ================================================================================
# REFERENCE
# ================================================================================
st.markdown("---")
st.markdown("""
<div class="citation-box">
<strong>📄 Reference:</strong><br>
Hamdaoui, K., Benzaamia, A., Sari Ahmed, B., Guellil, M.E. and Ghrici, M., 2025. Interpretable machine learning for predicting compression index of clays using SHAP and gradient boosting models.
Journal of Engineering and Applied Science, 72(1), p.148. 
<a href="https://doi.org/10.1186/s44147-025-00727-4" target="_blank">https://doi.org/10.1186/s44147-025-00727-4</a>
</div>
""", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; padding: 1rem; color: #6b7280; font-size: 0.85rem;">
🏛️ Hassiba Benbouali University | 🔬 Geomaterials Laboratory, Chlef, Algeria  
📧 m_ghrici@yahoo.fr | Updated: {datetime.now().strftime('%Y-%m-%d')}
</div>
""", unsafe_allow_html=True)
