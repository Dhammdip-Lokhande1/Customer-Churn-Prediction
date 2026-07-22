import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt

# Add project root to python path to ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocess import preprocess, align_features
from src.predict import load_artifacts, predict_churn
from src.explain import get_shap_explainer, local_explanation_plot, get_top_factors

# Configure Streamlit Page
st.set_page_config(
    page_title="Customer Detail - ChurnSight",
    page_icon="🔍",
    layout="wide"
)

# Custom styling for glassmorphic cards and badges
st.markdown("""
<style>
    .stApp {
        background-color: #0F1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    .glass-card {
        background: rgba(30, 33, 48, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    
    .metric-title {
        font-size: 0.95rem;
        font-weight: 500;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FAFAFA;
    }
    
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-high { background: rgba(255, 75, 75, 0.15); color: #FF4B4B; border: 1px solid rgba(255, 75, 75, 0.3); }
    .badge-medium { background: rgba(255, 166, 43, 0.15); color: #FFA62B; border: 1px solid rgba(255, 166, 43, 0.3); }
    .badge-low { background: rgba(45, 212, 191, 0.15); color: #2DD4BF; border: 1px solid rgba(45, 212, 191, 0.3); }
    
    .factor-item {
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 0.9rem;
    }
    .factor-positive {
        background: rgba(255, 75, 75, 0.08);
        border-left: 4px solid #FF4B4B;
    }
    .factor-negative {
        background: rgba(45, 212, 191, 0.08);
        border-left: 4px solid #2DD4BF;
    }
</style>
""", unsafe_allow_html=True)

# Path resolutions
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'telco_churn.csv')
DATA_PROCESSED_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_data.pkl')

@st.cache_data
def get_raw_data():
    return pd.read_csv(DATA_RAW_PATH)

@st.cache_data
def get_processed_data():
    return pd.read_pickle(DATA_PROCESSED_PATH)

try:
    df_raw = get_raw_data()
    df_processed = get_processed_data()
    model, scaler, features = load_artifacts()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.warning("Please ensure the training script `src/train.py` has run successfully.")
    st.stop()

# ─── Sidebar Lookup ─────────────────────────────────────────
st.sidebar.markdown("<h3 style='color: #6C63FF;'>🔍 Search Customer</h3>", unsafe_allow_html=True)

# Select box with some default customer IDs
sample_ids = df_raw['customerID'].head(50).tolist()
selected_id = st.sidebar.selectbox(
    "Select Customer ID",
    sample_ids,
    help="Select a customer ID from the list, or type it in below."
)

custom_id = st.sidebar.text_input("Or Enter Custom ID", "").strip()
search_id = custom_id if custom_id else selected_id

# Verify customer exists
if search_id not in df_raw['customerID'].values:
    st.error(f"Customer ID '{search_id}' not found. Please verify the ID.")
    st.stop()

# Fetch customer details
cust_raw = df_raw[df_raw['customerID'] == search_id].iloc[0]
cust_processed = df_processed.loc[[search_id]]

# Run model prediction
prob = predict_churn(cust_processed, model, scaler, features).iloc[0]
risk_score = prob * 100

if prob >= 0.5:
    risk_label = "High Churn Risk"
    risk_badge_class = "badge-high"
elif prob >= 0.25:
    risk_label = "Medium Churn Risk"
    risk_badge_class = "badge-medium"
else:
    risk_label = "Low Churn Risk"
    risk_badge_class = "badge-low"

# ─── Page Title ────────────────────────────────────────────
st.markdown(f"## 🔍 Customer Analysis: `{search_id}`")
st.caption("SHAP-based explanation of factors driving this customer's churn prediction")
st.divider()

# ─── Top Cards (Risk and Quick Info) ───────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='glass-card'>
        <div class='metric-title'>Churn Probability</div>
        <div class='metric-value'>{risk_score:.1f}%</div>
        <div style='margin-top: 8px;'>
            <span class='badge {risk_badge_class}'>{risk_label}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='glass-card'>
        <div class='metric-title'>Contract Type</div>
        <div class='metric-value' style='font-size: 1.6rem;'>{cust_raw['Contract']}</div>
        <div class='metric-title' style='margin-top: 10px;'>Tenure: {cust_raw['tenure']} Months</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='glass-card'>
        <div class='metric-title'>Monthly Charges</div>
        <div class='metric-value'>${cust_raw['MonthlyCharges']:.2f}</div>
        <div class='metric-title' style='margin-top: 10px;'>Payment: {cust_raw['PaymentMethod']}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='glass-card'>
        <div class='metric-title'>Total Charges</div>
        <div class='metric-value'>${cust_raw['TotalCharges']}</div>
        <div class='metric-title' style='margin-top: 10px;'>Tech Support: {cust_raw['TechSupport']}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Split Screen: Demographic Info & SHAP Explanations ────
left, right = st.columns([1, 1])

# Scale the preprocessed customer row for SHAP analysis
X_all = df_processed.drop(columns=['Churn'], errors='ignore')
X_all_aligned = align_features(X_all, features)
X_all_scaled = pd.DataFrame(scaler.transform(X_all_aligned), index=X_all_aligned.index, columns=X_all_aligned.columns)

# Get specific customer's scaled row
cust_scaled = X_all_scaled.loc[[search_id]]

# Instantiate TreeExplainer
# TreeExplainer is fast, but we cache it for performance
@st.cache_resource
def load_explainer(_model, _data):
    return get_shap_explainer(_model, _data)

explainer = load_explainer(model, X_all_scaled)

with left:
    st.subheader("👤 Demographic & Services Profile")
    
    # We display a structured table of demographics and services
    profile_data = {
        "Attribute": [
            "Gender", "Senior Citizen", "Partner", "Dependents",
            "Phone Service", "Multiple Lines", "Internet Service",
            "Online Security", "Online Backup", "Device Protection",
            "Tech Support", "Streaming TV", "Streaming Movies", "Paperless Billing"
        ],
        "Value": [
            cust_raw['gender'],
            "Yes" if cust_raw['SeniorCitizen'] == 1 else "No",
            cust_raw['Partner'], cust_raw['Dependents'],
            cust_raw['PhoneService'], cust_raw['MultipleLines'],
            cust_raw['InternetService'], cust_raw['OnlineSecurity'],
            cust_raw['OnlineBackup'], cust_raw['DeviceProtection'],
            cust_raw['TechSupport'], cust_raw['StreamingTV'],
            cust_raw['StreamingMovies'], cust_raw['PaperlessBilling']
        ]
    }
    st.table(pd.DataFrame(profile_data))

with right:
    st.subheader("💡 Churn Drivers (SHAP Attribution)")
    st.caption("How each feature shifts the churn probability from the base average.")
    
    # Get top 5 factors
    top_factors = get_top_factors(explainer, cust_scaled, top_n=8)
    
    # Render factors in pretty badges
    st.markdown("##### Key Factors Increasing Churn Risk:")
    pos_factors = [f for f in top_factors if f[1] > 0]
    if pos_factors:
        for name, val in pos_factors:
            st.markdown(f"""
            <div class='factor-item factor-positive'>
                🔴 <b>{name}</b> pushes risk <b>up</b> by <b>+{val*100:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #9CA3AF;'>No significant positive churn drivers found.</p>", unsafe_allow_html=True)
        
    st.markdown("##### Key Factors Reducing Churn Risk:")
    neg_factors = [f for f in top_factors if f[1] < 0]
    if neg_factors:
        for name, val in neg_factors:
            st.markdown(f"""
            <div class='factor-item factor-negative'>
                🟢 <b>{name}</b> pulls risk <b>down</b> by <b>{val*100:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #9CA3AF;'>No significant risk-reducing factors found.</p>", unsafe_allow_html=True)

st.divider()

# ─── Waterfall Plot ────────────────────────────────────────
st.subheader("📈 SHAP Waterfall Explanation Plot")
st.caption("Visualizing the step-by-step impact of features on this prediction. Positive SHAP values (red) shift the prediction toward Churn. Negative values (blue) shift it toward Retention.")

with st.spinner("Generating SHAP Waterfall Plot..."):
    try:
        # Use matplotlib waterfall plot
        fig_waterfall = local_explanation_plot(explainer, cust_scaled)
        st.pyplot(fig_waterfall)
    except Exception as e:
        st.error(f"Error plotting SHAP waterfall: {e}")
