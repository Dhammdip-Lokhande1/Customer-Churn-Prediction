import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to python path to ensure src modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess
from src.predict import load_artifacts, predict_churn
from app.components.sidebar import render_sidebar_filters
from app.components.charts import plot_risk_pie, plot_risk_vs_tenure

# Configure Streamlit Page
st.set_page_config(
    page_title="ChurnSight",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom Premium CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #0F1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1rem;
        color: #9CA3AF;
        margin-top: 0;
        margin-bottom: 20px;
    }
    .glass-card {
        background: rgba(30, 33, 48, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 0.9rem;
        font-weight: 500;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FAFAFA;
        line-height: 1.2;
    }
    .metric-delta {
        font-size: 0.85rem;
        margin-top: 6px;
        font-weight: 600;
    }
    .delta-up { color: #FF4B4B; }
    .delta-down { color: #2DD4BF; }
</style>
""", unsafe_allow_html=True)

# Cache data loading & predictions
@st.cache_data
def load_and_predict_data():
    raw_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'telco_churn.csv')
    df_raw = pd.read_csv(raw_path)
    df_preprocessed = preprocess(df_raw)
    
    model, scaler, features = load_artifacts()
    probs = predict_churn(df_preprocessed, model, scaler, features)
    
    df_results = df_raw.copy()
    df_results['Churn_Probability'] = df_results['customerID'].map(probs)
    df_results['Risk_Score'] = (df_results['Churn_Probability'] * 100).round(1)
    
    return df_results

try:
    df_predictions = load_and_predict_data()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.warning("Please ensure the training script `src/train.py` has run successfully.")
    st.stop()

# ─── Sidebar Filters Component ─────────────────────────────
with st.sidebar:
    threshold, risk_filter, contract_filter = render_sidebar_filters(df_predictions)

# Categorize risk levels based on dynamic threshold slider
def categorize_risk(prob, t):
    if prob >= t:
        return "High"
    elif prob >= max(0.0, t - 0.25):
        return "Medium"
    else:
        return "Low"
        
df_predictions['Risk_Level'] = df_predictions['Churn_Probability'].apply(lambda x: categorize_risk(x, threshold))

# Apply filters
filtered_df = df_predictions[
    (df_predictions['Risk_Level'].isin(risk_filter)) &
    (df_predictions['Contract'].isin(contract_filter))
]

# ─── Header ────────────────────────────────────────────────
st.markdown("<div class='main-header'>📡 ChurnSight Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Real-time customer churn prediction & explainable risk analytics</div>", unsafe_allow_html=True)

# ─── KPI Row ───────────────────────────────────────────────
kpi_cols = st.columns(4)

total_cust = len(df_predictions)
high_risk_cust = len(df_predictions[df_predictions['Risk_Level'] == 'High'])
avg_prob = df_predictions['Churn_Probability'].mean() * 100
high_risk_ratio = (high_risk_cust / total_cust) * 100

with kpi_cols[0]:
    st.markdown(f"""
    <div class='glass-card'>
        <div class='metric-title'>Total Customers</div>
        <div class='metric-value'>{total_cust:,}</div>
        <div class='metric-delta delta-down'>Active Monitoring</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class='glass-card'>
        <div class='metric-title'>High Risk Customers</div>
        <div class='metric-value'>{high_risk_cust:,}</div>
        <div class='metric-delta delta-up'>{high_risk_ratio:.1f}% of base</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
    <div class='glass-card'>
        <div class='metric-title'>Avg Churn Probability</div>
        <div class='metric-value'>{avg_prob:.2f}%</div>
        <div class='metric-delta delta-up'>Dataset Rate: 26.5%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown("""
    <div class='glass-card'>
        <div class='metric-title'>Model Performance</div>
        <div class='metric-value'>0.847</div>
        <div class='metric-delta delta-down'>ROC-AUC Score</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Main Content ──────────────────────────────────────────
left, right = st.columns([5, 3])

with left:
    st.subheader("📋 Customer Risk Database")
    st.caption("Search and filter customers by details. Copy ID to view their detailed SHAP explanation.")
    
    # Search input
    search_query = st.text_input("🔍 Search by Customer ID", "").strip()
    if search_query:
        display_df = filtered_df[filtered_df['customerID'].str.contains(search_query, case=False)]
    else:
        display_df = filtered_df
        
    # Format and present database
    cols_to_show = ['customerID', 'gender', 'tenure', 'Contract', 'PaymentMethod', 'MonthlyCharges', 'Risk_Score', 'Risk_Level']
    table_df = display_df[cols_to_show].copy()
    table_df.columns = ['Customer ID', 'Gender', 'Tenure (months)', 'Contract', 'Payment Method', 'Monthly Charges ($)', 'Risk Score (%)', 'Risk Level']
    
    st.dataframe(
        table_df,
        use_container_width=True,
        height=450,
        hide_index=True
    )
    
    st.info("💡 **Pro-Tip**: Copy a 'Customer ID' from this table and paste it into the **Customer Detail** page in the sidebar to run a full SHAP local attribution explanation for them.")

with right:
    st.subheader("📊 Risk Profile Analysis")
    st.caption("Overview of risk counts and contract segment breakdown.")
    
    # Risk Distribution Pie Chart
    fig_pie = plot_risk_pie(df_predictions)
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Risk vs Tenure Box Plot
    st.markdown("##### Churn Risk vs. Customer Tenure")
    fig_box = plot_risk_vs_tenure(df_predictions)
    st.plotly_chart(fig_box, use_container_width=True)
