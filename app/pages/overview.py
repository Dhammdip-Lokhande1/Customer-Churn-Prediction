import streamlit as st

st.set_page_config(
    page_title="Overview - ChurnSight",
    page_icon="📡",
    layout="wide"
)

# Custom CSS for dark glassmorphic layout
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
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }
    .title-accent {
        color: #6C63FF;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📡 Project Overview & Guide")
st.caption("Welcome to ChurnSight — Churn Intelligence Dashboard")
st.divider()

st.markdown("""
<div class='glass-card'>
    <h3>🔍 What is <span class='title-accent'>ChurnSight</span>?</h3>
    <p>
        ChurnSight is an AI-powered churn intelligence dashboard built to help subscription-based businesses (like Telecom providers) 
        identify customers at risk of leaving (churning) and understand the specific reasons driving each customer's risk profile.
    </p>
    <p>
        Instead of treating machine learning predictions as a "black box", ChurnSight uses <b>SHAP (SHapley Additive exPlanations)</b> 
        to break down predictions into individual feature attributions, explaining the exact factors pushing a customer's score up or down.
    </p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown("""
    <div class='glass-card'>
        <h3>🚀 Features</h3>
        <ul>
            <li><b>Active Risk Database</b>: View and search through the customer portfolio filtered by risk levels (High, Medium, Low).</li>
            <li><b>Explainable AI (XAI)</b>: View individual customer reports showing demographic details and waterfall SHAP attribution charts.</li>
            <li><b>Global Model Diagnostics</b>: Evaluate the underlying XGBoost model's performance via ROC-AUC curves, confusion matrices, and global SHAP beeswarm feature importance.</li>
            <li><b>Interactive Risk Thresholds</b>: Adjust the risk classification threshold in the sidebar to simulate operational trade-offs between precision and recall.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("""
    <div class='glass-card'>
        <h3>📖 How to Use the Dashboard</h3>
        <ol>
            <li><b>Review the Dashboard</b>: Go to the main entry page (<b>main.py</b>) to see high-level KPI metrics, active churn risk distribution, and a searchable database table.</li>
            <li><b>Analyze a Customer</b>: Copy a <i>Customer ID</i> (e.g. <code>7590-VHVEG</code>) from the risk table.</li>
            <li><b>Run SHAP Explanations</b>: Navigate to the <b>Customer Detail</b> page, paste the ID (or select from the sample list), and explore the waterfall plot of their churn drivers.</li>
            <li><b>Inspect Model Performance</b>: Navigate to the <b>Model Metrics</b> page to review performance statistics and the global feature importance beeswarm chart.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
