import streamlit as st

def render_sidebar_filters(df):
    """
    Renders sidebar title, logo, risk threshold slider, risk level filter, 
    and contract type filter. Returns the filter values.
    """
    st.markdown("<h2 style='text-align: center; color: #6C63FF; margin-bottom: 0;'>📡 ChurnSight</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9CA3AF; font-size: 0.85rem; margin-top: 0;'>AI-Powered Churn Intelligence</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Dashboard Filters")
    
    threshold = st.slider(
        "Churn Risk Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Probability threshold above which a customer is classified as 'High Churn Risk'."
    )
    
    risk_filter = st.multiselect(
        "Filter by Risk Level",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )
    
    contract_filter = st.multiselect(
        "Filter by Contract Type",
        df['Contract'].unique().tolist(),
        default=df['Contract'].unique().tolist()
    )
    
    st.divider()
    
    st.caption("🤖 Model: **XGBoost Classifier v1.0**")
    st.caption("📊 Dataset: **IBM Telco Churn (~7k rows)**")
    
    return threshold, risk_filter, contract_filter
