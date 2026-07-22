import os
import shap
import matplotlib.pyplot as plt
import pandas as pd

def get_shap_explainer(model, X_train=None):
    """
    Create a SHAP TreeExplainer for the XGBoost model.
    """
    # TreeExplainer works directly on Tree models without background data, 
    # but providing X_train helps check shapes and baseline values.
    explainer = shap.TreeExplainer(model, data=X_train)
    return explainer

def global_feature_importance_plot(explainer, X_df):
    """
    Generates a beeswarm plot showing the global feature importance, 
    direction, and magnitude.
    """
    shap_values = explainer(X_df)
    
    # Create a matplotlib figure
    fig = plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP Global Feature Importance (Beeswarm)", fontsize=14, pad=15)
    plt.tight_layout()
    return fig

def local_explanation_plot(explainer, customer_row):
    """
    Generates a waterfall plot explaining the prediction for a single customer.
    """
    if isinstance(customer_row, pd.Series):
        customer_row = pd.DataFrame([customer_row])
        
    shap_values = explainer(customer_row)
    
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.title(f"SHAP Waterfall Prediction Explanation", fontsize=14, pad=15)
    plt.tight_layout()
    return fig

def get_top_factors(explainer, customer_row, top_n=5):
    """
    Get the top N features contributing positively or negatively to the prediction.
    Returns list of tuples: (feature_name, impact_value)
    """
    if isinstance(customer_row, pd.Series):
        customer_row = pd.DataFrame([customer_row])
        
    shap_values = explainer(customer_row)
    
    # Extract values, base value, and feature names
    impacts = shap_values.values[0]
    feature_names = shap_values.feature_names
    
    # Pair feature names with their SHAP impact values
    factors = list(zip(feature_names, impacts))
    
    # Sort by absolute SHAP impact
    sorted_factors = sorted(factors, key=lambda x: abs(x[1]), reverse=True)
    
    return sorted_factors[:top_n]
