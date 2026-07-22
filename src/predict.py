import os
import sys

# Add project root to python path to ensure cross-module imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from src.preprocess import align_features

# Resolve project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_artifacts():
    """
    Load the trained XGBoost model, the fitted scaler, and training feature columns.
    """
    model_path = os.path.join(MODELS_DIR, 'xgb_churn_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    feature_path = os.path.join(MODELS_DIR, 'feature_columns.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(feature_path):
        raise FileNotFoundError(
            "Model artifacts not found. Please train the model by running src/train.py first."
        )
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(feature_path)
    
    return model, scaler, features

def predict_churn(df_preprocessed, model, scaler, features):
    """
    Predict churn probabilities for preprocessed input data.
    Ensures columns match training set, scales them, and runs model prediction.
    """
    # Align features to ensure column structure matches training data
    df_aligned = align_features(df_preprocessed, features)
    
    # Scale features
    scaled_data = scaler.transform(df_aligned)
    df_scaled = pd.DataFrame(scaled_data, index=df_aligned.index, columns=df_aligned.columns)
    
    # Predict churn probability
    probabilities = model.predict_proba(df_scaled)[:, 1]
    
    return pd.Series(probabilities, index=df_preprocessed.index, name='Churn_Probability')
