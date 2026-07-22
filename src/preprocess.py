import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Resolve project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Set customerID as index if it exists in columns
    if 'customerID' in df.columns:
        df.set_index('customerID', inplace=True)

    # Fix TotalCharges (has spaces for new customers)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    # Encode binary columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            # Map Yes/No, Male/Female, etc.
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
            # If mapping produced NaNs (e.g. if already numeric), fill or leave
            df[col] = df[col].fillna(0).astype(int)

    # One-hot encode multi-class categoricals
    cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    
    # We only encode columns that exist in the dataframe
    cols_to_encode = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    return df

def split_and_scale(df):
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    X = df.drop(columns=['Churn'], errors='ignore')
    y = df['Churn'] if 'Churn' in df.columns else None

    # Stratified split to handle class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled_arr = scaler.fit_transform(X_train)
    X_test_scaled_arr = scaler.transform(X_test)
    
    # Reconstruct DataFrames to keep column names and index (customerID)
    X_train_scaled = pd.DataFrame(X_train_scaled_arr, index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled_arr, index=X_test.index, columns=X_test.columns)

    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(MODELS_DIR, 'feature_columns.pkl'))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()

# Helper for processing single inference row or alignment with training features
def align_features(df_preprocessed, feature_cols):
    """
    Ensure the preprocessed DataFrame has exactly the same columns as the training features,
    filling missing columns with 0 (since they represent missing dummy categories) and
    reordering columns to match.
    """
    for col in feature_cols:
        if col not in df_preprocessed.columns:
            df_preprocessed[col] = 0
    return df_preprocessed[feature_cols]
