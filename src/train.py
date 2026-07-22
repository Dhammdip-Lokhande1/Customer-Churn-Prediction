import os
import sys

# Add project root to python path to ensure cross-module imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
from src.preprocess import preprocess, split_and_scale

# Resolve project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

def train_model(X_train, y_train):
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.7,   # Handles class imbalance (73/27 ratio)
        eval_metric='logloss',
        random_state=42
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"5-Fold CV ROC-AUC: {np.mean(cv_scores):.4f} \u00b1 {np.std(cv_scores):.4f}")

    model.fit(X_train, y_train)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'xgb_churn_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Test Set Evaluation ---")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate recall and precision for class 1 (churn)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    print(f"Recall (Churn Class): {recall:.4f}")
    print(f"Precision (Churn Class): {precision:.4f}")
    
    return y_prob

if __name__ == "__main__":
    raw_data_path = os.path.join(DATA_DIR, 'raw', 'telco_churn.csv')
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}. Please run src/download_data.py first.")

    print(f"Loading data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)
    
    print("Preprocessing data...")
    df_processed = preprocess(df)
    
    # Save the processed data to processed/cleaned_data.pkl
    processed_dir = os.path.join(DATA_DIR, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    df_processed.to_pickle(os.path.join(processed_dir, 'cleaned_data.pkl'))
    print(f"Cleaned data saved to {processed_dir}")

    print("Splitting and scaling...")
    X_train, X_test, y_train, y_test, features = split_and_scale(df_processed)
    
    # Train
    model = train_model(X_train, y_train)
    
    # Evaluate
    evaluate(model, X_test, y_test)
