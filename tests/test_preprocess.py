import os
import sys
import pandas as pd
import numpy as np
import pytest

# Add project root to python path to import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess, split_and_scale, align_features

@pytest.fixture
def sample_data():
    """Generates synthetic Telco Churn DataFrame for testing preprocessing."""
    data = {
        'customerID': ['0001-AAAA', '0002-BBBB', '0003-CCCC', '0004-DDDD', '0005-EEEE'],
        'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'Partner': ['Yes', 'No', 'No', 'Yes', 'No'],
        'Dependents': ['No', 'Yes', 'No', 'No', 'Yes'],
        'tenure': [12, 24, 1, 36, 4],
        'PhoneService': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'MultipleLines': ['No', 'No phone service', 'Yes', 'Yes', 'No phone service'],
        'InternetService': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic'],
        'OnlineSecurity': ['Yes', 'No', 'No internet service', 'No', 'Yes'],
        'OnlineBackup': ['No', 'Yes', 'No internet service', 'Yes', 'No'],
        'DeviceProtection': ['No', 'No', 'No internet service', 'Yes', 'No'],
        'TechSupport': ['Yes', 'No', 'No internet service', 'No', 'Yes'],
        'StreamingTV': ['No', 'Yes', 'No internet service', 'Yes', 'No'],
        'StreamingMovies': ['No', 'Yes', 'No internet service', 'No', 'Yes'],
        'Contract': ['One year', 'Month-to-month', 'Two year', 'Month-to-month', 'Month-to-month'],
        'PaperlessBilling': ['Yes', 'No', 'Yes', 'Yes', 'No'],
        'PaymentMethod': ['Mailed check', 'Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)', 'Electronic check'],
        'MonthlyCharges': [50.5, 80.25, 20.0, 95.0, 75.8],
        'TotalCharges': ['606', '1926', '20', '3420', '303.2'],
        'Churn': ['No', 'Yes', 'No', 'Yes', 'No']
    }
    return pd.DataFrame(data)

def test_preprocess_drops_customer_id_and_sets_index(sample_data):
    df_clean = preprocess(sample_data)
    assert 'customerID' not in df_clean.columns
    assert df_clean.index.name == 'customerID'
    assert len(df_clean) == 5

def test_preprocess_maps_binary_cols(sample_data):
    df_clean = preprocess(sample_data)
    # Check that gender is mapped (Male -> 1, Female -> 0)
    assert df_clean.loc['0001-AAAA', 'gender'] == 1
    assert df_clean.loc['0002-BBBB', 'gender'] == 0
    # Check Partner (Yes -> 1, No -> 0)
    assert df_clean.loc['0001-AAAA', 'Partner'] == 1
    assert df_clean.loc['0002-BBBB', 'Partner'] == 0
    # Check Churn mapping (Yes -> 1, No -> 0)
    assert df_clean.loc['0001-AAAA', 'Churn'] == 0
    assert df_clean.loc['0002-BBBB', 'Churn'] == 1

def test_preprocess_total_charges_numeric(sample_data):
    # Add a row with space in TotalCharges to test handling of missing values
    bad_row = sample_data.iloc[0].copy()
    bad_row['customerID'] = '0006-FFFF'
    bad_row['TotalCharges'] = ' '  # space value
    sample_data_with_space = pd.concat([sample_data, pd.DataFrame([bad_row])], ignore_index=True)
    
    df_clean = preprocess(sample_data_with_space)
    assert df_clean.loc['0006-FFFF', 'TotalCharges'] == 0.0
    assert pd.api.types.is_numeric_dtype(df_clean['TotalCharges'])

def test_align_features():
    train_features = ['tenure', 'MonthlyCharges', 'Contract_One year', 'Contract_Two year']
    test_df = pd.DataFrame({
        'tenure': [10],
        'MonthlyCharges': [45.0],
        'Contract_One year': [1]
    })
    
    aligned_df = align_features(test_df, train_features)
    
    assert list(aligned_df.columns) == train_features
    assert aligned_df.loc[0, 'Contract_Two year'] == 0
