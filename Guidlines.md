# 📋 Project Guidelines & PRD
## Customer Churn Prediction + SHAP Explainability
> Built with Antigravity IDE | Intermediate ML Portfolio Project

---

## 1. Project Overview

| Field | Detail |
|---|---|
| **Project Name** | ChurnSight — AI-Powered Churn Intelligence Dashboard |
| **Goal** | Predict which customers are likely to churn and explain *why* using SHAP values |
| **Tech Stack** | Python, XGBoost, SHAP, Streamlit, Plotly, Pandas, Scikit-learn |
| **IDE** | Antigravity IDE |
| **Audience** | Portfolio reviewers, ML hiring managers, data science interviewers |

---

## 2. Product Requirements Document (PRD)

### 2.1 Problem Statement

Businesses lose revenue when customers leave (churn). Traditional analytics can flag who left — but not *why*. This project builds a system that:

- Predicts churn probability for each customer
- Ranks customers by risk level
- Explains the top factors driving each prediction using SHAP
- Presents insights in a clean, interactive dashboard

---

### 2.2 Core Features

#### MVP (Must Have)
- [ ] Data ingestion from CSV (Telco Customer Churn dataset)
- [ ] Preprocessing pipeline (encoding, scaling, null handling)
- [ ] XGBoost / LightGBM model training with cross-validation
- [ ] SHAP global feature importance chart
- [ ] SHAP local explanation for individual customers
- [ ] Churn risk score per customer (0–100%)
- [ ] Streamlit dashboard with filters and search

#### V2 (Nice to Have)
- [ ] Real-time CSV upload from user
- [ ] Export prediction results to CSV
- [ ] Model comparison (XGBoost vs Logistic Regression vs Random Forest)
- [ ] Threshold tuning slider (precision vs recall tradeoff)
- [ ] Email alert simulation for high-risk customers

---

### 2.3 User Stories

```
As a business analyst, I want to see which customers are most at risk
so I can prioritize retention outreach.

As a data scientist, I want to see which features drive churn
so I can validate the model's logic.

As a recruiter reviewing this portfolio, I want to see clean code,
clear documentation, and a polished UI.
```

---

## 3. Folder Structure

```
churnsight/
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv          # Original dataset
│   └── processed/
│       └── cleaned_data.pkl         # Preprocessed data
│
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb       # Feature engineering
│   ├── 03_modeling.ipynb            # Model training & evaluation
│   └── 04_shap_analysis.ipynb       # SHAP explainability
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                # Data cleaning pipeline
│   ├── train.py                     # Model training
│   ├── predict.py                   # Inference functions
│   └── explain.py                   # SHAP wrapper functions
│
├── models/
│   └── xgb_churn_model.pkl          # Saved trained model
│
├── app/
│   ├── main.py                      # Streamlit entry point
│   ├── pages/
│   │   ├── overview.py              # KPI summary page
│   │   ├── customer_detail.py       # Per-customer SHAP view
│   │   └── model_metrics.py         # Evaluation metrics page
│   └── components/
│       ├── charts.py                # Reusable Plotly charts
│       └── sidebar.py               # Sidebar filters
│
├── tests/
│   └── test_preprocess.py
│
├── requirements.txt
├── README.md
└── guidelines.md                    # This file
```

---

## 4. Dataset

**Source:** IBM Telco Customer Churn (free, public)
- Kaggle: `https://www.kaggle.com/datasets/blastchar/telco-customer-churn`
- Direct link: search "Telco Customer Churn Kaggle"
- Rows: ~7,000 customers | Columns: 21 features

**Key Features:**
| Feature | Type | Notes |
|---|---|---|
| `tenure` | Numeric | Months as customer |
| `MonthlyCharges` | Numeric | Monthly bill amount |
| `TotalCharges` | Numeric | All-time billing total |
| `Contract` | Categorical | Month-to-month, 1yr, 2yr |
| `InternetService` | Categorical | DSL, Fiber, None |
| `PaymentMethod` | Categorical | Credit card, bank transfer, etc. |
| `Churn` | Target | Yes / No (binary) |

---

## 5. Step-by-Step Build Guide

### Step 1 — Set Up Antigravity IDE

```bash
# Create new Python project in Antigravity IDE
# File → New Project → Python → "ChurnSight"

# Open integrated terminal and run:
pip install xgboost lightgbm shap streamlit plotly pandas scikit-learn joblib seaborn matplotlib
```

Create `requirements.txt`:
```
xgboost>=1.7.0
lightgbm>=3.3.0
shap>=0.42.0
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
seaborn>=0.12.0
matplotlib>=3.7.0
```

---

### Step 2 — EDA (Exploratory Data Analysis)

In `notebooks/01_eda.ipynb`:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../data/raw/telco_churn.csv')

# Check class imbalance
df['Churn'].value_counts(normalize=True).plot(kind='bar')

# Correlation heatmap (numeric only)
sns.heatmap(df.select_dtypes('number').corr(), annot=True, cmap='coolwarm')

# Churn rate by contract type
df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean()).plot(kind='bar')
```

**Key EDA Insights to Document:**
- Churn rate is ~26% (class imbalance — address this!)
- Month-to-month contracts churn 3x more than 2-year
- High monthly charges + low tenure = highest risk segment

---

### Step 3 — Preprocessing Pipeline

In `src/preprocess.py`:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Drop customer ID
    df.drop(columns=['customerID'], inplace=True)

    # Fix TotalCharges (has spaces for new customers)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    # Encode binary columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})

    # One-hot encode multi-class categoricals
    cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df

def split_and_scale(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, '../models/scaler.pkl')
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
```

---

### Step 4 — Model Training

In `src/train.py`:

```python
import xgboost as xgb
from sklearn.metrics import (classification_report, roc_auc_score,
                              confusion_matrix, RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import numpy as np

def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.7,   # Handles class imbalance (73/27 ratio)
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring='roc_auc')
    print(f"CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    model.fit(X_train, y_train)
    joblib.dump(model, '../models/xgb_churn_model.pkl')
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    return y_prob
```

**Target Metrics:**
| Metric | Target |
|---|---|
| ROC-AUC | > 0.82 |
| Recall (churn class) | > 0.75 |
| Precision | > 0.65 |

---

### Step 5 — SHAP Explainability

In `src/explain.py`:

```python
import shap
import matplotlib.pyplot as plt
import pandas as pd

def get_shap_explainer(model, X_train):
    explainer = shap.TreeExplainer(model)
    return explainer

def global_feature_importance(explainer, X_df, feature_names):
    """Beeswarm plot — shows direction + magnitude of each feature"""
    shap_values = explainer.shap_values(X_df)
    shap.summary_plot(shap_values, X_df, feature_names=feature_names,
                      plot_type='beeswarm', show=False)
    plt.tight_layout()
    plt.savefig('shap_global.png', dpi=150)
    plt.close()
    return shap_values

def local_explanation(explainer, customer_row, feature_names):
    """Waterfall plot — explains ONE customer's prediction"""
    shap_values = explainer.shap_values(customer_row)
    shap.plots.waterfall(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=customer_row.iloc[0],
        feature_names=feature_names
    ))
```

---

### Step 6 — Streamlit Dashboard (UI/UX)

#### Design System

Follow these UI principles for a **modern, professional dashboard**:

**Color Palette:**
```python
COLORS = {
    "primary":    "#6C63FF",   # Purple — brand accent
    "danger":     "#FF4B4B",   # Red — high churn risk
    "warning":    "#FFA62B",   # Amber — medium risk
    "success":    "#2DD4BF",   # Teal — low risk / positive
    "background": "#0F1117",   # Dark background (dark mode)
    "card":       "#1E2130",   # Card surface
    "text":       "#FAFAFA",   # Primary text
    "muted":      "#9CA3AF",   # Secondary text
}
```

**Custom CSS in Streamlit:**
```python
# In app/main.py
st.markdown("""
<style>
    /* Dark professional theme */
    .stApp { background-color: #0F1117; }

    /* Metric cards */
    .metric-card {
        background: #1E2130;
        border-radius: 12px;
        padding: 20px 24px;
        border: 1px solid #2D3748;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Risk badge */
    .badge-high   { background:#FF4B4B20; color:#FF4B4B; border-radius:20px; padding:4px 12px; font-size:12px; font-weight:600; }
    .badge-medium { background:#FFA62B20; color:#FFA62B; border-radius:20px; padding:4px 12px; font-size:12px; font-weight:600; }
    .badge-low    { background:#2DD4BF20; color:#2DD4BF; border-radius:20px; padding:4px 12px; font-size:12px; font-weight:600; }

    /* Table styling */
    .dataframe { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)
```

#### Main Dashboard Layout (`app/main.py`):

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
from src.preprocess import preprocess
from src.explain import get_shap_explainer, local_explanation

st.set_page_config(
    page_title="ChurnSight",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.image("assets/logo.png", width=160)   # Add a simple logo
    st.title("ChurnSight")
    st.caption("AI-Powered Churn Intelligence")
    st.divider()

    risk_filter = st.multiselect(
        "Filter by Risk Level",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"]
    )
    threshold = st.slider("Churn Threshold", 0.0, 1.0, 0.5, 0.05)
    st.divider()
    st.caption("Model: XGBoost v1.7 | Dataset: Telco Churn")

# ─── Header ────────────────────────────────────────────────
st.markdown("## 📡 ChurnSight Dashboard")
st.caption("Real-time churn risk predictions with AI explainability")
st.divider()

# ─── KPI Row ───────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Customers", "7,043", delta=None)
with col2:
    st.metric("High Risk", "1,823", delta="↑ 12% vs last month", delta_color="inverse")
with col3:
    st.metric("Avg Churn Probability", "26.4%")
with col4:
    st.metric("Model AUC", "0.847")

st.divider()

# ─── Two-Column Layout ─────────────────────────────────────
left, right = st.columns([2, 1])

with left:
    st.subheader("Customer Risk Table")
    # Load predictions dataframe and display with color coding
    # df filtered by risk_filter and threshold
    st.dataframe(styled_df, use_container_width=True, height=400)

with right:
    st.subheader("Risk Distribution")
    fig = px.pie(
        values=[high, medium, low],
        names=["High", "Medium", "Low"],
        color_discrete_sequence=["#FF4B4B", "#FFA62B", "#2DD4BF"],
        hole=0.6
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
```

#### Customer Detail Page (`app/pages/customer_detail.py`):

```python
import streamlit as st

st.title("🔍 Customer Detail")
st.caption("SHAP-based explanation for individual predictions")

customer_id = st.text_input("Enter Customer ID", "7590-VHVEG")

if customer_id:
    # Fetch customer row, run SHAP
    # Display waterfall chart
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Churn Probability", f"{prob*100:.1f}%")
        st.metric("Risk Level", risk_label)

    with col2:
        st.markdown("**Top Risk Factors:**")
        for factor, impact in top_factors:
            color = "🔴" if impact > 0 else "🟢"
            st.markdown(f"{color} **{factor}**: {impact:+.3f}")

    st.subheader("SHAP Waterfall Explanation")
    st.pyplot(shap_fig)
```

---

## 6. UI/UX Design Principles

### Layout
- Use **wide layout** (`layout="wide"`) always
- **3-column KPI row** at the top — always visible
- **Left-heavy split** (2:1 ratio) for tables + charts
- Use `st.tabs()` for switching between Global/Local SHAP views

### Typography
- Headers: bold, large (`##` or `###`)
- Captions below every section title
- Avoid walls of text — use metrics, badges, and short labels

### Charts (Plotly)
- Always set `paper_bgcolor='rgba(0,0,0,0)'` for dark mode
- Use consistent color palette (see Design System above)
- Add `hovertemplate` for informative tooltips
- Prefer **bar charts** for feature importance, **gauge charts** for risk scores

### Interactivity
- Sidebar filters update all charts simultaneously (use `st.session_state`)
- Click on a row in the table → opens that customer's SHAP waterfall
- Add loading spinners (`st.spinner()`) for model inference

### Responsiveness
- Always use `use_container_width=True` on charts
- Avoid fixed pixel widths
- Test at 1280px, 1440px, and 1920px

---

## 7. Model Evaluation Checklist

Before publishing, verify:

- [ ] ROC-AUC > 0.82 on held-out test set
- [ ] Confusion matrix shows recall > 0.75 for churn class
- [ ] Cross-validation spread < 0.03 (no overfitting)
- [ ] SHAP beeswarm matches business intuition (contract type, tenure matter most)
- [ ] No data leakage (scaler fit only on train set)
- [ ] `TotalCharges` NaN values handled correctly
- [ ] Class imbalance addressed (`scale_pos_weight` or `class_weight`)

---

## 8. Git & Documentation Standards

### Commit Convention
```
feat: add SHAP waterfall chart for customer detail page
fix: resolve TotalCharges NaN preprocessing bug
refactor: extract chart logic into components/charts.py
docs: update README with dashboard screenshots
```

### README Must Include
1. Project overview (2-3 sentences)
2. Screenshot of the dashboard
3. Quick start (3 commands to run)
4. Model performance table
5. Key SHAP insights (2-3 bullet points)
6. Dataset source and license

### README Quick Start Template
```markdown
## 🚀 Quick Start

git clone https://github.com/yourname/churnsight
cd churnsight
pip install -r requirements.txt
streamlit run app/main.py
```

---

## 9. Deployment Options

| Platform | Cost | Notes |
|---|---|---|
| **Streamlit Cloud** | Free | Best for portfolios — 1-click deploy from GitHub |
| **Hugging Face Spaces** | Free | Great for ML community visibility |
| **Render** | Free tier | Good if you need a persistent backend |

**Recommended: Streamlit Cloud**
1. Push to GitHub
2. Go to `share.streamlit.io`
3. Connect repo → select `app/main.py`
4. Done — shareable URL in 2 minutes

---

## 10. Interview Talking Points

Prepare answers to these questions:

**"Walk me through your model choice."**
> XGBoost handles non-linear relationships well and natively supports class imbalance via `scale_pos_weight`. I compared it against Logistic Regression and Random Forest using 5-fold CV AUC, and XGBoost outperformed both.

**"How did you handle class imbalance?"**
> The dataset has ~26% churn (minority class). I used `scale_pos_weight = (1 - churn_rate) / churn_rate ≈ 2.7` in XGBoost, which penalizes false negatives more heavily. I also used `StratifiedKFold` to maintain class ratio across folds.

**"What do the SHAP values tell you?"**
> SHAP values are additive feature attributions — they tell us how much each feature pushed the prediction above or below the base rate. For example, a customer with a month-to-month contract and high monthly charges gets a large positive SHAP value for those features, meaning they strongly increase the predicted churn probability.

**"What would you do to improve this in production?"**
> I would add: (1) a model retraining pipeline on fresh data, (2) drift detection to catch feature distribution shifts, (3) a feedback loop where retention teams log outcomes to label new data, and (4) A/B testing to measure whether model-driven interventions actually reduce churn.

---
---

*Built with ❤️ using Antigravity IDE | ChurnSight v1.0*