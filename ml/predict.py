import numpy as np
import os
import joblib
import pandas as pd
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models/lgbm_model.pkl")
PREPROCESS_PATH = os.path.join(BASE_DIR, "models/preprocess.pkl")
EXPLAINER_PATH = os.path.join(BASE_DIR, "models/shap_explainer.pkl")

# Load model
model = joblib.load(MODEL_PATH)

# Load preprocessing info
with open(PREPROCESS_PATH, "rb") as f:
    preprocess_info = pickle.load(f)

# Load explainer
with open(EXPLAINER_PATH, "rb") as f:
    explainer = pickle.load(f)

def preprocess_input(data: dict):
    df = pd.DataFrame([data])
    
    numeric_cols = preprocess_info["numeric_cols"]
    categorical_cols = preprocess_info["categorical_cols"]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    
    return df

def predict_claim(data: dict):
    df = preprocess_input(data)
    
    proba = model.predict_proba(df)[0][1]
    prediction = int(proba > 0.5)
    
    shap_values = explainer.shap_values(df)
    
    shap_values_1 = shap_values[0]
    
    # Top 5 impactful features
    top_indicies = np.argsort(np.abs(shap_values_1))[::-1][:5]
    
    feature_names = df.columns
    
    key_risk_factors = [
        f"{feature_names[i]} : {df.iloc[0, i]}"
        for i in top_indicies
    ]
    
    # Risk level classification
    if proba < 0.3:
        risk_level = "low"
    elif proba <0.7:
        risk_level = "medium"
    else:
        risk_level = "high"
    
    return {
        "risk_level": risk_level,
        "fraud_probability": float(proba),
        "prediction": prediction,
        "key_risk_factors": key_risk_factors
    }