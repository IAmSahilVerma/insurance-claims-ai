import os
import joblib
import pandas as pd
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models/lgbm_model.pkl")
PREPROCESS_PATH = os.path.join(BASE_DIR, "models/preprocess.pkl")

# Load model
model = joblib.load(MODEL_PATH)

# Load preprocessing info
with open(PREPROCESS_PATH, "rb") as f:
    preprocess_info = pickle.load(f)
    
numeric_cols = preprocess_info["numeric_cols"]
categorical_cols = preprocess_info["categorical_cols"]

def preprocess_input(data: dict):
    df = pd.DataFrame([data])
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    
    return df

def predict_claim(data: dict):
    df = preprocess_input(data)
    
    proba = model.predict_proba(df)[0][1]
    prediction = int(proba > 0.5)
    
    return {
        "fraud_probability": float(proba),
        "prediction": prediction
    }