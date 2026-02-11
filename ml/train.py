import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.lightgbm
import joblib
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load the dataset
DATA_PATH = "data/raw/insurance_claims.csv"
df = pd.read_csv(DATA_PATH)

# Define target and features
TARGET = "FraudFound_P"

# Columns to drop
DROP_COLS = ['PolicyNumber', 'RepNumber', 'AddressChange_Claim', 'Year', TARGET]

# Features
X = df.drop(columns=DROP_COLS)
y = df[TARGET]

# Identify categorical & numerical features
numerical_features = [
    'Age',
    'VehiclePrice',
    'Deductible', 
    'DriverRating', 
    'Days_Policy_Accident',
    'Days_Policy_Claim',
    'PastNumberOfClaims',
    'AgeOfVehicle',
    'AgeOfPolicyHolder',
    'NumberOfSuppliments',
    'NumberOfCars'
]

for col in numerical_features:
    X[col] = pd.to_numeric(X[col], errors='coerce')
    X[col] = X[col].fillna(X[col].median())
    
categorical_features = [col for col in X.columns if col not in numerical_features]

# Convert categorical columns to 'category' dtype
for col in categorical_features:
    X[col] = X[col].astype('category')
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Start MLflow run
mlflow.set_experiment("insurance_claim_fraud_detection")
with mlflow.start_run():
    model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        scale_pos_weight=(len(y_train[y_train==0]) / len(y_train[y_train==1]))
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        categorical_feature=categorical_features
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    print(f"ROC-AUC: {roc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Log metrics and model to MLflow
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("num_leaves", 31)

    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("f1_score", f1)
    
    mlflow.lightgbm.log_model(model, artifact_path="lgbm_model")
    
    joblib.dump(model, os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    print(f"Model saved to {MODEL_DIR}/lgbm_model.pkl")    
    
print("Training complete. Model logged to MLflow.")