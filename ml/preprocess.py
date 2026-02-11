import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/raw/insurance_claims.csv")
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

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

print(X.dtypes)

preprocess_info = {
    "numeric_cols": numerical_features,
    "categorical_cols": categorical_features
}

with open("models/preprocess.pkl", "wb") as f:
    pickle.dump(preprocess_info, f)