import pandas as pd

# Load the dataset
DATA_PATH = "data/raw/insurance_claims.csv"
df = pd.read_csv(DATA_PATH)

# Basic info
print("Dataset shape:", df.shape)
print("\nColumn names:\n", df.columns.tolist())

# Target distribution
print("\nTarget (FraudFound_P) distribution:\n", df["FraudFound_P"].value_counts(normalize=True))

# Quick stats for numeric columns
numeric_cols = [
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

print("\nNumeric column stats:\n", df[numeric_cols].describe())

# Check for missing values
print("\nMissing values per column:\n", df.isna().sum())