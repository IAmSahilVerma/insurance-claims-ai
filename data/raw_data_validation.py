import pandas as pd

df = pd.read_csv("data/raw/insurance_claims.csv")
print(df.shape)
print(df["FraudFound_P"].value_counts())
print(df.columns)