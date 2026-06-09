import pandas as pd

# Load dataset
df = pd.read_csv("data/Social Media Engagement Dataset.csv")

# First 5 rows
print("\nFIRST 5 ROWS")
print(df.head())

# Shape
print("\nDATASET SHAPE")
print(df.shape)

# Information
print("\nDATASET INFO")
print(df.info())

# Missing values
print("\nMISSING VALUES")
print(df.isnull().sum())