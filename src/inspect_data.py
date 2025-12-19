import pandas as pd

# Load data
df = pd.read_csv("data/walmart_sales.csv")


print("===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA INFO =====")
print(df.info())

print("\n===== MISSING VALUES =====")
print(df.isna().sum())

print("\n===== BASIC STATISTICS =====")
print(df.describe(include="all"))
