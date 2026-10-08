from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "house_prices.csv"

df = pd.read_csv(DATA_PATH)

print("HOUSE PRICE DATASET")
print("-------------------")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nLocation Distribution:")
print(df["location"].value_counts())

print("\nAverage Price by Location:")
print(df.groupby("location")["price"].mean())