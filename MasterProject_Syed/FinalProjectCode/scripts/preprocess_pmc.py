# scripts/preprocess_pmc.py
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))


import pandas as pd
from config import PMC_RAW_PATH, PMC_CLEAN_PATH

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

print("Loading PMC Patients dataset from:", PMC_RAW_PATH)
df = pd.read_csv(PMC_RAW_PATH)

# Inspect raw data
print("\nRaw PMC Data Preview:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# Rename columns as needed.
# Here, we rename the 'patient' column to 'abstract' so that we have a consistent field for text.
df.rename(columns={
    "patient": "abstract"
}, inplace=True)

# Optionally, you can rename other columns if necessary. For example, if you want to rename
# 'Patient_Title' to 'title', do that here. (Based on your preview, the 'title' column is already present.)

# Convert 'year' and 'age' columns to numeric if they exist.
for col in ["year", "age"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows missing essential fields (title and abstract)
df = df.dropna(subset=["title", "abstract"]).reset_index(drop=True)

print("\nCleaned PMC Data Preview:")
print(df.head())
print("\nData types after cleaning:")
print(df.dtypes)

# Save the cleaned dataset as JSON for later use (e.g., indexing)
df.to_json(PMC_CLEAN_PATH, orient="records", indent=4)
print(f"\n✅ Cleaned PMC dataset saved to: {PMC_CLEAN_PATH}")
