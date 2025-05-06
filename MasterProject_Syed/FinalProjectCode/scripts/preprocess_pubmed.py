# scripts/preprocess_pubmed.py

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import os
import pandas as pd
from config import PUBMED_RAW_PATH, PUBMED_CLEAN_PATH

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

print("Loading PubMed Articles dataset from:", PUBMED_RAW_PATH)
# Load the CSV file (adjust parameters if needed)
df = pd.read_csv(PUBMED_RAW_PATH)

# Display raw data preview and data types for inspection
print("\n🔍 Raw data preview:")
print(df.head())
print("\nData types before cleaning:")
print(df.dtypes)

# Rename columns:
# Rename 'article' to 'title' so that our pipeline can find a title.
df.rename(columns={
    "article": "title"
}, inplace=True)

# Drop rows missing essential fields: 'title' and 'abstract'
df = df.dropna(subset=["title", "abstract"]).reset_index(drop=True)

print("\nCleaned data preview:")
print(df.head())
print("\nData types after cleaning:")
print(df.dtypes)

# Save the cleaned dataset to a JSON file for further processing (indexing)
df.to_json(PUBMED_CLEAN_PATH, orient="records", indent=4)
print(f"\n✅ Cleaned PubMed dataset saved to: {PUBMED_CLEAN_PATH}")
