"""
Federated Learning Preprocessing Pipeline for CICIDS2017
========================================================

This script:
1. Loads CICIDS2017 CSV files
2. Cleans invalid rows (NaN / Inf)
3. Filters required attack classes
4. Encodes labels
5. Selects numerical features
6. Removes highly correlated features
7. Scales features using StandardScaler
8. Creates non-IID federated client datasets
9. Exports processed CSV files

OUTPUT:
-------
processed_output/
    global_processed.csv
    client_1.csv
    client_2.csv
    client_3.csv
    client_4.csv

REQUIREMENTS:
-------------
pip install pandas numpy scikit-learn

USAGE:
------
1. Place all CICIDS2017 CSV files inside:
       ./raw_dataset/

2. Run:
       python preprocess_federated_ids.py
"""

import os
import glob
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================

RAW_DATASET_FOLDER = "raw_dataset"
OUTPUT_FOLDER = "processed_output"

SELECTED_LABELS = [
    "Benign",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "DDoS",
    "PortScan"
]

LABEL_MAPPING = {
    "Benign": 0,
    "DoS Hulk": 1,
    "DoS GoldenEye": 1,
    "DoS slowloris": 1,
    "DoS Slowhttptest": 1,
    "DDoS": 2,
    "PortScan": 3
}

RANDOM_STATE = 42

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# LOAD ALL CSV FILES
# ============================================================

csv_files = glob.glob(os.path.join(RAW_DATASET_FOLDER, "*.csv"))

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No CSV files found inside '{RAW_DATASET_FOLDER}'"
    )

print(f"\n[INFO] Found {len(csv_files)} CSV files.")

dataframes = []

for file in csv_files:
    print(f"[INFO] Loading: {file}")
    df = pd.read_csv(file, low_memory=False)
    dataframes.append(df)

df = pd.concat(dataframes, ignore_index=True)

print(f"\n[INFO] Combined dataset shape: {df.shape}")

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()

# ============================================================
# VERIFY LABEL COLUMN
# ============================================================

if "Label" not in df.columns:
    raise ValueError("Column 'Label' not found in dataset.")

# ============================================================
# FILTER REQUIRED LABELS
# ============================================================

df = df[df["Label"].isin(SELECTED_LABELS)]

print(f"[INFO] Shape after label filtering: {df.shape}")

# ============================================================
# REPLACE INF / -INF
# ============================================================

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ============================================================
# DROP NaN ROWS
# ============================================================

before_drop = len(df)

df.dropna(inplace=True)

after_drop = len(df)

print(f"[INFO] Removed {before_drop - after_drop} invalid rows.")

# ============================================================
# LABEL ENCODING
# ============================================================

df["Label"] = df["Label"].map(LABEL_MAPPING)

# ============================================================
# KEEP ONLY NUMERIC FEATURES
# ============================================================

numeric_df = df.select_dtypes(include=[np.number]).copy()

# ============================================================
# REMOVE CONSTANT COLUMNS
# ============================================================

nunique = numeric_df.nunique()

constant_columns = nunique[nunique <= 1].index.tolist()

if constant_columns:
    print(f"[INFO] Removing {len(constant_columns)} constant columns.")
    numeric_df.drop(columns=constant_columns, inplace=True)

# ============================================================
# SEPARATE FEATURES AND LABELS
# ============================================================

X = numeric_df.drop(columns=["Label"])
y = numeric_df["Label"]

print(f"[INFO] Feature count before correlation filtering: {X.shape[1]}")

# ============================================================
# REMOVE HIGHLY CORRELATED FEATURES
# ============================================================

correlation_matrix = X.corr().abs()

upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

to_drop = [
    column
    for column in upper_triangle.columns
    if any(upper_triangle[column] > 0.95)
]

X.drop(columns=to_drop, inplace=True)

print(f"[INFO] Removed {len(to_drop)} highly correlated features.")
print(f"[INFO] Feature count after filtering: {X.shape[1]}")

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# ============================================================
# FINAL DATAFRAME
# ============================================================

final_df = X_scaled.copy()
final_df["Label"] = y.values

print(f"[INFO] Final processed shape: {final_df.shape}")

# ============================================================
# SAVE GLOBAL DATASET
# ============================================================

global_output_path = os.path.join(
    OUTPUT_FOLDER,
    "global_processed.csv"
)

final_df.to_csv(global_output_path, index=False)

print(f"[INFO] Saved global dataset: {global_output_path}")

# ============================================================
# CREATE NON-IID CLIENT SPLITS
# ============================================================

# Client distribution strategy:
#
# Client 1 -> Mostly DoS
# Client 2 -> Mostly DDoS
# Client 3 -> Mostly PortScan
# Client 4 -> Mixed traffic

benign_df = final_df[final_df["Label"] == 0]
dos_df = final_df[final_df["Label"] == 1]
ddos_df = final_df[final_df["Label"] == 2]
portscan_df = final_df[final_df["Label"] == 3]

# Shuffle each class
benign_df = benign_df.sample(frac=1, random_state=RANDOM_STATE)
dos_df = dos_df.sample(frac=1, random_state=RANDOM_STATE)
ddos_df = ddos_df.sample(frac=1, random_state=RANDOM_STATE)
portscan_df = portscan_df.sample(frac=1, random_state=RANDOM_STATE)

# ============================================================
# CLIENT 1 (Mostly DoS)
# ============================================================

client_1 = pd.concat([
    dos_df.iloc[:5000],
    benign_df.iloc[:2000],
    ddos_df.iloc[:500],
    portscan_df.iloc[:500]
])

# ============================================================
# CLIENT 2 (Mostly DDoS)
# ============================================================

client_2 = pd.concat([
    ddos_df.iloc[500:5500],
    benign_df.iloc[2000:4000],
    dos_df.iloc[5000:5500],
    portscan_df.iloc[500:1000]
])

# ============================================================
# CLIENT 3 (Mostly PortScan)
# ============================================================

client_3 = pd.concat([
    portscan_df.iloc[:1956],
    benign_df.iloc[4000:6000],
    dos_df.iloc[5500:5750],
    ddos_df.iloc[5500:5750]
])

# ============================================================
# CLIENT 4 (Mixed)
# ============================================================

client_4 = pd.concat([
    benign_df.iloc[6000:9000],
    dos_df.iloc[6000:8000],
    ddos_df.iloc[6000:8000],
    portscan_df.iloc[1000:1956]
])

clients = {
    "client_1.csv": client_1,
    "client_2.csv": client_2,
    "client_3.csv": client_3,
    "client_4.csv": client_4
}

# ============================================================
# SAVE CLIENT DATASETS
# ============================================================

for filename, client_df in clients.items():

    client_df = client_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    output_path = os.path.join(OUTPUT_FOLDER, filename)

    client_df.to_csv(output_path, index=False)

    print(f"[INFO] Saved {filename} -> Shape: {client_df.shape}")

# ============================================================
# SUMMARY
# ============================================================

print("\n==============================")
print(" PREPROCESSING COMPLETE")
print("==============================")

print("\nGenerated Files:")
for file in os.listdir(OUTPUT_FOLDER):
    print(f" - {file}")

print("\nLabel Encoding:")
print("0 -> BENIGN")
print("1 -> DoS")
print("2 -> DDoS")
print("3 -> PortScan")

print("\nYou can now use these CSVs for:")
print(" - Centralized Logistic Regression")
print(" - Federated Learning")
print(" - FedAvg aggregation")
