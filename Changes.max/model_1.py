"""
model_1.py
==========

Client-side DDoS detection model for federated learning systems.

Purpose
-------
This script is a corrected and production-grade version of what model_1.ipynb
was trying to do:

1. Train a local Logistic Regression model on network-flow data
2. Detect benign vs DDoS traffic individually
3. Save deployable artifacts
4. Expose weights for federated aggregation
5. Support loading global parameters from a server

Why this version is better
--------------------------
- Trains properly (not 1 iteration / not 100 rows only)
- Uses pipeline-consistent preprocessing
- Handles infinities / NaNs
- Uses class imbalance controls
- Saves scaler + model + metadata
- Supports FedAvg integration cleanly

Expected Dataset
----------------
CSV with a column named: Label

Benign labels may contain:
    BENIGN, Benign, normal, 0

Attack labels:
    anything else -> attack (1)

Usage
-----
Train local model:

    python model_1.py train --data ddos.csv --out client_1.pkl

Evaluate:

    python model_1.py eval --model client_1.pkl --data test.csv

Predict one row batch:

    python model_1.py predict --model client_1.pkl --data sample.csv

Export local weights for server:

    python model_1.py export --model client_1.pkl

Load global weights from server:

    python model_1.py import-global --model client_1.pkl --weights global_weights.npz
"""

import argparse
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from sklearn.linear_model import LogisticRegression

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False


# =========================================================
# CONFIG
# =========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_CONFIG = dict(
    solver="saga",
    penalty="l2",
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

DROP_COLUMNS = [
    "Fwd Packets Length Total", "Bwd Packets Length Total", "Fwd Packet Length Max", 
    "Bwd Packet Length Max", "Fwd Packet Length Min", "Bwd Packet Length Min",
    "Fwd Packet Length Std", "Bwd Packet Length Std", "Flow IAT Std", "Flow IAT Total",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Std", 
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", 
    "Bwd URG Flags", "Packet Length Variance", "PSH Flag Count", "URG Flag Count", 
    "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Avg Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Avg Bytes/Bulk", 
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", 
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Bwd Bytes",
    "Init Fwd Win Bytes", "Init Bwd Win Bytes", "Fwd Act Data Packets", 
    "Fwd Seg Size Min", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]


# =========================================================
# DATA PROCESSING
# =========================================================

def normalize_label(value):
    s = str(value).strip().lower()

    benign_tokens = {
        "benign",
        "normal",
        "0",
        "legitimate",
    }

    return 0 if s in benign_tokens else 1


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    if "Label" not in df.columns:
        raise ValueError("Dataset must contain a 'Label' column.")

    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=col)

    y = df["Label"].apply(normalize_label).astype(int)
    X = df.drop(columns=["Label"])

    # Keep numeric only
    X = X.select_dtypes(include=[np.number])

    # Replace invalid values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y


# =========================================================
# TRAINING
# =========================================================

def train_local_model(data_path, out_path):
    X, y = load_dataset(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Balance training set
    if HAS_SMOTE:
        values, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(values, counts))

        minority = min(class_counts, key=class_counts.get)
        majority = max(class_counts, key=class_counts.get)

        min_count = class_counts[minority]
        maj_count = class_counts[majority]

        ratio = min_count / maj_count

        if ratio < 0.5:
            smote = SMOTE(
                random_state=RANDOM_STATE,
                sampling_strategy=0.5
            )
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print("SMOTE applied.")
        else:
            print("SMOTE skipped (classes already balanced enough).")

    model = LogisticRegression(**MODEL_CONFIG)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n=== LOCAL MODEL REPORT ===")
    print(classification_report(y_test, y_pred, digits=4))
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("ROC AUC  :", round(roc_auc_score(y_test, y_prob), 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "config": MODEL_CONFIG,
    }

    joblib.dump(artifact, out_path)
    print(f"\nSaved model to: {out_path}")


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(model_path, data_path):
    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    features = artifact["feature_names"]

    X, y = load_dataset(data_path)
    X = X.reindex(columns=features, fill_value=0)

    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    print("\n=== EVALUATION REPORT ===")
    print(classification_report(y, y_pred, digits=4))
    print("Accuracy :", round(accuracy_score(y, y_pred), 4))
    print("ROC AUC  :", round(roc_auc_score(y, y_prob), 4))


# =========================================================
# PREDICTION
# =========================================================

def predict(model_path, data_path):
    artifact = joblib.load(model_path)
    model = artifact["model"]
    scaler = artifact["scaler"]
    features = artifact["feature_names"]

    df = pd.read_csv(data_path)
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = df.reindex(columns=features, fill_value=0)
    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result = pd.DataFrame({
        "prediction": preds,
        "attack_probability": probs
    })

    print(result)


# =========================================================
# FEDERATED EXPORT
# =========================================================

def export_weights(model_path):
    artifact = joblib.load(model_path)
    model = artifact["model"]

    np.savez(
        "local_weights.npz",
        coef=model.coef_,
        intercept=model.intercept_
    )

    print("Saved federated payload: local_weights.npz")


# =========================================================
# FEDERATED IMPORT
# =========================================================

def import_global_weights(model_path, weights_path):
    artifact = joblib.load(model_path)
    model = artifact["model"]

    data = np.load(weights_path)

    model.coef_ = data["coef"]
    model.intercept_ = data["intercept"]
    model.classes_ = np.array([0, 1])

    artifact["model"] = model
    joblib.dump(artifact, model_path)

    print("Global weights loaded into local model.")


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="command")

    p1 = sub.add_parser("train")
    p1.add_argument("--data", required=True)
    p1.add_argument("--out", required=True)

    p2 = sub.add_parser("eval")
    p2.add_argument("--model", required=True)
    p2.add_argument("--data", required=True)

    p3 = sub.add_parser("predict")
    p3.add_argument("--model", required=True)
    p3.add_argument("--data", required=True)

    p4 = sub.add_parser("export")
    p4.add_argument("--model", required=True)

    p5 = sub.add_parser("import-global")
    p5.add_argument("--model", required=True)
    p5.add_argument("--weights", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train_local_model(args.data, args.out)

    elif args.command == "eval":
        evaluate_model(args.model, args.data)

    elif args.command == "predict":
        predict(args.model, args.data)

    elif args.command == "export":
        export_weights(args.model)

    elif args.command == "import-global":
        import_global_weights(args.model, args.weights)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()