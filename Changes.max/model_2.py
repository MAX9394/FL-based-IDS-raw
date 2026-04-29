# model_2.py
# ------------------------------------------------------------
# PortScan IDS Trainer (Standalone + Federated Ready)
# Rebuilt properly from model_2.ipynb intent
#
# Usage:
#   python model_2.py --data Portscan-Friday-no-metadata.csv
#
# Output:
#   model_2.pkl                (single packaged artifact)
#   model_2_metrics.json
#   model_2_federated.npz
# ------------------------------------------------------------

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

RANDOM_STATE = 42

COLS_TO_DROP = [
    "Fwd Packets Length Total", "Bwd Packets Length Total",
    "Fwd Packet Length Max", "Bwd Packet Length Max",
    "Fwd Packet Length Min", "Bwd Packet Length Min",
    "Fwd Packet Length Std", "Bwd Packet Length Std",
    "Flow IAT Std", "Flow IAT Total",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags",
    "Packet Length Variance",
    "PSH Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Avg Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Bwd Bytes",
    "Init Fwd Win Bytes", "Init Bwd Win Bytes",
    "Fwd Act Data Packets", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

LABEL_CANDIDATES = ["Label", "label", "Class", "Attack", "Target"]


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def detect_label_column(df):
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col

    # fallback: find object column containing PortScan/Benign
    for col in df.columns:
        vals = df[col].astype(str).str.strip().unique().tolist()
        if "PortScan" in vals and "Benign" in vals:
            return col

    raise ValueError("Could not detect label column.")


def clean_numeric(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df.clip(lower=0, upper=1e12)
    return df


def optimize_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-12
    )

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def evaluate(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "classification_report": classification_report(
            y_true, preds, output_dict=True, zero_division=0
        )
    }

    return metrics


# ------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ------------------------------------------------------------

def train(csv_path):
    print("[1/7] Loading dataset...")
    df = pd.read_csv(csv_path)

    label_col = detect_label_column(df)
    print(f"Detected label column: {label_col}")

    # Keep only Benign + PortScan
    df = df[df[label_col].isin(["Benign", "PortScan"])].copy()

    # Encode labels
    y = df[label_col].map({
        "Benign": 0,
        "PortScan": 1
    }).astype(int)

    # Drop label + unwanted columns
    drop_cols = [label_col] + [c for c in COLS_TO_DROP if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Keep numeric only
    X = X.select_dtypes(include=[np.number])

    # Clean
    print("[2/7] Cleaning data...")
    X = clean_numeric(X)

    # log1p transform
    print("[3/7] Applying log transform...")
    X = np.log1p(X)

    feature_names = list(X.columns)

    # Train/test split
    print("[4/7] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y.values,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y.values
    )

    # Scale
    print("[5/7] Scaling...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Balance
    print("[6/7] Applying SMOTETomek...")
    sampler = SMOTETomek(
        sampling_strategy=0.10,
        random_state=RANDOM_STATE
    )
    X_train_bal, y_train_bal = sampler.fit_resample(X_train, y_train)

    # Model
    print("[7/7] Training Logistic Regression...")
    model = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        warm_start=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train_bal, y_train_bal)

    # Threshold tuning
    probs = model.predict_proba(X_test)[:, 1]
    threshold = optimize_threshold(y_test, probs)

    # Metrics
    metrics = evaluate(y_test, probs, threshold)

    # --------------------------------------------------------
    # SAVE SINGLE PACKAGED ARTIFACT
    # --------------------------------------------------------
    artifact = {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "feature_names": feature_names,
        "metadata": {
            "model_name": "PortScan IDS Logistic Regression",
            "attack_class": "PortScan",
            "benign_class": "Benign",
            "num_features": len(feature_names),
            "train_samples": int(len(X_train_bal)),
            "test_samples": int(len(X_test)),
            "random_state": RANDOM_STATE
        }
    }

    joblib.dump(artifact, "model_2.pkl")

    with open("model_2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Federated export
    np.savez(
        "model_2_federated.npz",
        coef=model.coef_,
        intercept=model.intercept_,
        num_features=len(feature_names)
    )

    # Console Summary
    print("\nTraining Complete.")
    print("-" * 50)
    print(f"Features Used : {len(feature_names)}")
    print(f"Best Threshold: {threshold:.4f}")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1 Score      : {metrics['f1_score']:.4f}")
    print(f"ROC AUC       : {metrics['roc_auc']:.4f}")
    print("-" * 50)
    print("Saved:")
    print("  model_2.pkl")
    print("  model_2_metrics.json")
    print("  model_2_federated.npz")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to PortScan CSV dataset"
    )

    args = parser.parse_args()

    csv_path = Path(args.data)

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    train(csv_path)