
"""
portscan_rf_model.py
=========================================================
Robust Random Forest PortScan Classifier

Purpose
-------
Train and evaluate a production-oriented Random Forest
classifier for binary PortScan detection.

Target Classes
--------------
0 -> Benign
1 -> PortScan

Key Design Decisions
--------------------
- Random Forest instead of Logistic Regression
- Multi-stage cleaning
- Feature validation
- Probability calibration
- Controlled tree depth to reduce overfitting
- Balanced subsampling
- OOB validation support
- Threshold optimization
- Serializable artifact for live detector usage

Expected Dataset
----------------
CSV with a label column such as:
    Label
    label
    Class
    Attack

Expected labels:
    Benign
    PortScan

Usage
-----
Train:
    python portscan_rf_model.py train --data dataset.csv --out portscan_rf.pkl

Evaluate:
    python portscan_rf_model.py eval --model portscan_rf.pkl --data dataset.csv

Predict:
    python portscan_rf_model.py predict --model portscan_rf.pkl --data sample.csv
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve
)

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

LABEL_CANDIDATES = [
    "Label",
    "label",
    "Class",
    "Attack",
    "Target"
]

# Remove noisy / unstable / redundant fields
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
# HELPERS
# =========================================================

def detect_label_column(df):
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError("Could not detect label column.")


def normalize_labels(series):
    s = series.astype(str).str.strip().str.lower()

    return np.where(
        s == "benign",
        0,
        1
    )


def clean_features(df):
    # Drop unwanted columns
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")

    # Keep numeric only
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace bad values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # Clip extreme values
    df = df.clip(lower=0, upper=1e12)

    # Log compression
    df = np.log1p(df)

    return df


def validate_features(df):
    if len(df.columns) == 0:
        raise ValueError("No usable numeric features found.")

    if df.isnull().sum().sum() > 0:
        raise ValueError("NaN values remain after cleaning.")

    return True


def optimize_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)

    if len(thresholds) == 0:
        return 0.5

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
    ) / (precision[:-1] + recall[:-1] + 1e-12)

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
            y_true,
            preds,
            output_dict=True,
            zero_division=0
        )
    }

    return metrics


# =========================================================
# DATA LOADING
# =========================================================

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    label_col = detect_label_column(df)

    # Keep only benign + portscan
    mask = df[label_col].astype(str).str.strip().isin([
        "Benign",
        "PortScan"
    ])

    df = df[mask].copy()

    y = normalize_labels(df[label_col])

    X = df.drop(columns=[label_col], errors="ignore")

    X = clean_features(X)

    validate_features(X)

    return X, y


# =========================================================
# MODEL CREATION
# =========================================================

def build_model():
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    calibrated = CalibratedClassifierCV(
        estimator=rf,
        method="isotonic",
        cv=3
    )

    return calibrated


# =========================================================
# TRAINING
# =========================================================

def train(data_path, out_path):
    print("[1/7] Loading dataset...")
    X, y = load_dataset(data_path)

    feature_names = list(X.columns)

    print("[2/7] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("[3/7] Building Random Forest...")
    model = build_model()

    print("[4/7] Training model...")
    model.fit(X_train, y_train)

    print("[5/7] Running inference...")
    probs = model.predict_proba(X_test)[:, 1]

    print("[6/7] Optimizing threshold...")
    threshold = optimize_threshold(y_test, probs)

    print("[7/7] Evaluating...")
    metrics = evaluate(y_test, probs, threshold)

    artifact = {
        "model": model,
        "threshold": threshold,
        "feature_names": feature_names,
        "metadata": {
            "model_type": "RandomForest PortScan Detector",
            "attack_class": "PortScan",
            "benign_class": "Benign",
            "random_state": RANDOM_STATE,
            "features": len(feature_names),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test))
        }
    }

    joblib.dump(artifact, out_path)

    metrics_path = Path(out_path).with_suffix(".metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nTraining Complete.")
    print("-" * 50)
    print(f"Threshold  : {threshold:.4f}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"F1 Score   : {metrics['f1_score']:.4f}")
    print(f"ROC AUC    : {metrics['roc_auc']:.4f}")
    print("-" * 50)

    print(f"Saved model   : {out_path}")
    print(f"Saved metrics : {metrics_path}")


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(model_path, data_path):
    artifact = joblib.load(model_path)

    model = artifact["model"]
    threshold = artifact["threshold"]
    feature_names = artifact["feature_names"]

    X, y = load_dataset(data_path)

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]

    probs = model.predict_proba(X)[:, 1]

    metrics = evaluate(y, probs, threshold)

    print("\n=== EVALUATION REPORT ===")
    print(json.dumps(metrics, indent=4))


# =========================================================
# PREDICTION
# =========================================================

def predict(model_path, data_path):
    artifact = joblib.load(model_path)

    model = artifact["model"]
    threshold = artifact["threshold"]
    feature_names = artifact["feature_names"]

    df = pd.read_csv(data_path)

    X = clean_features(df)

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    result = pd.DataFrame({
        "prediction": preds,
        "label": np.where(preds == 1, "PortScan", "Benign"),
        "portscan_probability": probs
    })

    print(result)


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="command")

    p1 = sub.add_parser("train")
    p1.add_argument("--data", default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\datasets\Portscan.csv")
    p1.add_argument("--out", default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\models\PortScan_RF.pkl")

    p2 = sub.add_parser("eval")
    p2.add_argument("--model", default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\models\PortScan_RF.pkl")
    p2.add_argument("--data", default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\datasets\Portscan.csv")

    p3 = sub.add_parser("predict")
    p3.add_argument("--model", default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\models\PortScan_RF.pkl")
    p3.add_argument("--data", default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\datasets\Portscan.csv")

    args = parser.parse_args()

    if args.command == "train":
        train(args.data, args.out)

    elif args.command == "eval":
        evaluate_model(args.model, args.data)

    elif args.command == "predict":
        predict(args.model, args.data)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
