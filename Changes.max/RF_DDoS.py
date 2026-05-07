
"""
raw_ddos_model.py
=================

Production-grade raw DDoS detection model.

Purpose
-------
This script trains and serves a standalone Random Forest model for
flow-based DDoS detection.

This file intentionally contains:
    - NO packet capture
    - NO Scapy logic
    - NO networking code
    - NO threading
    - NO live traffic handling

It is ONLY the ML layer.

Architecture
------------
Dataset
    -> sanitation
    -> feature engineering
    -> preprocessing
    -> training
    -> threshold optimization
    -> packaged artifact

Designed For
------------
- CICIDS-style flow datasets
- Binary DDoS detection
- Live detector integration
- Offline training / online inference

Usage
-----

Train:
    python raw_ddos_model.py train --data ddos.csv --out ddos_rf.pkl

Evaluate:
    python raw_ddos_model.py eval --model ddos_rf.pkl --data test.csv

Predict:
    python raw_ddos_model.py predict --model ddos_rf.pkl --data sample.csv
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

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
    "class",
    "Attack",
    "Target",
]

# Conservative reduced feature philosophy
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

MODEL_CONFIG = dict(
    n_estimators=300,
    max_depth=14,
    min_samples_leaf=3,
    class_weight="balanced_subsample",
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)


# =========================================================
# HELPERS
# =========================================================

def detect_label_column(df: pd.DataFrame) -> str:
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not detect label column."
    )


def normalize_label(value) -> int:
    """
    Binary encoding:

        0 = Benign
        1 = DDoS/Attack
    """

    s = str(value).strip().lower()

    benign_tokens = {
        "benign",
        "normal",
        "0",
        "legitimate",
    }

    return 0 if s in benign_tokens else 1


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggressive sanitation for noisy IDS datasets.
    """

    # Keep numeric only
    df = df.select_dtypes(include=[np.number])

    # Force numeric conversion
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Replace invalids
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill missing
    df = df.fillna(0)

    # Clip extreme explosions
    df = df.clip(lower=0, upper=1e12)

    # Log compression
    df = np.log1p(df)

    return df


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    label_col = detect_label_column(df)

    y = df[label_col].apply(normalize_label).astype(int)

    X = df.drop(columns=[label_col], errors="ignore")

    # Drop optional metadata columns
    X = X.drop(columns=DROP_COLUMNS, errors="ignore")

    # Sanitize
    X = sanitize_dataframe(X)

    return X, y


def optimize_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probs
    )

    if len(thresholds) == 0:
        return 0.5

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
    ) / (
        precision[:-1] + recall[:-1] + 1e-12
    )

    best_idx = np.argmax(f1_scores)

    return float(thresholds[best_idx])


def evaluate_predictions(y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, preds)
        ),
        "precision": float(
            precision_score(
                y_true,
                preds,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                preds,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                preds,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probs)
        ),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(
            y_true,
            preds
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            preds,
            output_dict=True,
            zero_division=0
        )
    }

    return metrics


# =========================================================
# MODEL CLASS
# =========================================================

class RawDDoSModel:

    VERSION = "1.0.0"

    def __init__(self):
        self.model = RandomForestClassifier(
            **MODEL_CONFIG
        )

        self.feature_names = None
        self.threshold = 0.5
        self.is_trained = False

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    def train(self, csv_path):

        print("[1/7] Loading dataset...")
        X, y = load_dataset(csv_path)

        self.feature_names = list(X.columns)

        print("[2/7] Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X.values,
            y.values,
            test_size=TEST_SIZE,
            stratify=y.values,
            random_state=RANDOM_STATE
        )

        # print("[3/7] Balancing classes...")
        # sampler = SMOTETomek(
        #     sampling_strategy=0.5,
        #     random_state=RANDOM_STATE
        # )

        # X_train_bal, y_train_bal = sampler.fit_resample(
        #     X_train,
        #     y_train
        # )

        print("[3/7] Skipping SMOTETomek...")
        X_train_bal, y_train_bal = X_train, y_train

        print("[4/7] Training Random Forest...")
        self.model.fit(
            X_train_bal,
            y_train_bal
        )

        self.is_trained = True

        print("[5/7] Optimizing threshold...")
        probs = self.model.predict_proba(X_test)[:, 1]

        self.threshold = optimize_threshold(
            y_test,
            probs
        )

        print("[6/7] Evaluating...")
        metrics = evaluate_predictions(
            y_test,
            probs,
            self.threshold
        )

        print("[7/7] Training complete.\n")

        self.print_metrics(metrics)

        return metrics

    # -----------------------------------------------------
    # INFERENCE
    # -----------------------------------------------------

    def prepare_inference(self, df: pd.DataFrame):

        df = df.drop(
            columns=DROP_COLUMNS,
            errors="ignore"
        )

        X = sanitize_dataframe(df)

        # Add missing columns
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0

        # Preserve order
        X = X[self.feature_names]

        return X.values

    def predict_proba(self, df: pd.DataFrame):

        self._check_ready()

        X = self.prepare_inference(df)

        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame):

        probs = self.predict_proba(df)

        return (probs >= self.threshold).astype(int)

    def predict_single(self, flow_dict: dict):

        df = pd.DataFrame([flow_dict])

        prob = float(
            self.predict_proba(df)[0]
        )

        pred = int(prob >= self.threshold)

        return {
            "prediction": pred,
            "label": "DDoS" if pred == 1 else "Benign",
            "probability": round(prob, 6),
            "threshold": round(self.threshold, 6),
        }

    # -----------------------------------------------------
    # SAVE / LOAD
    # -----------------------------------------------------

    def save(self, path):

        self._check_ready()

        artifact = {
            "version": self.VERSION,
            "model": self.model,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "drop_columns": DROP_COLUMNS,
            "model_config": MODEL_CONFIG,
        }

        joblib.dump(artifact, path)

        print(f"Saved model artifact -> {path}")

    @classmethod
    def load(cls, path):

        artifact = joblib.load(path)

        obj = cls()

        obj.model = artifact["model"]
        obj.feature_names = artifact["feature_names"]
        obj.threshold = artifact["threshold"]
        obj.is_trained = True

        return obj

    # -----------------------------------------------------
    # UTILITIES
    # -----------------------------------------------------

    @staticmethod
    def print_metrics(metrics):

        print("-" * 60)
        print(f"Accuracy   : {metrics['accuracy']:.4f}")
        print(f"Precision  : {metrics['precision']:.4f}")
        print(f"Recall     : {metrics['recall']:.4f}")
        print(f"F1 Score   : {metrics['f1_score']:.4f}")
        print(f"ROC AUC    : {metrics['roc_auc']:.4f}")
        print(f"Threshold  : {metrics['threshold']:.4f}")
        print("-" * 60)

    def _check_ready(self):

        if not self.is_trained:
            raise ValueError(
                "Model is not trained or loaded."
            )


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="command")

    # TRAIN
    p1 = sub.add_parser("train")
    p1.add_argument("--data", required=False, default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\datasets\DDoS.csv")
    p1.add_argument("--out", required=False, default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\models\DDoS_RF.pkl")

    # EVAL
    p2 = sub.add_parser("eval")
    p2.add_argument("--model", required=False, default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\models\DDoS_RF.pkl")
    p2.add_argument("--data", required=False, default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\datasets\DDoS.csv")

    # PREDICT
    p3 = sub.add_parser("predict")
    p3.add_argument("--model", required=False, default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\models\DDoS_RF.pkl")
    p3.add_argument("--data", required=False, default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\datasets\DDoS.csv")

    args = parser.parse_args()

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    if args.command == "train":

        model = RawDDoSModel()

        metrics = model.train(args.data)

        model.save(args.out)

        metrics_path = Path(args.out).with_suffix(".metrics.json")

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Saved metrics -> {metrics_path}")

    # -----------------------------------------------------
    # EVAL
    # -----------------------------------------------------

    elif args.command == "eval":

        model = RawDDoSModel.load(args.model)

        df = pd.read_csv(args.data)

        label_col = detect_label_column(df)

        y = df[label_col].apply(normalize_label).astype(int)

        X_raw = df.drop(columns=[label_col], errors="ignore")

        probs = model.predict_proba(X_raw)

        metrics = evaluate_predictions(
            y.values,
            probs,
            model.threshold
        )

        model.print_metrics(metrics)

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    elif args.command == "predict":

        model = RawDDoSModel.load(args.model)

        df = pd.read_csv(args.data)

        probs = model.predict_proba(df)

        preds = model.predict(df)

        result = pd.DataFrame({
            "prediction": preds,
            "probability": probs
        })

        print(result)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
