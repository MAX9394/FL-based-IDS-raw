# multiclass_model_strong.py
# ------------------------------------------------------------
# Multiclass IDS Trainer (Federated-Ready, Strong Version)
# Logistic Regression Only (as requested)
# ------------------------------------------------------------

import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)

from imblearn.combine import SMOTETomek


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.2

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


# ------------------------------------------------------------
# LOAD + PREPROCESS
# ------------------------------------------------------------


def load_data(path):
    df = pd.read_csv(path)

    if "Label" not in df.columns:
        raise ValueError("Dataset must contain 'Label' column")

    # Drop columns
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns], errors="ignore")

    y = df["Label"].astype(int)
    X = df.drop(columns=["Label"])

    # Keep numeric only
    X = X.select_dtypes(include=[np.number])

    # FIRST CLEAN
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # CLIP NEGATIVE VALUES (CRITICAL)
    X = X.clip(lower=0)

    # LOG TRANSFORM
    X = np.log1p(X)

    # SECOND CLEAN (CRITICAL FIX)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

def train(data_path, out_path):

    print("[1/7] Loading data...")
    X, y = load_data(data_path)

    print("[2/7] Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("NaN:", np.isnan(X).sum().sum())
    print("Inf:", np.isinf(X).sum().sum())

    print("[3/7] Scaling...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[4/7] Skipping SMOTETomek (using class_weight instead)...")
    # print("[4/7] Balancing with SMOTETomek...")
    # sampler = SMOTETomek(random_state=RANDOM_STATE)
    # X_train, y_train = sampler.fit_resample(X_train, y_train)

    print("[5/7] Training Logistic Regression...")

    model = LogisticRegression(
        # multi_class="multinomial",
        solver="saga",
        max_iter=500,
        class_weight="balanced",
        # n_jobs=-1,
        verbose=0
    )

    model.fit(X_train, y_train)

    print("[6/7] Evaluating...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))

    try:
        print("ROC AUC  :", round(roc_auc_score(y_test, y_prob, multi_class="ovr"), 4))
    except:
        print("ROC AUC  : skipped")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --------------------------------------------------------
    # SAVE ARTIFACT (FEDERATED READY)
    # --------------------------------------------------------
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "metadata": {
            "num_classes": len(np.unique(y)),
            "classes": sorted(list(np.unique(y))),
            "model_type": "multiclass_logistic_regression"
        }
    }

    joblib.dump(artifact, out_path)

    print(f"\nSaved model → {out_path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # parser.add_argument("--data", required=True, help="Path to multiclass dataset CSV")
    parser.add_argument("--data", default=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Projects\FL-based-IDS-raw\Changes.max\datasets\multiclass_ids.csv")
    parser.add_argument("--out", default=r"models\FL_LogReg.pkl")

    args = parser.parse_args()

    train(args.data, args.out)
