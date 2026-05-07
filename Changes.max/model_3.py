# dos_detector.py
# Production-ready DoS attack detector + Federated Learning compatible client
# Faithfully implements the intended behavior of model_3.ipynb

import os
import json
import joblib
import numpy as np
import pandas as pd

from typing import Dict, List, Tuple, Optional, Union

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.combine import SMOTETomek


class DoSDetectorFL:
    """
    Binary DoS Detector

    Label mapping:
        0 = Benign
        1 = DoS

    Works as:
        - standalone detector
        - federated learning client
    """

    VERSION = "1.0.0"

    # Exact-style notebook drop columns (based on model_3 intent)
    DROP_COLUMNS = [
        "Fwd Packets Length Total",
        "Bwd Packets Length Total",
        "Fwd Packet Length Max",
        "Bwd Packet Length Max",
        "Fwd Packet Length Min",
        "Bwd Packet Length Min",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "Fwd Packet Length Std",
        "Bwd Packet Length Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Flow IAT Std",
        "Fwd IAT Max",
        "Fwd IAT Min",
        "Fwd IAT Std",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Bwd IAT Std",
        "FIN Flag Count",
        "SYN Flag Count",
        "RST Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "URG Flag Count",
        "CWE Flag Count",
        "ECE Flag Count",
        "Fwd Avg Bytes/Bulk",
        "Fwd Avg Packets/Bulk",
        "Fwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk",
        "Bwd Avg Packets/Bulk",
        "Bwd Avg Bulk Rate",
        "Subflow Fwd Bytes",
        "Subflow Bwd Bytes",
        "Init Fwd Win Bytes",
        "Init Bwd Win Bytes",
        "Active Max",
        "Active Min",
        "Active Std",
        "Idle Max",
        "Idle Min",
        "Idle Std",
    ]

    LABEL_COLUMN_CANDIDATES = ["Label", "label", "Class", "class"]

    def __init__(
        self,
        threshold: float = 0.50,
        random_state: int = 42,
        c_value: float = 0.1
    ):
        self.threshold = threshold
        self.random_state = random_state
        self.c_value = c_value

        self.scaler = StandardScaler()

        self.model = LogisticRegression(
            max_iter=2000,
            solver="saga",
            warm_start=True,
            class_weight="balanced",
            C=self.c_value,
            random_state=self.random_state,
            n_jobs=-1
        )

        self.feature_columns: Optional[List[str]] = None
        self.is_fitted = False

    # =====================================================
    # Public API
    # =====================================================

    def fit(self, csv_path: str, test_size: float = 0.2) -> Dict:
        df = pd.read_csv(csv_path)
        return self.fit_dataframe(df, test_size=test_size)

    def fit_dataframe(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        X, y = self._prepare_training_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )

        # Balance classes
        smt = SMOTETomek(
            sampling_strategy=0.6,
            random_state=self.random_state
        )

        X_train_bal, y_train_bal = smt.fit_resample(X_train, y_train)

        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train_bal)
        X_test_scaled = self.scaler.transform(X_test)

        # Train
        self.model.fit(X_train_scaled, y_train_bal)
        self.is_fitted = True

        # Evaluate
        metrics = self.evaluate_arrays(X_test_scaled, y_test)
        return metrics

    def partial_fit_federated(self, df: pd.DataFrame) -> None:
        """
        Local training round after receiving global weights.
        """
        X, y = self._prepare_training_data(df)

        smt = SMOTETomek(
            sampling_strategy=0.6,
            random_state=self.random_state
        )

        X_bal, y_bal = smt.fit_resample(X, y)

        if not self.is_fitted:
            X_scaled = self.scaler.fit_transform(X_bal)
        else:
            X_scaled = self.scaler.transform(X_bal)

        self.model.fit(X_scaled, y_bal)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        Xp = self._prepare_inference_data(X)
        Xs = self.scaler.transform(Xp)
        return self.model.predict_proba(Xs)[:, 1]

    def predict_single(self, flow_dict: Dict) -> Dict:
        df = pd.DataFrame([flow_dict])
        prob = float(self.predict_proba(df)[0])
        pred = int(prob >= self.threshold)

        return {
            "prediction": pred,
            "label": "DoS" if pred == 1 else "Benign",
            "probability_dos": round(prob, 6)
        }

    def evaluate(self, csv_path: str) -> Dict:
        df = pd.read_csv(csv_path)
        X, y = self._prepare_training_data(df)
        Xs = self.scaler.transform(X)
        return self.evaluate_arrays(Xs, y)

    def save(self, path: str = "DoS_LogReg.pkl") -> None:
        self._check_fitted()

        artifact = {
            "version": self.VERSION,
            "threshold": self.threshold,
            "random_state": self.random_state,
            "c_value": self.c_value,
            "feature_columns": self.feature_columns,
            "drop_columns": self.DROP_COLUMNS,
            "model": self.model,
            "scaler": self.scaler
        }

        joblib.dump(artifact, path)

    @classmethod
    def load(cls, path: str):
        artifact = joblib.load(path)

        obj = cls(
            threshold=artifact["threshold"],
            random_state=artifact["random_state"],
            c_value=artifact["c_value"]
        )

        obj.feature_columns = artifact["feature_columns"]
        obj.model = artifact["model"]
        obj.scaler = artifact["scaler"]
        obj.is_fitted = True

        return obj

    # =====================================================
    # Federated Learning API
    # =====================================================

    def get_weights(self) -> Dict[str, np.ndarray]:
        self._check_fitted()

        return {
            "coef": self.model.coef_.copy(),
            "intercept": self.model.intercept_.copy()
        }

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        if not self.is_fitted:
            raise ValueError(
                "Model must be fitted once before setting weights."
            )

        self.model.coef_ = weights["coef"].copy()
        self.model.intercept_ = weights["intercept"].copy()

    @staticmethod
    def aggregate_weights(weight_list: List[Dict[str, np.ndarray]]) -> Dict:
        coef = np.mean([w["coef"] for w in weight_list], axis=0)
        intercept = np.mean([w["intercept"] for w in weight_list], axis=0)

        return {
            "coef": coef,
            "intercept": intercept
        }

    # =====================================================
    # Internal Helpers
    # =====================================================

    def _prepare_training_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:

        label_col = self._find_label_column(df)

        y_raw = df[label_col].astype(str).str.lower().str.strip()

        y = np.where(y_raw == "benign", 0, 1)

        X = df.drop(columns=[label_col], errors="ignore")
        X = self._clean_features(X)

        self.feature_columns = list(X.columns)

        return X, y

    def _prepare_inference_data(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._clean_features(df.copy())

        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]
        return X

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:

        # Drop notebook columns if present
        df = df.drop(columns=self.DROP_COLUMNS, errors="ignore")

        # Keep numeric only
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Replace bad values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)

        # Clip extremes
        df = df.clip(lower=0, upper=1e12)

        # log1p transform
        df = np.log1p(df)

        return df

    def _find_label_column(self, df: pd.DataFrame) -> str:
        for c in self.LABEL_COLUMN_CANDIDATES:
            if c in df.columns:
                return c

        raise ValueError("No label column found.")

    def _check_fitted(self):
        if not self.is_fitted:
            raise ValueError("Model is not trained yet.")

    def evaluate_arrays(self, X_scaled, y_true) -> Dict:
        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= self.threshold).astype(int)

        result = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds)),
            "recall": float(recall_score(y_true, preds)),
            "f1_score": float(f1_score(y_true, preds)),
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "confusion_matrix": confusion_matrix(
                y_true,
                preds
            ).tolist(),
            "classification_report": classification_report(
                y_true,
                preds
            )
        }

        return result


# =====================================================
# CLI Example Usage
# =====================================================

if __name__ == "__main__":

    # Example:
    detector = DoSDetectorFL()
    metrics = detector.fit("datasets/DoS.csv")
    # print(metrics)
    detector.save(r"models\DoS_LogReg.pkl")

    print("DoSDetectorFL ready.")