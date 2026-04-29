# integrated_ids.py
# -------------------------------------------------------------
# Unified Attack Detector using EXISTING trained models
#
# Keeps your current files:
#   model_1.pkl -> DDoS detector
#   model_2.pkl -> PortScan detector
#   model_3.pkl -> DoS detector
#
# Does NOT retrain anything.
# Does NOT discard existing work.
#
# It wraps all three models into ONE prediction system.
#
# Output:
#   Benign
#   DDoS
#   PortScan
#   DoS
#   Multiple_Attacks_Suspected
#
# -------------------------------------------------------------

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any


class IntegratedIDS:
    def __init__(
        self,
        model1_path="model_1.pkl",
        model2_path="model_2.pkl",
        model3_path="model_3.pkl",
        threshold_ddos=0.50,
        threshold_portscan=0.50,
        threshold_dos=0.50
    ):
        """
        model_1 = DDoS
        model_2 = PortScan
        model_3 = DoS
        """

        self.thresholds = {
            "DDoS": threshold_ddos,
            "PortScan": threshold_portscan,
            "DoS": threshold_dos
        }

        self.model_1 = self._load_model(model1_path)
        self.model_2 = self._load_model(model2_path)
        self.model_3 = self._load_model(model3_path)

    # ---------------------------------------------------------
    # MODEL LOADING
    # ---------------------------------------------------------
    def _load_model(self, path):
        obj = joblib.load(path)

        # Some files may save raw model
        if hasattr(obj, "predict_proba"):
            return {
                "model": obj,
                "scaler": None,
                "feature_names": None,
                "threshold": 0.50
            }

        # Some files may save artifact dict
        if isinstance(obj, dict):
            return {
                "model": obj.get("model"),
                "scaler": obj.get("scaler"),
                "feature_names": obj.get("feature_names"),
                "threshold": obj.get("threshold", 0.50)
            }

        raise ValueError(f"Unsupported model format: {path}")

    # ---------------------------------------------------------
    # DATA PREP
    # ---------------------------------------------------------
    def _prepare(self, df, artifact):
        X = df.copy()

        # keep numeric only
        X = X.select_dtypes(include=[np.number])

        # clean
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # align columns
        feats = artifact["feature_names"]
        if feats is not None:
            X = X.reindex(columns=feats, fill_value=0)

        # scale if scaler exists
        if artifact["scaler"] is not None:
            X = artifact["scaler"].transform(X)

        return X

    # ---------------------------------------------------------
    # SINGLE MODEL PROBABILITY
    # ---------------------------------------------------------
    def _score(self, artifact, df):
        X = self._prepare(df, artifact)

        model = artifact["model"]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1]
            return prob

        # fallback
        pred = model.predict(X)
        return pred.astype(float)

    # ---------------------------------------------------------
    # MAIN PREDICT
    # ---------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input = dataframe of flows
        Output = one unified decision
        """

        ddos_prob = self._score(self.model_1, df)
        port_prob = self._score(self.model_2, df)
        dos_prob = self._score(self.model_3, df)

        outputs = []

        for i in range(len(df)):
            scores = {
                "DDoS": float(ddos_prob[i]),
                "PortScan": float(port_prob[i]),
                "DoS": float(dos_prob[i]),
            }

            flags = {
                k: scores[k] >= self.thresholds[k]
                for k in scores
            }

            triggered = [k for k, v in flags.items() if v]

            # -------------------------------------------------
            # Decision Logic
            # -------------------------------------------------
            if len(triggered) == 0:
                label = "Benign"

            elif len(triggered) == 1:
                label = triggered[0]

            else:
                # Multiple models fired
                # choose strongest OR mark suspicious
                strongest = max(scores, key=scores.get)

                if scores[strongest] >= 0.80:
                    label = strongest
                else:
                    label = "Multiple_Attacks_Suspected"

            row = {
                "prediction": label,
                "ddos_probability": scores["DDoS"],
                "portscan_probability": scores["PortScan"],
                "dos_probability": scores["DoS"]
            }

            outputs.append(row)

        return pd.DataFrame(outputs)

    # ---------------------------------------------------------
    # PREDICT ONE FLOW
    # ---------------------------------------------------------
    def predict_single(self, flow_dict: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame([flow_dict])
        result = self.predict(df)
        return result.iloc[0].to_dict()


# -------------------------------------------------------------
# CLI EXAMPLE
# -------------------------------------------------------------
if __name__ == "__main__":

    # Example usage:
    # python integrated_ids.py
    #
    # Replace sample.csv with real traffic features

    ids = IntegratedIDS(
        model1_path="model_1.pkl",
        model2_path="model_2.pkl",
        model3_path="model_3.pkl"
    )

    try:
        sample = pd.read_csv("sample.csv")
        result = ids.predict(sample)
        print(result)

    except Exception as e:
        print("System loaded successfully.")
        print("Provide input dataframe or sample.csv")
        print("Error:", e)