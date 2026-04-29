import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# --- IMPORTANT: We must define the class so joblib can unpickle it ---
class XGBoostFederatedEnsemble:
    def __init__(self, models):
        self.models = models
    def predict(self, X):
        preds = np.array([model.predict(X) for model in self.models])
        final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=preds)
        return final_preds
    def predict_proba(self, X):
        probs = np.array([model.predict_proba(X) for model in self.models])
        return np.mean(probs, axis=0)

# --- STEP 1: Load the Federated Ensemble ---
try:
    global_xgb = joblib.load('global_federated_xgb_model.pkl')
    print("Successfully loaded Federated XGBoost Ensemble.")
except Exception as e:
    print(f"Error: Could not load the model. Ensure the pkl exists! {e}")
    exit()

# --- STEP 2: Load Test Data (DoS Wednesday) ---
test_path = "C:\\Datasets\\IDS_Dataset\\DoS-Wednesday-no-metadata.csv"
df = pd.read_csv(test_path)

# Apply the standard 49-column drop
cols_to_drop = [
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
df = df.drop(columns=cols_to_drop, axis=1, errors='ignore')
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

# Binary Labeling
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)
X_test = df.drop('Label', axis=1).values
y_test = df['Label'].values

# --- STEP 3: Evaluation ---
print("Running Federated XGBoost Committee Vote...")
y_pred = global_xgb.predict(X_test)

print("\n--- FEDERATED XGBOOST REPORT ---")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))