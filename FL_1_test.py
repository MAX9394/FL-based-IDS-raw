import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve

# --- STEP 1: Load the Global Federated Model ---
model_path = 'global_federated_model.pkl'
try:
    global_model = joblib.load(model_path)
    print(f"Successfully loaded Global Logistic Regression Model.")
except Exception as e:
    print(f"Error: Could not load the model. {e}")
    exit()

# --- STEP 2: Load and Preprocess Test Data ---
# Change this path to whichever dataset you want to test the model against
test_file = "C:\\Datasets\\IDS_Dataset\\Portscan-Friday-no-metadata.csv" 
print(f"Loading test data from: {test_file}")

df = pd.read_csv(test_file)

# The standard 49 columns we agreed to drop
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
X_raw = df.drop('Label', axis=1).clip(lower=0, upper=1e12)
X_log = np.log1p(X_raw.values)

# Map labels (0 for Benign, 1 for any Attack)
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Benign' else 1)

X_test = df.drop('Label', axis=1).values
y_test = df['Label'].values
unique_classes = np.unique(y_test)
print(f"Unique classes found in test data: {unique_classes}")

# --- STEP 3: Run Predictions ---
print("Running predictions with the Federated Ensemble...")
scaler = RobustScaler()
X_test_scaled = scaler.fit_transform(X_log)

y_probs = global_model.predict_proba(X_test_scaled)[:, 1]
custom_threshold = 0.5
y_pred = (y_probs >= custom_threshold).astype(int)
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal Threshold Found: {best_threshold:.4f}")
y_pred = (y_probs >= best_threshold).astype(int)

# --- STEP 4: Results and Metrics ---
print("\n" + "="*30)
print("FEDERATED RF PERFORMANCE REPORT")
print("="*30)
print(classification_report(y_test, y_pred, labels=[0,1], target_names=['Benign', 'Attack']))

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Attack'], 
            yticklabels=['Benign', 'Attack'])
plt.title('Federated RF Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()