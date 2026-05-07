# Changes.max Folder Guide

This folder is a **model-development and deployment workspace** for the IDS project.  
It contains:

1. **Training scripts** for three attack families (DDoS, DoS, PortScan)
2. **Algorithm variants** (Logistic Regression, Random Forest, XGBoost)
3. **Live detection scripts** that consume saved model artifacts
4. **Saved `.pkl` model artifacts** used for inference

In short: this folder is where the project evolved from notebook ideas into production-style Python scripts for training and live traffic detection.

---

## High-Level Structure

- `model_1.py`, `model_2.py`, `model_3.py`  
  Logistic-regression based detectors (federated-learning friendly).
- `RF_*.py`, `XGB_*.py`  
  Alternative model families for the same attack tasks.
- `live_*.py`, `Live_DDoS_detection.py`  
  Real-time traffic capture + inference scripts.
- `FL_model_1.py`  
  Multiclass/federated-ready logistic trainer.
- `models/` and root-level `.pkl` files  
  Serialized trained artifacts used by live scripts or evaluation.

---

## File-by-file intent

### 1) Core Logistic Regression Pipelines

#### `model_1.py`
**Role:** DDoS client model for federated workflows (binary benign vs attack).  
**Intent:** Provide a corrected standalone script replacing notebook-only training with a reproducible CLI pipeline.

What it does:
- Detects and normalizes labels into binary classes
- Drops noisy/redundant CICIDS columns
- Cleans inf/NaN values, keeps numeric features
- Trains Logistic Regression (with optional SMOTE balancing)
- Supports train/eval/predict/export/import-global subcommands
- Exposes and consumes weights for FedAvg-style aggregation

Use this when you want a **federated-client compatible Logistic model** for DDoS.

---

#### `model_2.py`
**Role:** PortScan logistic trainer (standalone + federated-ready export).  
**Intent:** Convert `model_2.ipynb` logic into a robust script with threshold tuning and packaged outputs.

What it does:
- Auto-detects label column names
- Filters to Benign + PortScan classes only
- Cleans and log-transforms feature space
- Applies `SMOTETomek` for class balancing
- Trains Logistic Regression and optimizes decision threshold using PR/F1
- Saves:
  - `model_2.pkl` (artifact)
  - `model_2_metrics.json`
  - `model_2_federated.npz` (coef/intercept payload)

Use this when you need a **strong PortScan LR baseline + threshold-aware deployment artifact**.

---

#### `model_3.py`
**Role:** DoS detector class (`DoSDetectorFL`) designed for both standalone and federated use.  
**Intent:** Provide a reusable Python class API (not just a one-off script) for DoS training and inference.

What it does:
- Encapsulates preprocessing, training, and inference in one class
- Supports full fit, federated local retraining, save/load, weight get/set, and aggregation helper
- Uses `SMOTETomek`, scaling, clipping, and log transforms
- Returns rich metrics (accuracy/precision/recall/F1/ROC-AUC/confusion data)

Use this when you need a **programmatic DoS model object** that can be embedded in larger FL orchestration.

---

### 2) Federated/Multiclass Training Utility

#### `FL_model_1.py`
**Role:** Multiclass logistic trainer marked as federated-ready artifact builder.  
**Intent:** Train one multiclass Logistic Regression model with stable preprocessing and export model/scaler/features metadata.

What it does:
- Loads CSV with required `Label`
- Drops known noisy columns
- Applies strict cleaning (`inf/NaN` handling, clipping, `log1p`)
- Trains multiclass Logistic Regression with class balancing
- Saves a packaged artifact for downstream federated usage

Use this when you want a **single multiclass baseline** rather than separate binary detectors.

---

### 3) DDoS Alternatives

#### `RF_DDoS.py`
**Role:** Random Forest DDoS detector (`RawDDoSModel`).  
**Intent:** Offer non-linear tree-based alternative to logistic DDoS model.

What it does:
- Binary benign/attack encoding
- CICIDS-style feature cleanup and log compression
- RF training with class-weight balancing
- Threshold optimization from PR curve
- Save/load + eval/predict CLI paths

Use this when DDoS decision boundaries are too complex for linear models.

---

#### `XGB_DDoS.py`
**Role:** XGBoost DDoS detector (`RawDDoSXGBoostModel`).  
**Intent:** Improve DDoS modeling with boosted trees and explicit imbalance handling (`scale_pos_weight`).

What it does:
- Same robust sanitation pipeline as RF path
- Computes class imbalance ratio dynamically
- Trains `XGBClassifier` (hist tree method)
- Optimizes threshold and exposes feature-importance helper
- Saves complete inference artifact

Use this when you want **higher-capacity boosted modeling** for DDoS.

---

### 4) DoS Alternatives

#### `RF_DoS.py`
**Role:** Random Forest DoS detector (`RawDoSModel`).  
**Intent:** Tree-based DoS classifier with stable preprocessing and threshold-aware inference.

What it does:
- Sanitizes features and maps labels to benign/DoS
- Trains RF with balanced subsampling
- Calculates PR-AUC in addition to standard metrics
- Stores model + threshold + metadata in serialized artifact

Use this for **interpretable and robust non-linear DoS detection**.

---

#### `XGB_DoS.py`
**Role:** XGBoost DoS detector (`XGBDoSModel`).  
**Intent:** CPU-optimized boosted-tree DoS model for stronger non-linear detection.

What it does:
- Performs same feature cleaning/log transform strategy
- Dynamically computes `scale_pos_weight`
- Trains with histogram-based XGBoost settings
- Saves deployable artifacts and evaluation metrics

Use this for **high-performance DoS detection** under class imbalance.

---

### 5) PortScan Alternatives

#### `RF_PortScan.py`
**Role:** Random Forest PortScan classifier with calibration.  
**Intent:** Improve PortScan probability quality using calibrated RF outputs.

What it does:
- Filters to Benign/PortScan only
- Cleans and validates numeric features
- Trains RF wrapped in `CalibratedClassifierCV` (isotonic)
- Optimizes threshold and exports artifact + metrics JSON

Use this when you need **better-calibrated PortScan probabilities**.

---

#### `XGB_PortScan.py`
**Role:** XGBoost PortScan classifier.  
**Intent:** Boosted-tree variant for PortScan with dynamic imbalance scaling.

What it does:
- Similar data preparation path as RF PortScan
- Computes `scale_pos_weight`
- Trains CPU-histogram XGBoost model
- Exports thresholded artifact and metrics

Use this for **higher-capacity PortScan classification** than RF/LR baselines.

---

### 6) Live Detection Scripts (Runtime Monitoring)

#### `live_ddos.py`
**Role:** Scapy-based live DDoS monitor.  
**Intent:** Capture packets in real time, aggregate into flows, and apply logistic DDoS model inference periodically.

Highlights:
- Uses `scapy.sniff`
- Builds lightweight flow statistics (rate, packet lengths, IATs)
- Aligns runtime features to trained model feature list
- Prints `[ALERT]` when DDoS probability is high

---

#### `Live_DDoS_detection.py`
**Role:** PyShark/TShark live DDoS detector (Windows-oriented).  
**Intent:** Alternative live DDoS runtime that uses PyShark capture and richer per-flow tracking.

Highlights:
- Captures from selected interface ID
- Tracks TCP flags, packet stats, and idle flow timeout
- Scores completed flows with loaded model/scaler
- Emits clear alert lines with source/destination/probability

---

#### `live_dos.py`
**Role:** Live DoS detection using `DoSDetectorFL` from `model_3.py`.  
**Intent:** Reuse class-based DoS model API for real-time flow scoring.

Highlights:
- Maintains flow tables and derives CICIDS-like aggregate features
- Applies stricter runtime threshold than training default
- Uses threads for packet capture and flow processing loops

---

#### `live_portscan.py`
**Role:** Final PyShark live PortScan detector.  
**Intent:** Source-IP-centric scanning detection with practical runtime heuristics.

Highlights:
- Aggregates records by source IP
- Adds scan-centric features (`Destination Port Count`, `Target Host Count`)
- Uses heartbeat logs for runtime visibility
- Supports graceful shutdown and debug probability output

---

### 7) Serialized Artifacts

#### Root-level artifact
- `DoS_LogReg.pkl`  
  Saved DoS logistic artifact (includes model/scaler/features and metadata).

#### `models/` folder artifacts
- `DDoS_LogReg.pkl`
- `DDoS_RF.pkl`
- `DDoS_XGB.pkl`
- `DoS_LogReg.pkl`
- `DoS_RF.pkl`
- `DoS_XGB.pkl`
- `PortScan_LogReg.pkl`
- `PortScan_RF.pkl`
- `PortScan_XGB.pkl`

**Intent of these files:** pre-trained model checkpoints for direct inference/testing so live scripts can run without retraining every launch.

---

## How to read this folder quickly

If you are new to `Changes.max`, follow this order:
1. `model_1.py`, `model_2.py`, `model_3.py` (main LR baselines and FL logic)
2. One algorithm pair for comparison (`RF_*` vs `XGB_*`)
3. `live_*` scripts for deployment/runtime behavior
4. `models/` artifacts for expected runtime inputs

This gives the full picture: **training pipeline -> artifact packaging -> live detection integration**.
