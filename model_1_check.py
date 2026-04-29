import joblib
import numpy as np
import pandas as pd
from scapy.all import sniff, IP, TCP
from sklearn.preprocessing import StandardScaler

# 1. Load the specific Model 1 specialist
try:
    model_1 = joblib.load('model_1.pkl')
    print("Successfully loaded DDoS Specialist (Model 1).")
except:
    print("Error: model_1.pkl not found.")
    exit()

# 2. Define the exact feature list used in model_1.ipynb
FEATURES = [
    "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Bwd IAT Mean", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s",
    "Bwd Packets/s", "Packet Length Min", "Packet Length Max", "Packet Length Mean",
    "Packet Length Std", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "ACK Flag Count", "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets"
]

def process_packet(packet):
    if IP in packet:
        try:
            # Basic behavioral feature extraction
            proto = packet.proto
            pkt_len = len(packet)
            
            # Extract TCP Flags (Critical for DDoS/DoS)
            syn = 1 if (TCP in packet and packet[TCP].flags == 'S') else 0
            fin = 1 if (TCP in packet and packet[TCP].flags == 'F') else 0
            
            # Construct feature vector
            # Most flow stats (Duration/IAT) require multiple packets, 
            # so we use 0 as a baseline for single-packet live testing.
            vector = np.zeros(len(FEATURES))
            vector[0] = proto          # Protocol
            vector[4] = pkt_len        # Fwd Packet Length Mean
            vector[23] = syn           # SYN Flag Count
            vector[22] = fin           # FIN Flag Count
            
            # --- Preprocessing (Must match model_1.ipynb) ---
            # 1. Handle potential large values
            vector = np.clip(vector, 0, 1e12)
            
            # 2. Log Transform
            vector_log = np.log1p(vector.reshape(1, -1))
            
            # 3. Predict
            # Model 1 uses Logistic Regression
            prediction = model_1.predict(vector_log)[0]
            probability = model_1.predict_proba(vector_log)[0][1]
            
            if prediction == 1:
                print(f"[!] DDoS ALERT: Malicious Traffic Detected! Confidence: {probability:.2%}")
            else:
                print(f"[*] Benign Traffic. Confidence: {1-probability:.2%}")
                
        except Exception as e:
            pass

print("[*] Monitoring Live DDoS Traffic... (Target: attack.py)")
sniff(filter="ip", prn=process_packet, store=0)