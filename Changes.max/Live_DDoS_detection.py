import time
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from scapy.all import sniff, IP, TCP, UDP

MODEL_PATH = "DDoS_LogReg.pkl"
FLOW_TIMEOUT = 5          # seconds
PREDICT_EVERY = 2         # seconds

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
scaler = artifact["scaler"]
feature_names = artifact["feature_names"]

# -------------------------------------------------
# SAME DROPPED COLUMNS AS TRAINING
# -------------------------------------------------

DROP_COLUMNS = [
    # add any columns you removed during training
]

# -------------------------------------------------
# FLOW STORAGE
# -------------------------------------------------

flows = defaultdict(list)
last_prediction = time.time()


# -------------------------------------------------
# FLOW KEY
# -------------------------------------------------

def get_flow_key(pkt):
    proto = "OTHER"

    if TCP in pkt:
        proto = "TCP"
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport

    elif UDP in pkt:
        proto = "UDP"
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport

    else:
        sport = 0
        dport = 0

    return (
        pkt[IP].src,
        pkt[IP].dst,
        sport,
        dport,
        proto
    )


# -------------------------------------------------
# PACKET HANDLER
# -------------------------------------------------

def process_packet(pkt):
    global flows

    if IP not in pkt:
        return

    key = get_flow_key(pkt)

    packet_info = {
        "time": time.time(),
        "length": len(pkt),
        "flags": pkt[TCP].flags if TCP in pkt else 0
    }

    flows[key].append(packet_info)


# -------------------------------------------------
# FEATURE GENERATION
# -------------------------------------------------

def build_features(packets):
    times = [p["time"] for p in packets]
    lens = [p["length"] for p in packets]

    duration = max(times) - min(times) if len(times) > 1 else 0.0
    total_packets = len(packets)
    total_bytes = sum(lens)

    pps = total_packets / (duration + 1e-6)
    bps = total_bytes / (duration + 1e-6)

    mean_len = np.mean(lens)
    std_len = np.std(lens)

    iats = np.diff(times) if len(times) > 1 else [0]
    mean_iat = np.mean(iats)
    std_iat = np.std(iats)

    row = {
        "Flow Duration": duration,
        "Total Fwd Packets": total_packets,
        "Flow Bytes/s": bps,
        "Flow Packets/s": pps,
        "Packet Length Mean": mean_len,
        "Packet Length Std": std_len,
        "Flow IAT Mean": mean_iat,
        "Flow IAT Std": std_iat,
    }

    return row


# -------------------------------------------------
# PREDICTION LOOP
# -------------------------------------------------

def predict_flows():
    global flows

    rows = []

    expired = []

    now = time.time()

    for key, packets in flows.items():

        if now - packets[-1]["time"] >= FLOW_TIMEOUT:
            feat = build_features(packets)
            rows.append(feat)
            expired.append(key)

    if not rows:
        return

    df = pd.DataFrame(rows)

    for col in DROP_COLUMNS:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    X = scaler.transform(df)
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    for i in range(len(df)):
        if preds[i] == 1:
            print(
                f"[ALERT] Possible DDoS detected "
                f"(prob={probs[i]:.4f})"
            )
        else:
            print(
                f"[OK] Normal traffic "
                f"(prob={probs[i]:.4f})"
            )

    for k in expired:
        del flows[k]


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def monitor():
    global last_prediction

    print("Live DDoS Detector Running...")

    def wrapper(pkt):
        global last_prediction

        process_packet(pkt)

        if time.time() - last_prediction >= PREDICT_EVERY:
            predict_flows()
            last_prediction = time.time()

    sniff(prn=wrapper, store=False)


if __name__ == "__main__":
    monitor()