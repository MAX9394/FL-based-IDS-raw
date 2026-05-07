# live_cicflow_detector.py
# Windows-ready Live DDoS Detector
# Uses TShark/PyShark + trained .pkl model

import time
import math
import joblib
import numpy as np
import pandas as pd
import pyshark

from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "DDoS_LogReg.pkl"
INTERFACE_ID = "5"              # Your Wi-Fi adapter
FLOW_TIMEOUT = 5               # seconds idle before scoring
MIN_PACKETS_PER_FLOW = 5
PREDICT_THRESHOLD = 0.50
VERBOSE_NORMAL = False         # set True to print benign flows

# =====================================================
# LOAD MODEL
# =====================================================

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
scaler = artifact["scaler"]
feature_names = artifact["feature_names"]

print("[+] Model Loaded")
print("[+] Features Expected:", len(feature_names))

# =====================================================
# FLOW STORAGE
# =====================================================

flows = defaultdict(list)


# =====================================================
# HELPERS
# =====================================================

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default


def now():
    return time.time()


def get_proto(pkt):
    if hasattr(pkt, "tcp"):
        return "TCP"
    elif hasattr(pkt, "udp"):
        return "UDP"
    return "OTHER"


def get_ports(pkt):
    try:
        if hasattr(pkt, "tcp"):
            return int(pkt.tcp.srcport), int(pkt.tcp.dstport)
        elif hasattr(pkt, "udp"):
            return int(pkt.udp.srcport), int(pkt.udp.dstport)
    except:
        pass
    return 0, 0


def flow_key(pkt):
    src = pkt.ip.src
    dst = pkt.ip.dst
    sport, dport = get_ports(pkt)
    proto = get_proto(pkt)

    return (src, dst, sport, dport, proto)


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def build_flow_features(records):
    times = [r["time"] for r in records]
    sizes = [r["length"] for r in records]

    duration = max(times) - min(times) if len(times) > 1 else 0.0
    total_packets = len(records)
    total_bytes = sum(sizes)

    pps = total_packets / (duration + 1e-6)
    bps = total_bytes / (duration + 1e-6)

    mean_len = np.mean(sizes)
    std_len = np.std(sizes)
    max_len = np.max(sizes)
    min_len = np.min(sizes)

    if len(times) > 1:
        iats = np.diff(times)
        mean_iat = np.mean(iats)
        std_iat = np.std(iats)
        max_iat = np.max(iats)
        min_iat = np.min(iats)
    else:
        mean_iat = std_iat = max_iat = min_iat = 0.0

    syn = sum(r["syn"] for r in records)
    ack = sum(r["ack"] for r in records)
    rst = sum(r["rst"] for r in records)

    row = {
        "Flow Duration": duration,
        "Total Fwd Packets": total_packets,
        "Total Backward Packets": 0,
        "Flow Bytes/s": bps,
        "Flow Packets/s": pps,

        "Packet Length Mean": mean_len,
        "Packet Length Std": std_len,
        "Packet Length Max": max_len,
        "Packet Length Min": min_len,

        "Flow IAT Mean": mean_iat,
        "Flow IAT Std": std_iat,
        "Flow IAT Max": max_iat,
        "Flow IAT Min": min_iat,

        "SYN Flag Count": syn,
        "ACK Flag Count": ack,
        "RST Flag Count": rst,
    }

    return row


# =====================================================
# MODEL INFERENCE
# =====================================================

def predict_row(row):
    df = pd.DataFrame([row])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    X = scaler.transform(df)
    prob = float(model.predict_proba(X)[0][1])
    pred = int(prob >= PREDICT_THRESHOLD)

    return pred, prob


# =====================================================
# FLOW SCORING
# =====================================================

def score_expired_flows():
    dead = []
    t = now()

    for key, records in flows.items():

        if len(records) < MIN_PACKETS_PER_FLOW:
            if t - records[-1]["time"] >= FLOW_TIMEOUT:
                dead.append(key)
            continue

        if t - records[-1]["time"] >= FLOW_TIMEOUT:

            row = build_flow_features(records)
            pred, prob = predict_row(row)

            src, dst, sport, dport, proto = key

            if pred == 1:
                print(
                    f"[ALERT] DDoS suspected | "
                    f"{src}:{sport} -> {dst}:{dport} | "
                    f"{proto} | prob={prob:.4f} | pkts={len(records)}"
                )
            else:
                if VERBOSE_NORMAL:
                    print(
                        f"[OK] {src}:{sport} -> {dst}:{dport} "
                        f"| prob={prob:.4f}"
                    )

            dead.append(key)

    for k in dead:
        del flows[k]


# =====================================================
# LIVE PACKET LOOP
# =====================================================

def process_packet(pkt):
    try:
        if not hasattr(pkt, "ip"):
            return

        key = flow_key(pkt)

        length = safe_float(pkt.length, 0)

        syn = ack = rst = 0

        if hasattr(pkt, "tcp"):
            flags = str(pkt.tcp.flags)

            # tshark flags style
            if "0x0002" in flags or flags == "2":
                syn = 1
            if "0x0010" in flags or flags == "16":
                ack = 1
            if "0x0004" in flags or flags == "4":
                rst = 1

        flows[key].append({
            "time": now(),
            "length": length,
            "syn": syn,
            "ack": ack,
            "rst": rst
        })

    except:
        pass


# =====================================================
# MAIN
# =====================================================

def main():
    print("[+] Starting live capture...")
    print("[+] Interface:", INTERFACE_ID)
    print("[+] Waiting for traffic...\n")

    capture = pyshark.LiveCapture(
        interface=INTERFACE_ID,
        tshark_path=r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Installs\Wireshark\tshark.exe"
    )

    last_cleanup = time.time()

    for pkt in capture.sniff_continuously():

        process_packet(pkt)

        if time.time() - last_cleanup >= 1:
            score_expired_flows()
            last_cleanup = time.time()


if __name__ == "__main__":
    main()