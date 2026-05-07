# live_portscan_final.py
# ------------------------------------------------------------
# FINAL Live PortScan Detector (PyShark Version)
#
# Aggregated fixes:
# ✔ Source-IP aggregation
# ✔ Multi-port scan logic
# ✔ Lower live threshold override
# ✔ Packet heartbeat
# ✔ Better shutdown handling
# ✔ Debug probabilities
# ✔ Works with model_2.pkl
#
# Requirements:
#   pip install pyshark pandas numpy joblib
#
# Run:
#   python live_portscan_final.py
#
# ------------------------------------------------------------

import time
import signal
import sys
import joblib
import numpy as np
import pandas as pd
import pyshark

from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "PortScan_LogReg.pkl"

INTERFACE_ID = "5"   # change if needed
TSHARK_PATH = r"C:\Users\Mayank Kumar Sagar\Desktop\MAX\Installs\Wireshark\tshark.exe"

FLOW_TIMEOUT = 5
MIN_PACKETS = 3

# Override aggressive offline threshold
LIVE_THRESHOLD = 0.30

VERBOSE_DEBUG = True
VERBOSE_BENIGN = False

HEARTBEAT_SECONDS = 2

# =====================================================
# LOAD MODEL
# =====================================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
scaler = artifact["scaler"]
feature_names = artifact["feature_names"]

saved_threshold = artifact.get("threshold", 0.5)
threshold = min(saved_threshold, LIVE_THRESHOLD)

print("[+] Model Loaded")
print("[+] Features:", len(feature_names))
print("[+] Saved Threshold :", round(saved_threshold, 4))
print("[+] Live Threshold  :", round(threshold, 4))

# =====================================================
# GLOBALS
# =====================================================

flows = defaultdict(list)
packet_counter = 0
running = True

# =====================================================
# HELPERS
# =====================================================

def now():
    return time.time()


def safe_float(x, d=0.0):
    try:
        return float(x)
    except:
        return d


def get_ports(pkt):
    try:
        if hasattr(pkt, "tcp"):
            return int(pkt.tcp.srcport), int(pkt.tcp.dstport)

        if hasattr(pkt, "udp"):
            return int(pkt.udp.srcport), int(pkt.udp.dstport)

    except:
        pass

    return 0, 0


# =====================================================
# FEATURE ENGINEERING
# =====================================================

def build_features(records):
    times = [r["time"] for r in records]
    sizes = [r["length"] for r in records]

    dports = set(r["dport"] for r in records)
    dstips = set(r["dst"] for r in records)

    duration = max(times) - min(times) if len(times) > 1 else 0.0
    duration = max(duration, 1e-6)

    total_packets = len(records)
    total_bytes = sum(sizes)

    pps = total_packets / duration
    bps = total_bytes / duration

    mean_len = np.mean(sizes)
    std_len = np.std(sizes)
    max_len = np.max(sizes)
    min_len = np.min(sizes)

    if len(times) > 1:
        iats = np.diff(times)
        mean_iat = np.mean(iats)
        max_iat = np.max(iats)
        min_iat = np.min(iats)
    else:
        mean_iat = max_iat = min_iat = 0.0

    syn = sum(r["syn"] for r in records)
    ack = sum(r["ack"] for r in records)
    rst = sum(r["rst"] for r in records)
    fin = sum(r["fin"] for r in records)

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
        "Flow IAT Max": max_iat,
        "Flow IAT Min": min_iat,

        "SYN Flag Count": syn,
        "ACK Flag Count": ack,
        "RST Flag Count": rst,
        "FIN Flag Count": fin,

        # critical PortScan indicators
        "Destination Port Count": len(dports),
        "Target Host Count": len(dstips),
    }

    return row


# =====================================================
# PREPROCESS
# =====================================================

def preprocess(row):
    df = pd.DataFrame([row])

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.clip(lower=0, upper=1e12)
    df = np.log1p(df)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    return df


# =====================================================
# PREDICT
# =====================================================

def predict(row):
    X = preprocess(row)
    Xs = scaler.transform(X)

    prob = float(model.predict_proba(Xs)[0][1])
    pred = int(prob >= threshold)

    return pred, prob


# =====================================================
# SCORE FLOWS
# =====================================================

def score_flows():
    dead = []
    t = now()

    for src, records in flows.items():

        if len(records) == 0:
            dead.append(src)
            continue

        idle = t - records[-1]["time"]

        if idle < FLOW_TIMEOUT:
            continue

        if len(records) < MIN_PACKETS:
            dead.append(src)
            continue

        row = build_features(records)
        pred, prob = predict(row)

        ports = row["Destination Port Count"]
        hosts = row["Target Host Count"]

        if VERBOSE_DEBUG:
            print(
                f"[DEBUG] {src} "
                f"prob={prob:.4f} "
                f"pkts={len(records)} "
                f"ports={ports} "
                f"hosts={hosts}"
            )

        if pred == 1:
            print("=" * 70)
            print("⚠ PORTSCAN DETECTED")
            print("Source IP   :", src)
            print("Probability :", round(prob, 4))
            print("Packets     :", len(records))
            print("Ports Hit   :", ports)
            print("Hosts Hit   :", hosts)
            print("Pkts/sec    :", round(row["Flow Packets/s"], 2))
            print("=" * 70)

        elif VERBOSE_BENIGN:
            print(f"[OK] {src} prob={prob:.4f}")

        dead.append(src)

    for k in dead:
        del flows[k]


# =====================================================
# PROCESS PACKETS
# =====================================================

def process_packet(pkt):
    global packet_counter

    try:
        if not hasattr(pkt, "ip"):
            return

        packet_counter += 1

        src = pkt.ip.src
        dst = pkt.ip.dst

        sport, dport = get_ports(pkt)

        syn = ack = rst = fin = 0

        if hasattr(pkt, "tcp"):
            flags = str(pkt.tcp.flags)

            if "0x0002" in flags or flags == "2":
                syn = 1

            if "0x0010" in flags or flags == "16":
                ack = 1

            if "0x0004" in flags or flags == "4":
                rst = 1

            if "0x0001" in flags or flags == "1":
                fin = 1

        flows[src].append({
            "time": now(),
            "dst": dst,
            "sport": sport,
            "dport": dport,
            "length": safe_float(pkt.length),
            "syn": syn,
            "ack": ack,
            "rst": rst,
            "fin": fin
        })

    except:
        pass


# =====================================================
# CLEAN EXIT
# =====================================================

def shutdown(sig=None, frame=None):
    global running
    running = False
    print("\n[+] Shutting down cleanly...")
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)


# =====================================================
# MAIN
# =====================================================

def main():
    global running

    print("[+] Starting Live PortScan Detector")
    print("[+] Interface:", INTERFACE_ID)
    print("[+] Waiting for traffic...\n")

    capture = pyshark.LiveCapture(
        interface=INTERFACE_ID,
        tshark_path=TSHARK_PATH
    )

    last_cleanup = time.time()
    last_heartbeat = time.time()

    try:
        for pkt in capture.sniff_continuously():

            if not running:
                break

            process_packet(pkt)

            now_t = time.time()

            if now_t - last_cleanup >= 1:
                score_flows()
                last_cleanup = now_t

            if now_t - last_heartbeat >= HEARTBEAT_SECONDS:
                print(
                    f"[HEARTBEAT] packets={packet_counter} "
                    f"tracked_sources={len(flows)}"
                )
                last_heartbeat = now_t

    except KeyboardInterrupt:
        shutdown()

    except Exception as e:
        print("[ERROR]", e)
        shutdown()


if __name__ == "__main__":
    main()