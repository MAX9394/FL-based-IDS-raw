import pyshark
import time
import threading
from collections import defaultdict
import pandas as pd

from model_3 import DoSDetectorFL


# ============================================
# CONFIG
# ============================================

INTERFACE = "Wi-Fi"   # Change if needed
FLOW_TIMEOUT = 5      # seconds
ALERT_THRESHOLD = 0.7 # stricter than training

MODEL_PATH = r"\models\DoS_LogReg.pkl"


# ============================================
# LOAD MODEL
# ============================================

detector = DoSDetectorFL.load(MODEL_PATH)
detector.threshold = ALERT_THRESHOLD


# ============================================
# FLOW STORAGE
# ============================================

flows = defaultdict(lambda: {
    "start_time": None,
    "last_time": None,
    "fwd_packets": 0,
    "bwd_packets": 0,
    "fwd_bytes": 0,
    "bwd_bytes": 0,
    "packet_lengths": [],
    "timestamps": [],
})


# ============================================
# FLOW KEY
# ============================================

def get_flow_key(pkt):
    try:
        src = pkt.ip.src
        dst = pkt.ip.dst
        proto = pkt.transport_layer
        sport = pkt[pkt.transport_layer].srcport
        dport = pkt[pkt.transport_layer].dstport

        return (src, sport, dst, dport, proto)
    except:
        return None


# ============================================
# UPDATE FLOW
# ============================================

def update_flow(pkt):
    key = get_flow_key(pkt)
    if key is None:
        return

    flow = flows[key]

    now = float(pkt.sniff_timestamp)
    length = int(pkt.length)

    if flow["start_time"] is None:
        flow["start_time"] = now

    flow["last_time"] = now
    flow["timestamps"].append(now)
    flow["packet_lengths"].append(length)

    # Direction heuristic
    if key[0] < key[2]:
        flow["fwd_packets"] += 1
        flow["fwd_bytes"] += length
    else:
        flow["bwd_packets"] += 1
        flow["bwd_bytes"] += length


# ============================================
# BUILD CICIDS-LIKE FEATURES
# ============================================

def build_features(flow):

    duration = flow["last_time"] - flow["start_time"] if flow["start_time"] else 0

    pkt_lengths = flow["packet_lengths"]
    timestamps = flow["timestamps"]

    if len(timestamps) > 1:
        iats = [t2 - t1 for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
    else:
        iats = [0]

    features = {
        "Flow Duration": duration,
        "Total Fwd Packets": flow["fwd_packets"],
        "Total Backward Packets": flow["bwd_packets"],
        "Total Length of Fwd Packets": flow["fwd_bytes"],
        "Total Length of Bwd Packets": flow["bwd_bytes"],
        "Flow Bytes/s": (flow["fwd_bytes"] + flow["bwd_bytes"]) / (duration + 1e-6),
        "Flow Packets/s": (flow["fwd_packets"] + flow["bwd_packets"]) / (duration + 1e-6),

        "Packet Length Mean": sum(pkt_lengths)/len(pkt_lengths) if pkt_lengths else 0,
        "Packet Length Max": max(pkt_lengths) if pkt_lengths else 0,
        "Packet Length Min": min(pkt_lengths) if pkt_lengths else 0,

        "Flow IAT Mean": sum(iats)/len(iats) if iats else 0,
        "Flow IAT Total": sum(iats) if iats else 0,
    }

    return features


# ============================================
# DETECTION LOOP
# ============================================

def process_flows():
    while True:
        now = time.time()

        to_delete = []

        for key, flow in flows.items():
            if flow["last_time"] is None:
                continue

            if now - flow["last_time"] > FLOW_TIMEOUT:

                features = build_features(flow)
                df = pd.DataFrame([features])

                try:
                    result = detector.predict_single(df.iloc[0].to_dict())

                    if result["prediction"] == 1:
                        print("\n🚨 DoS ATTACK DETECTED 🚨")
                        print(f"Flow: {key}")
                        print(f"Probability: {result['probability_dos']}")
                        print("-" * 50)

                except Exception as e:
                    print("Prediction error:", e)

                to_delete.append(key)

        for key in to_delete:
            del flows[key]

        time.sleep(1)


# ============================================
# PACKET CAPTURE
# ============================================

def capture_packets():
    capture = pyshark.LiveCapture(interface=INTERFACE)

    for pkt in capture.sniff_continuously():
        update_flow(pkt)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print("Starting Live DoS Detector...")
    print(f"Interface: {INTERFACE}")

    t1 = threading.Thread(target=capture_packets)
    t2 = threading.Thread(target=process_flows)

    t1.start()
    t2.start()

    t1.join()
    t2.join()