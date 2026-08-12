"""
generate_logs.py
-----------------
Generates a synthetic, labeled dataset that combines:
  1. Authentication logs (login attempts)
  2. Network traffic logs (connection metadata)

into a single per-session record, mimicking how a SIEM would correlate
auth events with network flow data for the same source IP.

Why synthetic data: labeled attack data is scarce/sensitive in the real
world too, so security ML teams commonly build simulators to generate
training data with known ground truth. This mirrors that practice.

Injected attack patterns (labeled anomaly=1):
  - Brute force login (many failed attempts, short intervals, single IP)
  - Credential stuffing (many distinct usernames, few attempts each, single IP)
  - Off-hours privileged access (login at unusual hours from unusual geo)
  - Port scan (many distinct destination ports, low bytes each, short duration)
  - Data exfiltration-like spike (large outbound bytes, long duration, off-hours)

Normal traffic (label=0) is modeled with realistic diurnal patterns
(more activity during business hours) and typical login success rates.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

N_NORMAL = 9000
N_ANOMALY_PER_TYPE = 250  # 5 attack types -> 1250 anomalies (~12% of dataset)

START_TIME = datetime(2026, 1, 1)
USERS = [f"user{i:03d}" for i in range(1, 120)]
COMMON_PORTS = [80, 443, 22, 3389, 445, 8080, 53]


def random_ip(private_bias=0.7):
    if RNG.random() < private_bias:
        return f"10.0.{RNG.integers(0, 50)}.{RNG.integers(1, 254)}"
    return f"{RNG.integers(1, 223)}.{RNG.integers(0, 255)}.{RNG.integers(0, 255)}.{RNG.integers(1, 254)}"


def diurnal_hour():
    # business-hours-weighted hour of day (9am-6pm more likely)
    weights = np.array([1, 1, 1, 1, 1, 1, 2, 4, 7, 9, 10, 10,
                         9, 10, 10, 9, 8, 6, 4, 3, 2, 2, 1, 1], dtype=float)
    weights /= weights.sum()
    return RNG.choice(24, p=weights)


def gen_normal(n):
    rows = []
    for _ in range(n):
        hour = diurnal_hour()
        ts = START_TIME + timedelta(
            days=int(RNG.integers(0, 60)), hours=int(hour),
            minutes=int(RNG.integers(0, 60)), seconds=int(RNG.integers(0, 60))
        )
        rows.append({
            "timestamp": ts,
            "src_ip": random_ip(),
            "username": RNG.choice(USERS),
            "login_attempts": int(RNG.poisson(1.2)) + 1,
            "login_success": 1 if RNG.random() < 0.94 else 0,
            "distinct_usernames_from_ip_1h": 1,
            "session_duration_sec": float(np.clip(RNG.normal(420, 180), 10, 3600)),
            "bytes_out": float(np.abs(RNG.normal(50_000, 30_000))),
            "bytes_in": float(np.abs(RNG.normal(80_000, 40_000))),
            "distinct_dst_ports_1h": int(RNG.integers(1, 4)),
            "dst_port": int(RNG.choice(COMMON_PORTS)),
            "hour_of_day": hour,
            "label": 0,
            "attack_type": "none",
        })
    return rows


def gen_brute_force(n):
    rows = []
    for _ in range(n):
        hour = diurnal_hour()
        ts = START_TIME + timedelta(days=int(RNG.integers(0, 60)), hours=int(hour),
                                     minutes=int(RNG.integers(0, 60)))
        ip = random_ip(private_bias=0.2)
        rows.append({
            "timestamp": ts, "src_ip": ip, "username": RNG.choice(USERS),
            "login_attempts": int(RNG.integers(15, 80)),
            "login_success": 1 if RNG.random() < 0.05 else 0,
            "distinct_usernames_from_ip_1h": int(RNG.integers(1, 3)),
            "session_duration_sec": float(np.clip(RNG.normal(30, 10), 5, 120)),
            "bytes_out": float(np.abs(RNG.normal(5_000, 2_000))),
            "bytes_in": float(np.abs(RNG.normal(5_000, 2_000))),
            "distinct_dst_ports_1h": 1,
            "dst_port": 22,
            "hour_of_day": hour, "label": 1, "attack_type": "brute_force",
        })
    return rows


def gen_credential_stuffing(n):
    rows = []
    for _ in range(n):
        hour = diurnal_hour()
        ts = START_TIME + timedelta(days=int(RNG.integers(0, 60)), hours=int(hour),
                                     minutes=int(RNG.integers(0, 60)))
        ip = random_ip(private_bias=0.1)
        rows.append({
            "timestamp": ts, "src_ip": ip, "username": RNG.choice(USERS),
            "login_attempts": int(RNG.integers(1, 3)),
            "login_success": 1 if RNG.random() < 0.08 else 0,
            "distinct_usernames_from_ip_1h": int(RNG.integers(20, 90)),
            "session_duration_sec": float(np.clip(RNG.normal(15, 5), 3, 60)),
            "bytes_out": float(np.abs(RNG.normal(3_000, 1_000))),
            "bytes_in": float(np.abs(RNG.normal(3_000, 1_000))),
            "distinct_dst_ports_1h": 1,
            "dst_port": 443,
            "hour_of_day": hour, "label": 1, "attack_type": "credential_stuffing",
        })
    return rows


def gen_offhours_access(n):
    rows = []
    for _ in range(n):
        hour = int(RNG.choice([0, 1, 2, 3, 4, 23]))
        ts = START_TIME + timedelta(days=int(RNG.integers(0, 60)), hours=hour,
                                     minutes=int(RNG.integers(0, 60)))
        rows.append({
            "timestamp": ts, "src_ip": random_ip(private_bias=0.15),
            "username": RNG.choice(USERS[:15]),  # privileged accounts
            "login_attempts": 1, "login_success": 1,
            "distinct_usernames_from_ip_1h": 1,
            "session_duration_sec": float(np.clip(RNG.normal(1800, 600), 300, 5000)),
            "bytes_out": float(np.abs(RNG.normal(40_000, 15_000))),
            "bytes_in": float(np.abs(RNG.normal(60_000, 20_000))),
            "distinct_dst_ports_1h": int(RNG.integers(1, 3)),
            "dst_port": 3389,
            "hour_of_day": hour, "label": 1, "attack_type": "offhours_privileged_access",
        })
    return rows


def gen_port_scan(n):
    rows = []
    for _ in range(n):
        hour = diurnal_hour()
        ts = START_TIME + timedelta(days=int(RNG.integers(0, 60)), hours=int(hour),
                                     minutes=int(RNG.integers(0, 60)))
        rows.append({
            "timestamp": ts, "src_ip": random_ip(private_bias=0.1),
            "username": "N/A", "login_attempts": 0, "login_success": 0,
            "distinct_usernames_from_ip_1h": 0,
            "session_duration_sec": float(np.clip(RNG.normal(2, 1), 0.1, 10)),
            "bytes_out": float(np.abs(RNG.normal(200, 100))),
            "bytes_in": float(np.abs(RNG.normal(100, 50))),
            "distinct_dst_ports_1h": int(RNG.integers(50, 500)),
            "dst_port": int(RNG.integers(1, 65535)),
            "hour_of_day": hour, "label": 1, "attack_type": "port_scan",
        })
    return rows


def gen_exfiltration(n):
    rows = []
    for _ in range(n):
        hour = int(RNG.choice([1, 2, 3, 22, 23]))
        ts = START_TIME + timedelta(days=int(RNG.integers(0, 60)), hours=hour,
                                     minutes=int(RNG.integers(0, 60)))
        rows.append({
            "timestamp": ts, "src_ip": random_ip(private_bias=0.6),
            "username": RNG.choice(USERS), "login_attempts": 1, "login_success": 1,
            "distinct_usernames_from_ip_1h": 1,
            "session_duration_sec": float(np.clip(RNG.normal(2400, 800), 600, 6000)),
            "bytes_out": float(np.abs(RNG.normal(5_000_000, 2_000_000))),
            "bytes_in": float(np.abs(RNG.normal(20_000, 10_000))),
            "distinct_dst_ports_1h": int(RNG.integers(1, 3)),
            "dst_port": 443,
            "hour_of_day": hour, "label": 1, "attack_type": "data_exfiltration",
        })
    return rows


def main():
    rows = []
    rows += gen_normal(N_NORMAL)
    rows += gen_brute_force(N_ANOMALY_PER_TYPE)
    rows += gen_credential_stuffing(N_ANOMALY_PER_TYPE)
    rows += gen_offhours_access(N_ANOMALY_PER_TYPE)
    rows += gen_port_scan(N_ANOMALY_PER_TYPE)
    rows += gen_exfiltration(N_ANOMALY_PER_TYPE)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
    df.to_csv("/home/claude/security-anomaly-detector/data/security_logs.csv", index=False)

    print(f"Generated {len(df)} records")
    print(f"Normal: {(df.label == 0).sum()} | Anomalous: {(df.label == 1).sum()}")
    print(f"Anomaly rate: {df.label.mean():.2%}")
    print(df.attack_type.value_counts())


if __name__ == "__main__":
    main()
