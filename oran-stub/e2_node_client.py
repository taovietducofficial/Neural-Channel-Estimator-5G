import argparse
import csv
import json
import os
import socket
import time

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "bler_results.csv")


def load_real_bler_rows():
    with open(RESULTS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def send_report(host, port, cell_id, estimator, bler_measured, snr_db):
    report = {
        "type": "REPORT",
        "cell_id": cell_id,
        "estimator": estimator,
        "bler_measured": bler_measured,
        "snr_db": snr_db,
    }
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall((json.dumps(report) + "\n").encode("utf-8"))
        response = sock.makefile("r").readline()
    return json.loads(response)


def main(host="127.0.0.1", port=8765, cell_id="cell-1"):
    rows = load_real_bler_rows()
    print(f"[e2-node] loaded {len(rows)} real measured SNR/BLER points from {RESULTS_CSV}\n")

    for row in rows:
        snr_db = float(row["snr_db"])
        bler_cnn = float(row["cnn"])
        control = send_report(host, port, cell_id, "cnn", bler_cnn, snr_db)
        print(f"[e2-node] received CONTROL: {control['action']} -> use {control['target']}\n")
        time.sleep(0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cell-id", default="cell-1")
    args = parser.parse_args()
    main(args.host, args.port, args.cell_id)
