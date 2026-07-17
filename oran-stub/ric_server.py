import argparse
import json
import socketserver

BLER_ROLLBACK_THRESHOLD = 0.15


class RicHandler(socketserver.StreamRequestHandler):
    def handle(self):
        line = self.rfile.readline()
        if not line:
            return
        report = json.loads(line.decode("utf-8"))
        print(f"[ric] REPORT from {report['cell_id']}: estimator={report['estimator']} "
              f"bler={report['bler_measured']:.3f} snr={report['snr_db']}dB")

        if report["estimator"] == "cnn" and report["bler_measured"] > BLER_ROLLBACK_THRESHOLD:
            action = {"type": "CONTROL", "action": "switch_estimator", "target": "classical_ls",
                      "reason": f"measured BLER {report['bler_measured']:.3f} exceeds "
                                f"rollback threshold {BLER_ROLLBACK_THRESHOLD}"}
        else:
            action = {"type": "CONTROL", "action": "keep", "target": report["estimator"],
                      "reason": "within acceptable BLER"}

        print(f"[ric] -> CONTROL: {action['action']} target={action['target']} ({action['reason']})")
        self.wfile.write((json.dumps(action) + "\n").encode("utf-8"))


def main(port=8765):
    with socketserver.TCPServer(("127.0.0.1", port), RicHandler) as server:
        print(f"[ric] listening on 127.0.0.1:{port} (Ctrl+C to stop)")
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    main(args.port)
