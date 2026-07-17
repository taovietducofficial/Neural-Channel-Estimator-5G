# O-RAN model-selection stub (illustrative, not spec-compliant)

`docs/oran-integration.md` argues that the neural channel estimator itself
runs far too fast (per-slot, sub-millisecond) to be an xApp or rApp -- but
the *decision of which estimator a cell should currently use* (based on
recently measured BLER) genuinely operates at rApp/xApp timescale. This is
a minimal, runnable demonstration of that specific pattern -- nothing more.

**What this is:** two independent local processes exchanging JSON messages
over a real TCP socket, using the exact BLER numbers already measured in
`results/bler_results.csv`:

- `ric_server.py` plays the RIC/xApp side: receives a `REPORT` (which
  estimator a cell is using and its recently measured BLER), applies a
  simple policy (roll back to the classical estimator if BLER exceeds a
  threshold), and replies with a `CONTROL` message.
- `e2_node_client.py` plays the E2 Node/O-DU side: sends one `REPORT` per
  real measured SNR point from the actual evaluation run, and prints the
  `CONTROL` response it gets back.

**What this is not:** a real E2AP/ASN.1-encoded interface, not SCTP
transport, not integrated with any real RIC platform (OSC or FlexRIC -- see
`docs/oran-integration.md` for why a full RIC build was assessed and
intentionally not attempted here). The JSON-over-TCP protocol here is a
stand-in chosen to demonstrate the *message-flow pattern* (periodic report
-> policy decision -> control action) at the right conceptual layer, not to
claim spec compliance.

## Run it

```bash
# terminal 1
python oran-stub/ric_server.py

# terminal 2
python oran-stub/e2_node_client.py
```
