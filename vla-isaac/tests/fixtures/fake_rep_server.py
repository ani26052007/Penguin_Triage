"""FIXTURE: a stand-in policy server speaking GR00T's wire format (msgpack over ZMQ REP).

    python fake_rep_server.py --port P [--delay S] [--exit-code N] [--ignore-kill]

Answers ``ping`` with {"status": "ok"}, ``kill`` by leaving its loop, anything else with
{"error": ...}. ``--delay`` waits before binding (a slow model load); ``--exit-code`` exits at once.
"""
import argparse
import sys
import time

import msgpack
import zmq

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--delay", type=float, default=0.0)
ap.add_argument("--exit-code", type=int, default=None)
ap.add_argument("--ignore-kill", action="store_true")
a = ap.parse_args()
if a.exit_code is not None:
    print("fake server failing on purpose", flush=True)
    sys.exit(a.exit_code)
time.sleep(a.delay)
sock = zmq.Context().socket(zmq.REP)
sock.bind(f"tcp://127.0.0.1:{a.port}")
print("fake server listening", flush=True)
while True:
    req = msgpack.unpackb(sock.recv())
    ep = req.get("endpoint")
    if ep == "ping":
        sock.send(msgpack.packb({"status": "ok", "message": "Server is running"}))
    elif ep == "kill":
        sock.send(msgpack.packb({}))
        if not a.ignore_kill:
            break
    else:
        sock.send(msgpack.packb({"error": f"Unknown endpoint: {ep}"}))
