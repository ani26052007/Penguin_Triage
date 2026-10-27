"""Start, ping, stop and restart the policy server (policy-wrapper's ``serve.py``).

The wire format is GR00T's (``G:gr00t/policy/server_client.py``): a ZeroMQ REP socket, one
msgpack request at a time, ``{"endpoint": <name>, "data": {...}}`` in and a msgpack reply out;
a failed handler replies ``{"error": ...}``. ``ping`` replies ``{"status": "ok", ...}`` and ``kill``
makes the server leave its loop (``:68-89, 110-142``). This module speaks that format itself and
copies no GR00T code (GR00T is under the NVIDIA License, IMPL_PLAN D15).

``zmq`` and ``msgpack`` are imported inside the functions, so the module imports without them.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from vla_isaac.config import ServerConfig
from vla_isaac.runner.subprocess_runner import log_tail


class ServerError(RuntimeError):
    """The policy server did not come up, or something else holds its port."""


def call_endpoint(host: str, port: int, endpoint: str, timeout_ms: int, data: Optional[dict] = None) -> Any:
    """One request/reply. Raises ``TimeoutError`` when nothing answers within ``timeout_ms``."""
    import msgpack
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
    sock.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
    try:
        sock.connect(f"tcp://{host}:{port}")
        req: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        sock.send(msgpack.packb(req))
        try:
            msg = sock.recv()
        except zmq.Again:
            raise TimeoutError(f"no reply from {host}:{port} to {endpoint!r} within {timeout_ms} ms") from None
        return msgpack.unpackb(msg)
    finally:
        sock.close(linger=0)


def ping(host: str, port: int, timeout_ms: int = 2000) -> bool:
    try:
        reply = call_endpoint(host, port, "ping", timeout_ms)
    except (TimeoutError, OSError):
        return False
    return isinstance(reply, dict) and reply.get("status") == "ok"


class PolicyServerProcess:
    """One server process. ``argv`` runs inside the materialised working copy (``cwd``)."""

    def __init__(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], cfg: ServerConfig,
                 log_path: Path, popen: Callable[..., subprocess.Popen] = subprocess.Popen,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.argv = list(argv)
        self.cwd, self.env, self.cfg, self.log_path = Path(cwd), dict(env), cfg, Path(log_path)
        self._popen, self._clock, self._sleep = popen, clock, sleep
        self._proc: Optional[subprocess.Popen] = None
        self._log = None

    def ping(self) -> bool:
        return ping(self.cfg.host, self.cfg.port, self.cfg.ping_timeout_ms)

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self.running:
            raise ServerError("server already started")
        if self.ping():
            raise ServerError(f"something already answers ping on {self.cfg.host}:{self.cfg.port}; "
                              f"refusing to evaluate against a server this backend did not start")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        self._proc = self._popen(self.argv, cwd=self.cwd, env={**os.environ, **self.env}, stdout=self._log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        deadline = self._clock() + self.cfg.start_timeout_s
        while True:
            code = self._proc.poll()
            if code is not None:
                self._close_log()
                raise ServerError(f"policy server exited with {code} before answering ping; log {self.log_path}\n"
                                  f"{log_tail(self.log_path)}")
            if self.ping():
                return
            if self._clock() >= deadline:
                self.stop()
                raise ServerError(f"policy server did not answer ping within {self.cfg.start_timeout_s:.0f}s; "
                                  f"log {self.log_path}\n{log_tail(self.log_path)}")
            self._sleep(self.cfg.ping_interval_s)

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    call_endpoint(self.cfg.host, self.cfg.port, "kill", self.cfg.ping_timeout_ms)
                except (TimeoutError, OSError):
                    pass
                try:
                    proc.wait(timeout=self.cfg.stop_timeout_s)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=self.cfg.stop_timeout_s)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        finally:
            self._close_log()

    def restart(self) -> None:
        self.stop()
        self.start()

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> "PolicyServerProcess":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
