"""Policy-server process control against a fake ZMQ REP server (GR00T's wire format)."""

from __future__ import annotations

import socket
import sys
import threading
from pathlib import Path

import msgpack
import pytest
import zmq

from vla_isaac.backend.server import PolicyServerProcess, ServerError, call_endpoint, ping
from vla_isaac.config import ServerConfig

FAKE = Path(__file__).parent / "fixtures" / "fake_rep_server.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def cfg(port: int, **kw) -> ServerConfig:
    base = dict(host="127.0.0.1", port=port, start_timeout_s=20, ping_timeout_ms=300, ping_interval_s=0.05,
                stop_timeout_s=5)
    base.update(kw)
    return ServerConfig(**base)


def proc(tmp_path, port, *extra, **kw) -> PolicyServerProcess:
    return PolicyServerProcess([sys.executable, str(FAKE), "--port", str(port), *extra], cwd=tmp_path, env={},
                               cfg=cfg(port, **kw), log_path=tmp_path / "server.log")


class ThreadServer:
    """In-process REP server answering a fixed reply per endpoint. The socket lives and dies in its
    own thread (closing a ZMQ socket from another thread while it blocks aborts libzmq)."""

    def __init__(self, replies):
        self.port, self.replies, self.seen = free_port(), replies, []
        self._stop, self._ready = threading.Event(), threading.Event()
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()
        self._ready.wait(5)

    def loop(self):
        sock = zmq.Context.instance().socket(zmq.REP)
        sock.bind(f"tcp://127.0.0.1:{self.port}")
        self._ready.set()
        try:
            while not self._stop.is_set():
                if not sock.poll(50):
                    continue
                req = msgpack.unpackb(sock.recv())
                self.seen.append(req)
                sock.send(msgpack.packb(self.replies.get(req["endpoint"], {"error": "unknown"})))
        finally:
            sock.close(linger=0)

    def close(self):
        self._stop.set()
        self.t.join(5)


def test_ping_speaks_the_wire_format():
    srv = ThreadServer({"ping": {"status": "ok", "message": "Server is running"}})
    try:
        assert ping("127.0.0.1", srv.port, 1000)
        assert srv.seen == [{"endpoint": "ping"}]
        assert call_endpoint("127.0.0.1", srv.port, "get_action", 1000, {"observation": {}}) == {"error": "unknown"}
        assert srv.seen[-1] == {"endpoint": "get_action", "data": {"observation": {}}}
    finally:
        srv.close()


def test_ping_is_false_for_an_error_reply_and_for_silence():
    srv = ThreadServer({"ping": {"error": "Unauthorized: Invalid API token"}})
    try:
        assert not ping("127.0.0.1", srv.port, 1000)
    finally:
        srv.close()
    assert not ping("127.0.0.1", free_port(), 200)


def test_start_waits_for_ping_then_stop_kills_it(tmp_path):
    port = free_port()
    p = proc(tmp_path, port, "--delay", "0.5")
    p.start()
    assert p.running and p.ping()
    p.stop()
    assert not p.running and not ping("127.0.0.1", port, 200)
    assert "fake server listening" in (tmp_path / "server.log").read_text()


def test_restart_and_context_manager(tmp_path):
    port = free_port()
    with proc(tmp_path, port) as p:
        p.restart()
        assert p.ping()
    assert not ping("127.0.0.1", port, 200)


def test_a_server_that_ignores_kill_is_terminated(tmp_path):
    port = free_port()
    p = proc(tmp_path, port, "--ignore-kill", stop_timeout_s=0.5)
    p.start()
    p.stop()
    assert not ping("127.0.0.1", port, 200)


def test_a_server_that_exits_before_answering_raises_with_its_log(tmp_path):
    with pytest.raises(ServerError, match="(?s)exited with 3.*failing on purpose"):
        proc(tmp_path, free_port(), "--exit-code", "3").start()


def test_a_server_that_never_answers_times_out(tmp_path):
    p = proc(tmp_path, free_port(), "--delay", "30", start_timeout_s=0.5)
    with pytest.raises(ServerError, match="did not answer ping"):
        p.start()
    assert not p.running


def test_start_refuses_a_port_someone_else_serves(tmp_path):
    srv = ThreadServer({"ping": {"status": "ok"}})
    try:
        with pytest.raises(ServerError, match="already answers"):
            proc(tmp_path, srv.port).start()
    finally:
        srv.close()
