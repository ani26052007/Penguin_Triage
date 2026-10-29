"""scripts/gpu/router_obs_check.py (from the 2026-10-29 GPU session) without a GPU.

Argument parsing, and a run on instrument files written by the real policy_wrapper Instrument driven with
synthetic 32x32 frames (a FIXTURE: the frames are not Isaac renders). The clean fixture checked against its own
nominal must raise no orientation or channel-order evidence."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import numpy as np

from tests import trace_fixtures
from tests.trace_fixtures import PW, run_instrument, script

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gpu" / "router_obs_check.py"


def load():
    spec = importlib.util.spec_from_file_location("router_obs_check", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_script_imports_and_parses_its_arguments():
    m = load()
    a = m.parse_args(["nom", "a", "b"])
    assert a.nominal_dir == Path("nom") and a.check_dirs == [Path("a"), Path("b")]
    with pytest.raises(SystemExit):
        m.parse_args(["only_one"])


def test_it_has_the_session_header():
    assert "2026-10-29" in SCRIPT.read_text().split('"""')[1]


@pytest.mark.skipif(not PW.exists(), reason="policy-wrapper checkout not present")
def test_a_clean_instrument_directory_gives_no_evidence_against_its_own_nominal(tmp_path, capsys, monkeypatch):
    def frame(step, env):                      # the shared fixture frame has constant columns: no column profile
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[..., 0] = np.linspace(0, 250, 32, dtype=np.uint8)[None, :]
        img[..., 1] = np.linspace(0, 250, 32, dtype=np.uint8)[:, None]
        img[..., 2] = (step * 7 + env * 13) % 200
        return img

    monkeypatch.setattr(trace_fixtures, "_frame", frame)
    d = tmp_path / "clean"
    run_instrument(d, 1, {0: [script("success"), script("success")]}, frame_every=5)
    assert load().main([str(d), str(d)]) == 0
    out = capsys.readouterr().out
    # the 32x32 fixture frames do break the script's 512x512 training spec; nothing else may fire
    assert "nominal: " in out and "clean: 1 evidence item(s)" in out
    assert "image_size" in out and "orientation" not in out and "channel_order" not in out
