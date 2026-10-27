"""Run one Arena command with a timeout and keep its log; a fake for tests.

A non-zero exit or a timeout raises ``ArenaRunError``. ``IsaacBackend.run`` lets it propagate:
the incident loop catches any ``Exception`` and ends the incident UNRESOLVED with
``error_stage`` router or rung1 (Fix F3, ``R:ladder/loop.py:313``).

``FakeArenaRunner`` writes ``episode_results_rebuild0.jsonl`` the way the experiment runner does,
with the field names of ``A:isaaclab_arena/recording/common_terms.py:131-172`` (core record,
``variations``, ``placement``) and ``progress_terms.py:18-67`` (``progress``). The values are
FIXTURE data.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

from vla_isaac.runner.arena_cmd import RunSpec


class ArenaRunError(RuntimeError):
    """An Arena run that did not finish cleanly (crash, non-zero exit, timeout)."""

    def __init__(self, message: str, *, returncode: Optional[int] = None, timed_out: bool = False,
                 log_path: Optional[Path] = None) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.timed_out = timed_out
        self.log_path = log_path


@dataclass(frozen=True)
class RunInvocation:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]                  # added to os.environ (PYTHONPATH is always set explicitly)
    timeout_s: float
    log_path: Path
    output_dir: Path                        # the experiment output directory
    spec: RunSpec                           # RunSpec for Arena runs; vla_isaac.reference.mimic.MimicRunSpec for Mimic


def log_tail(path: Optional[Path], n: int = 2000) -> str:
    if path is None or not Path(path).is_file():
        return ""
    data = Path(path).read_bytes()[-n:]
    return data.decode("utf-8", "backslashreplace")


class ArenaRunner(ABC):
    @abstractmethod
    def run(self, inv: RunInvocation) -> None:
        """Run ``inv`` to completion or raise ``ArenaRunError``."""


class SubprocessRunner(ArenaRunner):
    def run(self, inv: RunInvocation) -> None:
        inv.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **inv.env}
        with open(inv.log_path, "wb") as log:
            try:
                p = subprocess.run(list(inv.argv), cwd=inv.cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, timeout=inv.timeout_s)
            except subprocess.TimeoutExpired:
                raise ArenaRunError(f"Arena run {inv.spec.run_name} timed out after {inv.timeout_s:.0f}s; "
                                    f"log {inv.log_path}", timed_out=True, log_path=inv.log_path) from None
            except OSError as e:
                raise ArenaRunError(f"cannot start Arena run {inv.spec.run_name}: {e}", log_path=inv.log_path) from e
        if p.returncode != 0:
            raise ArenaRunError(f"Arena run {inv.spec.run_name} exited with {p.returncode}; log {inv.log_path}\n"
                                f"{log_tail(inv.log_path)}", returncode=p.returncode, log_path=inv.log_path)


# --------------------------------------------------------------------------- #
# fake
# --------------------------------------------------------------------------- #

def _uniform(*parts: object) -> float:
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


SuccessFn = Callable[[RunSpec, int], bool]


def rate_success(rate: float, salt: str = "fake") -> SuccessFn:
    """Episode k of a build succeeds with probability ``rate``, deterministically in
    (variations, seed, k): the same build and episode always give the same outcome."""
    def fn(spec: RunSpec, k: int) -> bool:
        return _uniform(salt, json.dumps(dict(spec.variations), sort_keys=True), spec.seed, k) < rate
    return fn


@dataclass
class FakeArenaRunner(ArenaRunner):
    """Writes fixture episode JSONL instead of running Isaac. ``crash_returncode`` makes the next
    runs fail like a crashed subprocess; ``short_by`` writes fewer episodes than asked; ``on_run(inv,
    run_dir, records)`` is called before the JSONL is written and may edit the records (tests use it
    to write trace fixtures that match them)."""
    success: SuccessFn = field(default_factory=lambda: rate_success(0.8))
    crash_returncode: Optional[int] = None
    short_by: int = 0
    calls: list[RunInvocation] = field(default_factory=list)
    on_run: Optional[Callable[[RunInvocation, Path, list[dict]], None]] = None

    def run(self, inv: RunInvocation) -> None:
        self.calls.append(inv)
        inv.log_path.parent.mkdir(parents=True, exist_ok=True)
        inv.log_path.write_text(f"FAKE {' '.join(inv.argv)}\n", encoding="utf-8")
        if self.crash_returncode is not None:
            raise ArenaRunError(f"Arena run {inv.spec.run_name} exited with {self.crash_returncode} (fake)",
                                returncode=self.crash_returncode, log_path=inv.log_path)
        spec = inv.spec
        run_dir = inv.output_dir / spec.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        records = fake_episode_records(spec, self.success, short_by=self.short_by)
        if self.on_run is not None:
            self.on_run(inv, run_dir, records)
        lines = [json.dumps(r) for r in records]
        (run_dir / "episode_results_rebuild0.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fake_episode_records(spec: RunSpec, success: SuccessFn, short_by: int = 0) -> list[dict]:
    """FIXTURE records in the experiment runner's format. Episode k (start order) runs in env
    ``k % num_envs`` as its ``k // num_envs``-th episode; records are written in that order."""
    out = []
    for k in range(spec.num_episodes - short_by):
        ok = bool(success(spec, k))
        out.append({
            "job_name": spec.run_name,
            "env_id": k % spec.num_envs,
            "episode_in_env": k // spec.num_envs,
            "seed": spec.seed,
            "success": ok,
            "episode_length": 500 if not ok else 380,
            "language_instruction": spec.language_instruction,
            "timestamp": "2000-01-01T00:00:00",
            # Recorded under "<asset>.<variation>" (VERIFY(V-var-record): value format of a build-time draw).
            "variations": {key[:-len(".sampler_cfg.low")]: value for key, value in spec.variations.items()
                           if key.endswith(".sampler_cfg.low")},
            "placement": {"ranch_dressing_hope_robolab": {"position_xyz": [0.1 * k, 0.0, 1.0]}},
            "progress": {
                "overall_score": 1.0 if ok else 0.5,
                "all_complete": ok,
                "criteria_by_name": {},
                "events": [{"step": 120, "criteria_name": "pick_and_place", "sequence_name": "main",
                            "predicate_index": 1, "predicate_name": "object_is_above_height", "score_delta": 0.25}],
            },
        })
    return out
