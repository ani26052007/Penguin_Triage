"""Shared pieces of the Phase 8 tests: the injection tool, injected repos, driven runs and their traces.

``bugs/inject.py`` lives outside any package (it must never reach a container), so it is loaded from
its file. Injected repos and driven runs are cached for the session: each is built once, from the real
policy-wrapper checkout with the ledger's ``base_tree``; the tests skip when that checkout is absent or
at another tree (the patches are diffs against it).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Optional, Sequence

import pytest

from router.prereq.trace import EpisodeTrace, InMemoryTraceSource
from router.schemas import ActorKind, ActorRef, CodeRef, Incident, Mode, SceneSpec, TrainingSpec
from router.stage1.nominal import NominalStats
from router.stage1.obs_contract import metric, observation_metrics, opponent, required_vectors

from vla_isaac.config import load_task_config
from vla_isaac.traces.instrument_reader import read_instrument_dir

from tests.helpers import PILOT_TASK

ROOT = Path(__file__).resolve().parents[1]
BUGS = ROOT / "bugs"
PW = Path(os.environ.get("POLICY_WRAPPER_ROOT", ROOT.parent / "policy-wrapper"))
DRIVER = Path(__file__).resolve().parent / "fixtures" / "pw_driver.py"
CKPT = Path(__file__).resolve().parent / "fixtures" / "checkpoint_gr1"
CKPT_ID = "nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task/ranch_bottle_into_fridge"
CAMERA = "ego_view"                     # the policy-input video key the instrument logs frames under
TASK = load_task_config(PILOT_TASK)


def _load_inject():
    spec = importlib.util.spec_from_file_location("pw_injection_tool", BUGS / "inject.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


inject = _load_inject()
LEDGER = inject.load_ledger()


def _base_head() -> str:
    if not (PW / ".git").exists():
        return ""
    r = subprocess.run(["git", "-C", str(PW), "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip()


def _base_tree() -> str:
    if not (PW / ".git").exists():
        return ""
    r = subprocess.run(["git", "-C", str(PW), "rev-parse", "HEAD^{tree}"], capture_output=True, text=True)
    return r.stdout.strip()


BASE_HEAD = _base_head()
BASE_TREE = _base_tree()
# The ledger's base is identified by its tree: the policy-wrapper history was rewritten (same tree, other hashes).
_at_base = BASE_TREE == LEDGER.base_tree if LEDGER.base_tree else BASE_HEAD.startswith(LEDGER.base_commit)
needs_base = pytest.mark.skipif(
    not _at_base,
    reason=f"policy-wrapper at {PW} is not the ledger's base (tree {LEDGER.base_tree[:10] or LEDGER.base_commit}; "
           f"HEAD {BASE_HEAD[:10] or 'none'}, tree {BASE_TREE[:10] or 'none'})")


# --------------------------------------------------------------------------- #
# injected repos and driven runs, cached for the session
# --------------------------------------------------------------------------- #

@dataclass
class Store:
    root: Path
    repos: dict
    runs: dict

    def repo(self, *bug_ids: str, case: Optional[str] = None):
        """The injected repo of ``bug_ids`` (+ ``case``); no ids: the clean repo, built the same way."""
        key = (tuple(bug_ids), case)
        if key not in self.repos:
            name = "-".join(bug_ids) or "clean"
            out = Path(tempfile.mkdtemp(prefix=f"{name.lower()}-", dir=self.root)) / "repo"
            self.repos[key] = inject.inject(PW, out, bug_ids, case=case, checkpoint_stats=CKPT / "statistics.json",
                                            ledger=LEDGER)
        return self.repos[key]

    def run(self, *bug_ids: str, scene: str = "bright", num_envs: int = 1, episodes: int = 3, steps: int = 48,
            seed: int = 0, env: Optional[Mapping[str, str]] = None, server_delay_ms: float = 0.0,
            step_ms: float = 0.0) -> "Run":
        """The injected repo's own client driven by ``tests/fixtures/pw_driver.py``."""
        env = dict(env or {})
        key = (tuple(bug_ids), scene, num_envs, episodes, steps, seed, tuple(sorted(env.items())), server_delay_ms,
               step_ms)
        if key not in self.runs:
            repo = self.repo(*bug_ids).repo
            out = Path(tempfile.mkdtemp(prefix="run-", dir=self.root))
            argv = [sys.executable, str(DRIVER), "--repo", str(repo), "--out", str(out), "--episodes", str(episodes),
                    "--steps", str(steps), "--num-envs", str(num_envs), "--scene", scene, "--seed", str(seed),
                    "--server-delay-ms", str(server_delay_ms), "--step-ms", str(step_ms)]
            for k, v in sorted(env.items()):
                argv += ["--env", f"{k}={v}"]
            clean_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            clean_env["PYTHONDONTWRITEBYTECODE"] = "1"          # the cached repo stays exactly as injected
            r = subprocess.run(argv, capture_output=True, text=True, env=clean_env, timeout=600)
            if r.returncode != 0:
                raise AssertionError(f"driver failed on {bug_ids or 'clean'}:\n{r.stderr[-3000:]}")
            self.runs[key] = Run(out)
        return self.runs[key]


@pytest.fixture(scope="session")
def store():
    root = Path(tempfile.mkdtemp(prefix="pw-phase8-"))
    try:
        yield Store(root, {}, {})
    finally:
        shutil.rmtree(root, ignore_errors=True)


@dataclass
class Run:
    out: Path

    def episodes(self):
        return read_instrument_dir(self.out)

    def actions(self) -> dict[tuple[int, int], list[list[float]]]:
        """The actions returned to ``env.step``, per (env, episode), in the 36-joint action order."""
        out = {}
        for p in sorted(self.out.glob("actions_env*_ep*.json")):
            e, k = p.stem.removeprefix("actions_env").split("_ep")
            out[(int(e), int(k))] = json.loads(p.read_text())
        return out

    def traces(self, mode: Mode, code_key: str, **extra: Any) -> list[EpisodeTrace]:
        return [EpisodeTrace(seed=i, mode=mode, frames=ep.frames, step_pairs=ep.step_pairs,
                             raw_actions=ep.raw_actions, code_key=code_key, **extra)
                for i, (_, ep) in enumerate(sorted(self.episodes().items()))]


# --------------------------------------------------------------------------- #
# incidents and nominal stats
# --------------------------------------------------------------------------- #

CODE = CodeRef("pw@injected", {"client": {}})
POLICY = ActorRef(ActorKind.POLICY, "ckpt")
SPEC = TrainingSpec("gr00t_n1d6_gr1_ranch_bottle_into_fridge", image_size=(512, 512), channel_order="RGB")


def incident(source_refs: Mapping[str, str], observed: Mode = Mode.CLEAN, spec: TrainingSpec = SPEC) -> Incident:
    return Incident("phase8", SceneSpec(TASK.nominal_scene), spec, CODE, POLICY, (0,), observed,
                    log_refs=dict(source_refs))


def source(by_mode: Mapping[Mode, Sequence[EpisodeTrace]]) -> tuple[InMemoryTraceSource, dict[str, str]]:
    refs = {m.value: f"traces/{m.value}" for m in by_mode}
    return InMemoryTraceSource({refs[m.value]: list(t) for m, t in by_mode.items()}), refs


def obs_nominal(clean: Sequence[EpisodeTrace], cameras: Sequence[str] = (CAMERA,), rel_floor: float = 0.05,
                abs_floor: float = 0.02) -> NominalStats:
    """Observation nominal calibrated on no-fault traces: reference vectors are the medians of the
    policy-input profiles / opponent vectors, metric floors ``max(rel_floor * |mean|, abs_floor)``."""
    vectors = {}
    for cam in cameras:
        frames = [fs.policy_input for t in clean for fs in t.frames if fs.camera == cam]
        vectors[metric(cam, "opp")] = tuple(median(c) for c in zip(*(opponent(f.channel_means) for f in frames)))
        vectors[metric(cam, "row_profile")] = tuple(median(c) for c in zip(*(f.row_profile for f in frames)))
        vectors[metric(cam, "col_profile")] = tuple(median(c) for c in zip(*(f.col_profile for f in frames)))
    assert set(vectors) == set(required_vectors(cameras))
    samples = observation_metrics(clean, cameras, vectors)
    floors = {k: max(rel_floor * abs(sum(v) / len(v)), abs_floor) for k, v in samples.items()}
    return NominalStats.from_samples(samples, vectors=vectors, min_std=floors)
