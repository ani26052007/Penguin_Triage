"""Synthetic traces, nominal statistics and incidents for the Stage 1 checks.

The mock side of Stage 1 (plan.md 3.2.14: "unit-test each Stage 1 check on
synthetic traces"). No simulator, policy or GPU. A trace is generated from a
seed and a ``Faults`` record; with ``Faults()`` it is a healthy episode, and
each field of ``Faults`` plants one known fault, so a test can check that the
matching check recovers it and that no other check fires.

What is calibration and what is test setup
------------------------------------------
``make_nominal`` builds a ``NominalStats`` from healthy synthetic episodes with
the checks' own metric functions, the same way step 6 will build it from real
no-fault runs. ``FLOORS`` are the per-metric ``min_std`` floors chosen for these
synthetic traces (a constant metric has zero spread, so a floor decides how
small a deviation counts); they are test settings, not calibrated values.

Everything is deterministic: ``random.Random`` seeded from (seed, mode).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from statistics import fmean
from typing import Callable, Mapping, Optional, Sequence, TypeVar

from router.prereq.trace import (EpisodeTrace, FrameSample, FrameStats, InMemoryTraceSource, SimHealth,
                                 StepPair, TraceSource, TraceUnavailable)
from router.schemas import ActorKind, ActorRef, CodeRef, Factor, Incident, Mode, SceneSpec, TrainingSpec
from router.stage1 import action_contract as A
from router.stage1 import obs_contract as O
from router.stage1 import sanity as S
from router.stage1 import staleness as T
from router.stage1.cfg_contract import ConfigMapping
from router.stage1.nominal import NominalStats

CAMERAS = ("front", "wrist")
PROFILE_BINS = 16
STEPS = 96
CHUNK = 8
N_JOINTS = 7
ACTION_DIMS = 8          # 7 joints + 1 gripper
BASE_LATENCY = 1         # healthy extra delay, in steps
GRIPPER_OPEN = 0.08      # metres


def _lin(a: float, b: float, n: int = PROFILE_BINS) -> list[float]:
    return [a + (b - a) * i / (n - 1) for i in range(n)]


_BASE = {
    "front": {"means": (0.55, 0.45, 0.30), "rows": _lin(0.8, 0.2), "cols": _lin(0.3, 0.7)},
    "wrist": {"means": (0.35, 0.50, 0.40), "rows": _lin(0.25, 0.75), "cols": _lin(0.7, 0.3)},
}

# test-setup floors (see module docstring)
FLOORS: Mapping[str, float] = {
    **{O.metric(c, "value_min"): 0.02 for c in CAMERAS}, **{O.metric(c, "value_max"): 0.02 for c in CAMERAS},
    **{O.metric(c, "opp_dist"): 0.01 for c in CAMERAS},
    **{O.metric(c, "row_corr"): 0.02 for c in CAMERAS}, **{O.metric(c, "col_corr"): 0.02 for c in CAMERAS},
    O.M_OPEN_LOOP_RATIO: 0.1,
    A.M_RANGE: 0.02, A.M_GRIPPER_SIGN: 0.05, A.M_GRIPPER_TRACK: 0.02, A.M_JOINT_TRACK: 0.02,
    T.M_CLEAN_DELAY: 0.25, T.M_MODE_GAP: 0.25,
    S.M_BLACK: 0.05, S.M_SATURATED: 0.01, S.M_MISSING_TEXTURE: 0.01, S.M_NAN: 0.01,
    S.M_PENETRATION: 0.002, S.M_REST_SHIFT: 0.005,
}


@dataclass(frozen=True)
class Faults:
    """Each field plants one fault; the default is a healthy episode."""
    # policy-input (observation side)
    swap_rb: bool = False            # R and B channels swapped
    vflip: bool = False              # image flipped top to bottom
    hflip: bool = False
    rot180: bool = False
    unnormalized: bool = False       # values left in 0..255
    wrong_resolution: bool = False
    # action side
    gripper_inverted: bool = False   # gripper command has the wrong sign
    gripper_stuck: bool = False      # threshold bug: command frozen closed
    action_scale: float = 1.0        # raw output scaled out of the training range
    drop_dim: bool = False           # raw output one dimension short
    gain: float = 0.6                # joint tracking gain per step (0.6 is healthy)
    # timing
    extra_delay: int = 0             # constant extra delay, any mode (code latency)
    server_delay_max: int = 0        # extra delay uniform in 0..max, incident mode only (server load)
    # render / physics
    black_render: bool = False
    saturated_render: bool = False
    missing_texture: bool = False
    nan_state: bool = False
    interpenetration: bool = False
    rest_shift: bool = False


# --------------------------------------------------------------------------- #
# one episode
# --------------------------------------------------------------------------- #

def _policy_frame(rng: random.Random, cam: str, f: Faults) -> FrameStats:
    b = _BASE[cam]
    means = [m + rng.gauss(0, 0.01) for m in b["means"]]
    if f.swap_rb:
        means[0], means[2] = means[2], means[0]
    rows = [x + rng.gauss(0, 0.01) for x in b["rows"]]
    cols = [x + rng.gauss(0, 0.01) for x in b["cols"]]
    if f.vflip or f.rot180:
        rows.reverse()
    if f.hflip or f.rot180:
        cols.reverse()
    lo, hi = abs(rng.gauss(0, 0.002)), 1.0 - abs(rng.gauss(0, 0.002))
    if f.unnormalized:
        lo, hi = lo * 255.0, hi * 255.0
    shape = (240, 320, 3) if f.wrong_resolution else (224, 224, 3)
    return FrameStats(shape, means, lo, hi, rows, cols, saturated_fraction=abs(rng.gauss(0.002, 0.001)))


def _render_frame(rng: random.Random, cam: str, f: Faults) -> FrameStats:
    means = [m + rng.gauss(0, 0.01) for m in _BASE[cam]["means"]]
    sat = abs(rng.gauss(0.002, 0.001))
    mx = 1.0
    if f.black_render:
        means, mx = [0.0, 0.0, 0.0], 0.0
    if f.saturated_render:
        sat = 0.35 + abs(rng.gauss(0, 0.02))
    return FrameStats((480, 640, 3), means, 0.0, mx, saturated_fraction=sat,
                      missing_texture_fraction=0.2 if f.missing_texture else 0.0)


def _grip(t: int, steps: int) -> float:
    """1 while the gripper should be closed, with short ramps."""
    a, b = int(0.35 * steps), int(0.7 * steps)
    up = min(1.0, max(0.0, (t - a) / 6.0))
    down = min(1.0, max(0.0, (t - b) / 6.0))
    return up - down


def make_trace(seed: int, mode: Mode = Mode.CLEAN, faults: Faults = Faults(), *, steps: int = STEPS,
               code_key: Optional[str] = None) -> EpisodeTrace:
    mode = Mode(mode)
    rng = random.Random(f"{seed}:{mode.value}")
    f = faults

    frames = [FrameSample(s, cam, _policy_frame(rng, cam, f), _render_frame(rng, cam, f))
              for s in range(0, steps, steps // 6) for cam in CAMERAS]

    phases = [rng.uniform(0, 2 * math.pi) for _ in range(N_JOINTS)]
    raw, jcmd, jpos, gcmd, gwid = [], [], [], [], []
    q = [0.8 * math.sin(p) for p in phases]
    w = 0.0
    for t in range(steps):
        cmd = [0.8 * math.sin(0.07 * t + p) for p in phases]
        g = _grip(t, steps)
        a = [(0.5 * math.sin(0.07 * t + p) + rng.gauss(0, 0.02)) * f.action_scale for p in phases]
        a.append((g + rng.gauss(0, 0.01)) * f.action_scale)
        raw.append(a[:-1] if f.drop_dim else a)
        if f.gripper_stuck:
            width_cmd = 0.0
        else:
            width_cmd = GRIPPER_OPEN * (g if f.gripper_inverted else 1.0 - g) + rng.gauss(0, 0.0005)
        if t == 0:
            w = width_cmd
        jcmd.append(cmd)
        jpos.append(list(q))
        gcmd.append(width_cmd)
        gwid.append(w)
        q = [qi + f.gain * (ci - qi) + rng.gauss(0, 0.002) for qi, ci in zip(q, cmd)]
        w = w + 0.5 * (width_cmd - w) + rng.gauss(0, 0.0003)

    pairs = []
    for t0 in range(0, steps, CHUNK):
        for k in range(CHUNK):
            server = rng.randint(0, f.server_delay_max) if (mode is Mode.INCIDENT and f.server_delay_max) else 0
            pairs.append(StepPair(t0, t0 + k + BASE_LATENCY + f.extra_delay + server, k))

    start = (0.40, 0.0, 0.02)
    rest = (start[0] + (0.12 if f.rest_shift else rng.gauss(0, 0.0005)), start[1] + rng.gauss(0, 0.0005), start[2])
    sim = SimHealth(nan_fraction=0.4 if f.nan_state else 0.0,
                    max_penetration=0.03 if f.interpenetration else 0.001 + abs(rng.gauss(0, 0.0003)),
                    object_initial_pos=start, object_rest_pos=rest)
    return EpisodeTrace(seed, mode, tuple(frames), tuple(pairs), raw, jcmd, jpos, gcmd, gwid, sim, code_key)


def default_code_key() -> str:
    """The code key of ``make_incident()``'s code: what ``make_traces`` / ``make_source`` stamp by default."""
    return CodeRef("c0", make_eval_config()).key()


def make_traces(n: int, mode: Mode = Mode.CLEAN, faults: Faults = Faults(), *, base_seed: int = 0,
                code_key: Optional[str] = None) -> list[EpisodeTrace]:
    key = default_code_key() if code_key is None else code_key
    return [make_trace(base_seed + i, mode, faults, code_key=key) for i in range(n)]


def make_source(clean: Faults = Faults(), incident: Optional[Faults] = None, *, n: int = 8,
                code_key: Optional[str] = None) -> InMemoryTraceSource:
    """Traces for one incident under refs ``"clean"`` and ``"incident"``."""
    incident = clean if incident is None else incident
    return InMemoryTraceSource({"clean": make_traces(n, Mode.CLEAN, clean, code_key=code_key),
                                "incident": make_traces(n, Mode.INCIDENT, incident, code_key=code_key)})


F = TypeVar("F")


def faults_for(incident: Incident, by_key: Mapping[str, F], file_faults: Sequence[tuple[Callable, F]],
               workspaces, default: Optional[F]) -> Optional[F]:
    """The planted faults of the code under test: by code key, then by the first file predicate that
    holds on ``workspaces.view(code_ref)``, then ``default``. Shared with the behavior source."""
    code = incident.code_ref
    if code.key() in by_key:
        return by_key[code.key()]
    if file_faults:
        root = workspaces.view(code)
        for pred, f in file_faults:
            if pred(root):
                return f
    return default


class CodeKeyedTraceSource(TraceSource):
    """Traces that depend on the code under test, so a fix makes its planted
    fault disappear: the episodes of ``incident.failing_seeds`` are generated
    with ``faults[incident.code_ref.key()]`` (``default`` for a code not
    listed). ``Faults.server_delay_max`` only acts in incident mode, so one
    record covers both modes.

    Patch-aware (Fix F2a): ``file_faults`` is a list of (file predicate,
    faults), e.g. ``(FileContains("controller/wrapper.py", "normalize"),
    Faults())``, evaluated on ``workspaces.view(incident.code_ref)`` (the same
    predicates as ``MockRule.files``), so a patch from the real repair agent,
    whose name is not known in advance, gets logs. Lookup order: the code key,
    then the first matching predicate, then ``default``.

    Missing logs raise ``TraceUnavailable``, as in ``InMemoryTraceSource``:
    the incident has no ``log_refs[mode]``, or the code matches nothing and
    ``default`` is None.
    """

    def __init__(self, faults: Mapping[str, Faults], default: Optional[Faults] = Faults(), *,
                 file_faults: Sequence[tuple[Callable, Faults]] = (), workspaces=None) -> None:
        if file_faults and workspaces is None:
            raise ValueError("file_faults need workspaces to look at the code's files")
        self.faults, self.default = dict(faults), default
        self.file_faults, self.workspaces = tuple(file_faults), workspaces
        self._made: dict[tuple, list[EpisodeTrace]] = {}

    def _load(self, incident: Incident, mode: Mode) -> list[EpisodeTrace]:
        mode, code = Mode(mode), incident.code_ref.key()
        if mode.value not in incident.log_refs:
            raise TraceUnavailable(f"incident {incident.incident_id} has no log_refs[{mode.value!r}]")
        f = faults_for(incident, self.faults, self.file_faults, self.workspaces, self.default)
        if f is None:
            raise TraceUnavailable(f"no {mode.value}-mode traces logged for code {code}")
        key = (code, mode, f, incident.failing_seeds)
        if key not in self._made:
            self._made[key] = [make_trace(s, mode, f, code_key=code) for s in incident.failing_seeds]
        return list(self._made[key])


# --------------------------------------------------------------------------- #
# nominal statistics
class NominalProbe(O.OpenLoopProbe):
    """An open-loop probe that finds the current preprocessing as good as the reference one (ratio 1.0,
    the synthetic nominal). The check's absence is a partial skip on ``contract_violation``, so fixtures
    that expect a clean obs contract pass this."""

    def action_error(self, incident, path):
        return 0.1


# --------------------------------------------------------------------------- #

def reference_vectors(traces: Sequence[EpisodeTrace], cameras: Sequence[str]) -> dict[str, tuple[float, ...]]:
    """Nominal opponent / profile vectors of each camera, from healthy traces."""
    out = {}
    for cam in cameras:
        fr = [fs.policy_input for t in traces for fs in t.frames if fs.camera == cam]
        out[O.metric(cam, "opp")] = tuple(fmean(c) for c in zip(*(O.opponent(f.channel_means) for f in fr)))
        out[O.metric(cam, "row_profile")] = tuple(fmean(c) for c in zip(*(f.row_profile for f in fr)))
        out[O.metric(cam, "col_profile")] = tuple(fmean(c) for c in zip(*(f.col_profile for f in fr)))
    return out


def make_nominal(*, n_episodes: int = 30, n_gaps: int = 16, gap_episodes: int = 6, fire_z: float = 4.0) -> NominalStats:
    spec = make_training_spec()
    clean = make_traces(n_episodes, Mode.CLEAN, base_seed=10_000)
    vectors = reference_vectors(clean, CAMERAS)
    samples: dict[str, list[float]] = {}
    for part in (O.observation_metrics(clean, CAMERAS, vectors), A.action_metrics(clean, A.action_space(spec)),
                 S.sanity_metrics(clean)):
        samples.update(part)
    samples[T.M_CLEAN_DELAY] = T.episode_mean_delays(clean)
    samples[T.M_MODE_GAP] = [
        T.mode_gap(make_traces(gap_episodes, Mode.CLEAN, base_seed=20_000 + 100 * k),
                   make_traces(gap_episodes, Mode.INCIDENT, base_seed=30_000 + 100 * k)) for k in range(n_gaps)]
    rng = random.Random("open-loop")
    samples[O.M_OPEN_LOOP_RATIO] = [1.0 + rng.gauss(0, 0.03) for _ in range(n_episodes)]
    return NominalStats.from_samples(samples, vectors=vectors, fire_z=fire_z, min_std=FLOORS)


# --------------------------------------------------------------------------- #
# training spec, eval config, incident
# --------------------------------------------------------------------------- #

def make_training_spec() -> TrainingSpec:
    return TrainingSpec(
        policy_name="synthetic_vla", image_size=(224, 224), channel_order="RGB",
        norm_stats={"action": {"min": [-1.0] * ACTION_DIMS, "max": [1.0] * ACTION_DIMS,
                               "mean": [0.0] * ACTION_DIMS, "std": [0.5] * ACTION_DIMS}},
        camera_poses={"front": [0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0], "wrist": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]},
        action_horizon=16, replan_steps=8, denoising_steps=4, control_hz=20.0,
        render_settings={"samples": 64, "denoise": True},
        action_layout={"joints": [0, 7], "gripper": [7, 8]},
        instruction_templates={"pick": "pick up the {object}"})


CONFIG_KEYS: Mapping[str, str] = {
    "action_horizon": "policy.chunk", "replan_steps": "policy.replan", "denoising_steps": "policy.denoise",
    "control_hz": "env.hz", "image_size": "env.image_size", "camera_poses": "env.cameras",
    "render_settings": "env.render", "norm_stats": "norm_stats", "instruction_templates": "env.instruction"}


def make_config_mapping(**overrides) -> ConfigMapping:
    return ConfigMapping(keys=CONFIG_KEYS, **overrides)


def make_eval_config() -> dict:
    """An eval config that matches ``make_training_spec`` exactly."""
    s = make_training_spec()
    cfg = {"policy": {"chunk": s.action_horizon, "replan": s.replan_steps, "denoise": s.denoising_steps},
           "env": {"hz": s.control_hz, "image_size": list(s.image_size), "cameras": s.camera_poses,
                   "render": s.render_settings, "instruction": "pick up the mug"},
           "norm_stats": s.norm_stats}
    return json.loads(json.dumps(cfg))


def make_incident(*, eval_config: Optional[dict] = None, scene: Optional[SceneSpec] = None,
                  observed_mode: Mode = Mode.INCIDENT, log_refs: Optional[Mapping[str, str]] = None,
                  training_spec: Optional[TrainingSpec] = None) -> Incident:
    return Incident(
        incident_id="inc-synthetic",
        scene=scene or SceneSpec("pick_mug", (Factor("lighting", "dark_01"),)),
        training_spec=training_spec or make_training_spec(),
        code_ref=CodeRef("c0", make_eval_config() if eval_config is None else eval_config),
        policy=ActorRef(ActorKind.POLICY, "ckpt_v1"), failing_seeds=tuple(range(10)),
        observed_mode=observed_mode, log_refs={"clean": "clean", "incident": "incident"} if log_refs is None else log_refs)
