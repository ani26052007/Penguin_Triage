"""Rollout instrumentation records and the trace source (plan.md 3.2.2).

The plan lists what a rollout must log ("rollout instrumentation") but leaves
the format to the real backend (``EpisodeResult.logs`` is an opaque mapping).
Stage 1 checks need typed records, so this module defines a minimal one.

Design points
-------------
* Frames are stored as summary statistics, not tensors (``FrameStats``). The
  logger computes them once; the checks never touch pixels. That keeps traces
  small and lets synthetic traces stand in for real ones.
* Each sampled frame carries two stat records: ``render`` (the simulator's
  camera output, before preprocessing) and ``policy_input`` (what the policy
  receives, after preprocessing). The render/physics sanity check reads
  ``render`` and the observation contract reads ``policy_input``. If both read
  the same frame, a normalisation bug (code) would also look like a saturated
  render (sim/asset), and row 2 of the decision table would win over row 4.
* Per-episode series are plain tuples so a trace round-trips through JSON
  (``to_dict`` / ``from_dict``) and can be stored in ``EpisodeResult.logs``.

TODO(verify): how the real backend logs each field and in which units / layout
(image layout CHW vs HWC, pixel scale, how penetration depth, NaN fraction and
the chunk offset are obtained from Isaac Lab and the policy wrapper).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from router.schemas import Incident, Mode

# ``Incident.log_refs`` keys: one reference per inference mode.
LOG_REF_CLEAN = Mode.CLEAN.value
LOG_REF_INCIDENT = Mode.INCIDENT.value


def _floats(xs: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) for x in xs)


def _rows(xs: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(_floats(r) for r in xs)


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FrameStats:
    """Summary of one image tensor.

    shape:                    (height, width, channels), already in HWC order.
    channel_means:            one mean per channel, in the tensor's channel order.
    row_profile / col_profile: mean intensity per row / column bucket (the same
                              number of buckets for every resolution, so a
                              profile can be compared across resolutions).
                              Used for orientation (flip / rotation) checks.
    saturated_fraction:       share of pixels at the maximum value.
    missing_texture_fraction: share of pixels in the engine's missing-texture
                              colour. TODO(verify): colour / detection in Isaac.
    """
    shape: tuple[int, int, int]
    channel_means: tuple[float, ...]
    min_value: float
    max_value: float
    row_profile: tuple[float, ...] = ()
    col_profile: tuple[float, ...] = ()
    saturated_fraction: float = 0.0
    missing_texture_fraction: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(x) for x in self.shape))
        object.__setattr__(self, "channel_means", _floats(self.channel_means))
        object.__setattr__(self, "row_profile", _floats(self.row_profile))
        object.__setattr__(self, "col_profile", _floats(self.col_profile))
        if len(self.shape) != 3:
            raise ValueError(f"shape must be (height, width, channels), got {self.shape}")
        if len(self.channel_means) != self.shape[2]:
            raise ValueError(f"{len(self.channel_means)} channel means for {self.shape[2]} channels")

    def to_dict(self) -> dict:
        return {"shape": list(self.shape), "channel_means": list(self.channel_means),
                "min_value": self.min_value, "max_value": self.max_value,
                "row_profile": list(self.row_profile), "col_profile": list(self.col_profile),
                "saturated_fraction": self.saturated_fraction,
                "missing_texture_fraction": self.missing_texture_fraction}

    @classmethod
    def from_dict(cls, d: Mapping) -> "FrameStats":
        return cls(shape=d["shape"], channel_means=d["channel_means"], min_value=float(d["min_value"]),
                   max_value=float(d["max_value"]), row_profile=d.get("row_profile", ()),
                   col_profile=d.get("col_profile", ()),
                   saturated_fraction=float(d.get("saturated_fraction", 0.0)),
                   missing_texture_fraction=float(d.get("missing_texture_fraction", 0.0)))


@dataclass(frozen=True)
class FrameSample:
    """One sampled step: the camera's render stats and the policy-input stats."""
    step: int
    camera: str
    policy_input: FrameStats
    render: Optional[FrameStats] = None

    def to_dict(self) -> dict:
        return {"step": self.step, "camera": self.camera, "policy_input": self.policy_input.to_dict(),
                "render": self.render.to_dict() if self.render else None}

    @classmethod
    def from_dict(cls, d: Mapping) -> "FrameSample":
        return cls(step=int(d["step"]), camera=d["camera"], policy_input=FrameStats.from_dict(d["policy_input"]),
                   render=FrameStats.from_dict(d["render"]) if d.get("render") else None)


# --------------------------------------------------------------------------- #
# actions, timing, simulator health
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StepPair:
    """For one applied action: the step of the observation it was computed from
    and the step it was applied at.

    chunk_offset is the action's position inside its chunk (0 for a policy that
    does not chunk). Action chunking makes ``applied - obs`` grow along the
    chunk even in a healthy system; subtracting the offset leaves the delay
    that is not explained by chunking (``extra_delay``).
    """
    obs_step: int
    applied_step: int
    chunk_offset: int = 0

    def __post_init__(self) -> None:
        if self.chunk_offset < 0 or self.applied_step - self.obs_step - self.chunk_offset < 0:
            raise ValueError(f"action applied before its observation allows: {self}")

    @property
    def extra_delay(self) -> int:
        return self.applied_step - self.obs_step - self.chunk_offset

    def to_dict(self) -> dict:
        return {"obs_step": self.obs_step, "applied_step": self.applied_step, "chunk_offset": self.chunk_offset}

    @classmethod
    def from_dict(cls, d: Mapping) -> "StepPair":
        return cls(int(d["obs_step"]), int(d["applied_step"]), int(d.get("chunk_offset", 0)))


@dataclass(frozen=True)
class SimHealth:
    """Per-episode simulator health, summarised by the logger.

    nan_fraction:      share of logged steps whose state held NaN / inf.
    max_penetration:   deepest contact penetration seen, in metres.
    object_initial_pos / object_rest_pos: the target object's position as
                       logged at reset and after it has settled (before the
                       robot touches it).
    """
    nan_fraction: float = 0.0
    max_penetration: float = 0.0
    object_initial_pos: tuple[float, ...] = ()
    object_rest_pos: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_initial_pos", _floats(self.object_initial_pos))
        object.__setattr__(self, "object_rest_pos", _floats(self.object_rest_pos))

    def to_dict(self) -> dict:
        return {"nan_fraction": self.nan_fraction, "max_penetration": self.max_penetration,
                "object_initial_pos": list(self.object_initial_pos), "object_rest_pos": list(self.object_rest_pos)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "SimHealth":
        return cls(float(d.get("nan_fraction", 0.0)), float(d.get("max_penetration", 0.0)),
                   d.get("object_initial_pos", ()), d.get("object_rest_pos", ()))


@dataclass(frozen=True)
class EpisodeTrace:
    """Everything Stage 1 reads from one episode (plan 3.2.2 instrumentation).

    raw_actions:        the policy's raw output vector at each applied action.
    joint_cmd/joint_pos: commanded vs tracked joint positions, one row per step.
    gripper_cmd_width/gripper_width: commanded vs actual gripper width per step.
    code_key:           ``CodeRef.key()`` of the code (and eval config) the episode ran with.
                        ``TraceSource.load`` refuses traces whose key is not the code under test.
    """
    seed: int
    mode: Mode
    frames: tuple[FrameSample, ...] = ()
    step_pairs: tuple[StepPair, ...] = ()
    raw_actions: tuple[tuple[float, ...], ...] = ()
    joint_cmd: tuple[tuple[float, ...], ...] = ()
    joint_pos: tuple[tuple[float, ...], ...] = ()
    gripper_cmd_width: tuple[float, ...] = ()
    gripper_width: tuple[float, ...] = ()
    sim: Optional[SimHealth] = None
    code_key: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", Mode(self.mode))
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "step_pairs", tuple(self.step_pairs))
        object.__setattr__(self, "raw_actions", _rows(self.raw_actions))
        object.__setattr__(self, "joint_cmd", _rows(self.joint_cmd))
        object.__setattr__(self, "joint_pos", _rows(self.joint_pos))
        object.__setattr__(self, "gripper_cmd_width", _floats(self.gripper_cmd_width))
        object.__setattr__(self, "gripper_width", _floats(self.gripper_width))
        if len(self.joint_cmd) != len(self.joint_pos):
            raise ValueError("joint_cmd and joint_pos must have the same number of steps")
        if len(self.gripper_cmd_width) != len(self.gripper_width):
            raise ValueError("gripper_cmd_width and gripper_width must have the same number of steps")

    def to_dict(self) -> dict:
        return {"seed": self.seed, "mode": self.mode.value,
                "frames": [f.to_dict() for f in self.frames],
                "step_pairs": [p.to_dict() for p in self.step_pairs],
                "raw_actions": [list(r) for r in self.raw_actions],
                "joint_cmd": [list(r) for r in self.joint_cmd], "joint_pos": [list(r) for r in self.joint_pos],
                "gripper_cmd_width": list(self.gripper_cmd_width), "gripper_width": list(self.gripper_width),
                "sim": self.sim.to_dict() if self.sim else None, "code_key": self.code_key}

    @classmethod
    def from_dict(cls, d: Mapping) -> "EpisodeTrace":
        return cls(seed=int(d["seed"]), mode=Mode(d["mode"]),
                   frames=tuple(FrameSample.from_dict(f) for f in d.get("frames", ())),
                   step_pairs=tuple(StepPair.from_dict(p) for p in d.get("step_pairs", ())),
                   raw_actions=d.get("raw_actions", ()), joint_cmd=d.get("joint_cmd", ()),
                   joint_pos=d.get("joint_pos", ()), gripper_cmd_width=d.get("gripper_cmd_width", ()),
                   gripper_width=d.get("gripper_width", ()),
                   sim=SimHealth.from_dict(d["sim"]) if d.get("sim") else None, code_key=d.get("code_key"))


# --------------------------------------------------------------------------- #
# trace source
# --------------------------------------------------------------------------- #

class TraceUnavailable(RuntimeError):
    """The traces a check needs cannot be found. This is not "checked, nothing
    found": a missing log must never read as a clean result."""


def verify_code_key(traces: Sequence, incident: Incident, what: str) -> None:
    """Raise ``TraceUnavailable`` unless every trace was logged with the code under test
    (``incident.code_ref.key()``). Logs of other code, or logs that do not say which code they
    come from, cannot stand for this incident: after a patch they would keep reporting the fault
    that was fixed (ISSUES #52)."""
    want = incident.code_ref.key()
    keys = {t.code_key for t in traces}
    if keys == {want}:
        return
    if None in keys:
        raise TraceUnavailable(f"{what} carry no code key, so they cannot be tied to the code under test ({want})")
    raise TraceUnavailable(f"{what} were logged for code {sorted(keys)}, not for the code under test ({want})")


class TraceSource(ABC):
    """Resolves an incident's log references into typed traces.

    Template method: ``load`` is the public call and checks the code key of what ``_load``
    returns; a subclass implements ``_load`` and never overrides ``load`` (a structural test
    enforces it)."""

    def load(self, incident: Incident, mode: Mode) -> list[EpisodeTrace]:
        """Traces of the incident's episodes logged in ``mode``, all logged with
        ``incident.code_ref``. Raises ``TraceUnavailable`` if there are none or they belong to
        other code."""
        traces = self._load(incident, mode)
        verify_code_key(traces, incident, f"{Mode(mode).value}-mode traces")
        return traces

    @abstractmethod
    def _load(self, incident: Incident, mode: Mode) -> list[EpisodeTrace]:
        """The stored traces for ``incident`` in ``mode``. Raises ``TraceUnavailable`` if there
        are none. Each trace carries its ``code_key``.
        TODO(verify): the real reference format (path, URI, cache key) and
        reading it back from ``EpisodeResult.logs`` or the log store.
        """


class InMemoryTraceSource(TraceSource):
    """Traces held in a dict, addressed by ``incident.log_refs[mode.value]``.
    Used by the mock backend and the tests."""

    def __init__(self, by_ref: Mapping[str, Sequence[EpisodeTrace]]) -> None:
        self._by_ref = {k: list(v) for k, v in by_ref.items()}

    def _load(self, incident: Incident, mode: Mode) -> list[EpisodeTrace]:
        mode = Mode(mode)
        ref = incident.log_refs.get(mode.value)
        if ref is None:
            raise TraceUnavailable(f"incident {incident.incident_id} has no log_refs[{mode.value!r}]")
        traces = self._by_ref.get(ref)
        if not traces:
            raise TraceUnavailable(f"no traces stored under {ref!r} ({mode.value} mode)")
        wrong = [t.seed for t in traces if t.mode is not mode]
        if wrong:
            raise ValueError(f"traces under {ref!r} are not all {mode.value}-mode (seeds {wrong})")
        return list(traces)
