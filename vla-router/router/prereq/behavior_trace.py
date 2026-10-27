"""Behavior-analysis records and their trace source (plan.md 3.2.2, 3.2.7 C_behavior).

C_behavior reads two things from the rollout instrumentation (3.2.2):

* "end-effector pose at the moment the gripper closes" together with the true
  pose of the target object: ``GraspRecord``;
* "per-subtask progress (reach, grasp, lift, place)": ``SubtaskProgress``.

They live in their own record (``BehaviorTrace``) and their own source
(``BehaviorSource``), not inside ``EpisodeTrace``: the Stage 1 trace format is
settled and its checks must not change for this step (DECISIONS.md, Step 6).
``BehaviorSource`` mirrors ``TraceSource``: a missing log raises
``TraceUnavailable``, it never reads as "checked, nothing found".

Position only: the plan says "pose", but a consistent frame or extrinsic bug
shows in position, and orientation adds a second unit system without a second
verdict. Orientation is not compared (DECISIONS.md, Step 6).

TODO(verify): how the real backend logs these fields. Open points: the frame
the two positions are expressed in (they must be the same frame; world or
robot base), the unit (metres assumed), how "gripper closes" is detected
(commanded width crossing a threshold, or actual width), whether an episode
that never closes the gripper logs no ``GraspRecord``, and how subtask
completion is decided in Isaac Lab.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from router.prereq.trace import TraceUnavailable, verify_code_key
from router.schemas import Incident, Mode

# Pick-and-place phases, in task order, then the optional phases after them (router change 2). An
# optional phase is judged only when an episode logs it: ``failure_phase`` looks only at the phases
# present in a record, so a pick-only task never names one. ``post_place`` is a composite task's
# subtask after the place (e.g. closing a door); the backend maps its own subtask names onto it.
PICK_PHASES = ("reach", "grasp", "lift", "place")
OPTIONAL_PHASES = ("post_place",)
PHASES = PICK_PHASES + OPTIONAL_PHASES


def _floats(xs: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(x) for x in xs)


@dataclass(frozen=True)
class GraspRecord:
    """The moment the gripper closed: where the end effector was, where the
    true target object was, and where any other graspable objects were.

    ee_pos / target_pos: same frame, same length (metres).
    distractor_pos:      positions of the other objects in the scene; used to
                         tell "wrong object" from "right object, wrong place".
    """
    step: int
    ee_pos: tuple[float, ...]
    target_pos: tuple[float, ...]
    distractor_pos: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "ee_pos", _floats(self.ee_pos))
        object.__setattr__(self, "target_pos", _floats(self.target_pos))
        object.__setattr__(self, "distractor_pos", tuple(_floats(p) for p in self.distractor_pos))
        n = len(self.ee_pos)
        if n == 0 or len(self.target_pos) != n or any(len(p) != n for p in self.distractor_pos):
            raise ValueError("ee_pos, target_pos and distractor_pos must all have the same, non-zero length")

    def to_dict(self) -> dict:
        return {"step": self.step, "ee_pos": list(self.ee_pos), "target_pos": list(self.target_pos),
                "distractor_pos": [list(p) for p in self.distractor_pos]}

    @classmethod
    def from_dict(cls, d: Mapping) -> "GraspRecord":
        return cls(int(d["step"]), d["ee_pos"], d["target_pos"], d.get("distractor_pos", ()))


@dataclass(frozen=True)
class SubtaskProgress:
    """Did the episode complete one subtask (one of ``PHASES``)?"""
    name: str
    completed: bool

    def __post_init__(self) -> None:
        if self.name not in PHASES:
            raise ValueError(f"subtask must be one of {PHASES}, got {self.name!r}")

    def to_dict(self) -> dict:
        return {"name": self.name, "completed": self.completed}

    @classmethod
    def from_dict(cls, d: Mapping) -> "SubtaskProgress":
        return cls(d["name"], bool(d["completed"]))


@dataclass(frozen=True)
class BehaviorTrace:
    """What C_behavior reads from one episode. ``grasp`` is None when the
    gripper never closed."""
    seed: int
    mode: Mode
    success: bool
    grasp: Optional[GraspRecord] = None
    subtasks: tuple[SubtaskProgress, ...] = ()
    code_key: Optional[str] = None       # ``CodeRef.key()`` the episode ran with (see ``BehaviorSource.load``)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", Mode(self.mode))
        object.__setattr__(self, "subtasks", tuple(self.subtasks))
        names = [s.name for s in self.subtasks]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate subtask records in episode {self.seed}: {names}")

    def failure_phase(self) -> Optional[str]:
        """First subtask, in task order, that the episode did not complete.
        None for a successful episode, for an episode with no subtask log, and
        for a failed episode that completed every logged subtask (the failure
        then lies outside the logged phases, so it names none)."""
        if self.success or not self.subtasks:
            return None
        done = {s.name: s.completed for s in self.subtasks}
        for phase in PHASES:
            if phase in done and not done[phase]:
                return phase
        return None

    def to_dict(self) -> dict:
        return {"seed": self.seed, "mode": self.mode.value, "success": self.success,
                "grasp": self.grasp.to_dict() if self.grasp else None,
                "subtasks": [s.to_dict() for s in self.subtasks], "code_key": self.code_key}

    @classmethod
    def from_dict(cls, d: Mapping) -> "BehaviorTrace":
        return cls(int(d["seed"]), Mode(d["mode"]), bool(d["success"]),
                   GraspRecord.from_dict(d["grasp"]) if d.get("grasp") else None,
                   tuple(SubtaskProgress.from_dict(s) for s in d.get("subtasks", ())), d.get("code_key"))


class BehaviorSource(ABC):
    """Resolves an incident's log references into behavior traces. Template method, like
    ``TraceSource``: ``load`` checks the code key, subclasses implement ``_load``."""

    def load(self, incident: Incident, mode: Mode) -> list[BehaviorTrace]:
        """Behavior traces of the incident's episodes logged in ``mode``, all logged with
        ``incident.code_ref``. Raises ``TraceUnavailable`` if there are none or they belong to
        other code."""
        traces = self._load(incident, mode)
        verify_code_key(traces, incident, f"{Mode(mode).value}-mode behavior traces")
        return traces

    @abstractmethod
    def _load(self, incident: Incident, mode: Mode) -> list[BehaviorTrace]:
        """The stored behavior traces for ``incident`` in ``mode``, each with its ``code_key``.

        Raises ``TraceUnavailable`` if there are none.
        TODO(verify): the real reference format and where the real backend
        stores these fields.
        """


class InMemoryBehaviorSource(BehaviorSource):
    """Traces held in a dict, addressed by ``incident.log_refs[mode.value]``
    (the same reference ``InMemoryTraceSource`` uses). Mock backend and tests."""

    def __init__(self, by_ref: Mapping[str, Sequence[BehaviorTrace]]) -> None:
        self._by_ref = {k: list(v) for k, v in by_ref.items()}

    def _load(self, incident: Incident, mode: Mode) -> list[BehaviorTrace]:
        mode = Mode(mode)
        ref = incident.log_refs.get(mode.value)
        if ref is None:
            raise TraceUnavailable(f"incident {incident.incident_id} has no log_refs[{mode.value!r}]")
        traces = self._by_ref.get(ref)
        if not traces:
            raise TraceUnavailable(f"no behavior traces stored under {ref!r} ({mode.value} mode)")
        wrong = [t.seed for t in traces if t.mode is not mode]
        if wrong:
            raise ValueError(f"behavior traces under {ref!r} are not all {mode.value}-mode (seeds {wrong})")
        return list(traces)
