"""The task config's ``traces:`` section: where one episode's evidence is and how it is read.

Two sources, both written by a run that records traces (``IsaacBackend.run(..., record_trajectories=True)``):

* the policy wrapper's instrument files (``policy_wrapper/instrument.py``): policy-input / render frame
  stats, (observation step, applied step) pairs, raw server chunks, and the observation keys listed
  under ``instrument.obs_keys``;
* Arena's HDF5 trajectory dataset: commanded and tracked joints, object poses (``hdf5:`` key paths).

Everything task-specific is here: the HDF5 key paths (Isaac Lab's recorder layout is unread,
VERIFY(V-hdf5-keys)), the joint orders, the object, the end-effector source, the hand-closure joints,
the reach / grasp thresholds (the grasping hand is chosen per episode, ``traces.events.choose_hand``), the progress events that mark subtasks, and the map from the task's
subtask names to the router's phases (``phase_map``, router change 2).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from router.prereq.behavior_trace import PHASES

from vla_isaac.config import ConfigError


def _strict(d: Any, cls, where: str) -> dict:
    if not isinstance(d, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown, missing = set(d) - known, known - set(d)
    if unknown or missing:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return dict(d)


def load_joint_order(path: Path) -> tuple[str, ...]:
    """Joint names in column order from a ``joints: {name: index}`` YAML (Arena's joint-space files)."""
    if not Path(path).is_file():
        raise ConfigError(f"no joint-space file at {path}")
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    joints = data.get("joints")
    if not isinstance(joints, Mapping) or not joints:
        raise ConfigError(f"{path}: no 'joints' mapping")
    order = sorted(joints, key=lambda n: int(joints[n]))
    if [int(joints[n]) for n in order] != list(range(len(order))):
        raise ConfigError(f"{path}: joint indices are not 0..{len(order) - 1}")
    return tuple(order)


@dataclass(frozen=True)
class Hdf5Keys:
    """Key paths inside one demo group. ``{object}`` is replaced by ``TraceLayout.object``."""
    file_glob: str                 # dataset files in the run directory
    demo_group: str                # group holding demo_0, demo_1, ...
    env_id: str                    # Arena EpisodeIdentityRecorder: joins a demo to its JSONL record
    episode_in_env: str
    joint_cmd: str                 # commanded joint targets per step, ``joints.action`` column order
    joint_pos: str                 # tracked joint positions per step, ``joints.state`` column order
    object_pose: str               # per step, position first
    object_initial_pose: str       # at reset


@dataclass(frozen=True)
class EventMatch:
    """A ``progress.events`` entry: the subtask is complete when this predicate fired."""
    criteria_name: str
    predicate_index: int


@dataclass(frozen=True)
class GraspRule:
    """Grasp = a hand's closure crosses ``closure_threshold`` while the hand is within the reach
    distance of the object and, over the next ``window`` steps, the object moves at least
    ``min_object_motion_m`` while its distance to that end effector changes by at most
    ``max_relative_motion_m`` (it moves with the hand)."""
    closure_threshold: float
    window: int
    min_object_motion_m: float
    max_relative_motion_m: float


@dataclass(frozen=True)
class TraceLayout:
    instrument_obs_keys: tuple[str, ...]
    hdf5: Hdf5Keys
    state_joints: tuple[str, ...]
    action_joints: tuple[str, ...]
    object: str
    ee_source: str                                      # instrument | hdf5
    ee_keys: Mapping[str, str]                          # hand -> instrument obs key or HDF5 path
    hands: Mapping[str, Mapping[str, tuple[float, float]]]   # hand -> joint -> (open, closed)
    reach_distance_m: float
    grasp: GraspRule
    subtasks: Mapping[str, EventMatch]                  # subtask completed by a progress event
    object_rest_event: Optional[EventMatch]
    phase_map: Mapping[str, str]                        # task subtask -> router phase
    check_episode_length: bool = True

    OFFLINE = ("reach", "grasp")                        # computed from the trajectories, not from events

    def __post_init__(self) -> None:
        if self.ee_source not in ("instrument", "hdf5"):
            raise ConfigError(f"traces.ee.source must be 'instrument' or 'hdf5', got {self.ee_source!r}")
        if set(self.ee_keys) != set(self.hands):
            raise ConfigError(f"traces: ee keys {sorted(self.ee_keys)} and hands {sorted(self.hands)} must name "
                              f"the same hands")
        for hand, joints in self.hands.items():
            if not joints:
                raise ConfigError(f"traces.hand_closure.hands.{hand} lists no joints")
            for j, (lo, hi) in joints.items():
                if j not in self.state_joints or j not in self.action_joints:
                    raise ConfigError(f"traces.hand_closure: {j} is not both a state and an action joint")
                if lo == hi:
                    raise ConfigError(f"traces.hand_closure: {j} has open == closed")
        missing = [a for a in self.action_joints if a not in self.state_joints]
        if missing:
            raise ConfigError(f"traces.joints: action joints not in the state joints: {missing}")
        names = set(self.OFFLINE) | set(self.subtasks)
        unmapped = names - set(self.phase_map)
        if unmapped:
            raise ConfigError(f"traces.phase_map does not map subtasks {sorted(unmapped)}")
        bad = {k: v for k, v in self.phase_map.items() if v not in PHASES}
        if bad:
            raise ConfigError(f"traces.phase_map targets must be router phases {PHASES}: {bad}")
        if set(self.phase_map) - names:
            raise ConfigError(f"traces.phase_map names unknown subtasks {sorted(set(self.phase_map) - names)}")
        targets = list(self.phase_map.values())
        if len(targets) != len(set(targets)):
            raise ConfigError(f"traces.phase_map maps two subtasks to one phase: {dict(self.phase_map)}")
        if self.ee_source == "instrument":
            absent = set(self.ee_keys.values()) - set(self.instrument_obs_keys)
            if absent:
                raise ConfigError(f"traces: ee keys {sorted(absent)} are not in instrument.obs_keys")

    def hdf5_path(self, key: str) -> str:
        return key.replace("{object}", self.object)


def _match(d: Any, where: str) -> EventMatch:
    d = _strict(d, EventMatch, where)
    return EventMatch(str(d["criteria_name"]), int(d["predicate_index"]))


def layout_from_dict(d: Mapping, base: Path) -> TraceLayout:
    """The ``traces:`` section; relative joint-space paths resolve against ``base``."""
    keys = ("instrument", "hdf5", "joints", "object", "ee", "hand_closure", "reach_distance_m", "grasp",
            "subtasks", "object_rest_event", "phase_map", "check_episode_length")
    if not isinstance(d, Mapping):
        raise ConfigError("task config [traces] must be a mapping")
    unknown, missing = set(d) - set(keys), set(keys) - set(d) - {"object_rest_event", "check_episode_length"}
    if unknown or missing:
        raise ConfigError(f"task config [traces]: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    joints = d["joints"]
    if not isinstance(joints, Mapping) or set(joints) != {"state", "action"}:
        raise ConfigError("traces.joints must have 'state' and 'action' joint-space files")
    ee, hc = d["ee"], d["hand_closure"]
    if not isinstance(ee, Mapping) or set(ee) != {"source", "keys"}:
        raise ConfigError("traces.ee must have 'source' and 'keys'")
    if not isinstance(hc, Mapping) or set(hc) != {"hands"}:
        raise ConfigError("traces.hand_closure must have 'hands' only (the grasping hand is chosen per episode)")
    hands = {str(h): {str(j): (float(v[0]), float(v[1])) for j, v in js.items()} for h, js in hc["hands"].items()}
    inst = d["instrument"]
    if not isinstance(inst, Mapping) or set(inst) != {"obs_keys"}:
        raise ConfigError("traces.instrument must have 'obs_keys'")
    rest = d.get("object_rest_event")
    return TraceLayout(
        instrument_obs_keys=tuple(str(k) for k in inst["obs_keys"]),
        hdf5=Hdf5Keys(**_strict(d["hdf5"], Hdf5Keys, "traces.hdf5")),
        state_joints=load_joint_order((base / joints["state"]).resolve()),
        action_joints=load_joint_order((base / joints["action"]).resolve()),
        object=str(d["object"]), ee_source=str(ee["source"]), ee_keys={str(k): str(v) for k, v in ee["keys"].items()},
        hands=hands, reach_distance_m=float(d["reach_distance_m"]),
        grasp=GraspRule(**{k: float(v) if k != "window" else int(v)
                           for k, v in _strict(d["grasp"], GraspRule, "traces.grasp").items()}),
        subtasks={str(k): _match(v, f"traces.subtasks.{k}") for k, v in d["subtasks"].items()},
        object_rest_event=_match(rest, "traces.object_rest_event") if rest is not None else None,
        phase_map={str(k): str(v) for k, v in d["phase_map"].items()},
        check_episode_length=bool(d.get("check_episode_length", True)))
