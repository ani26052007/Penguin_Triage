"""Subtask events of one episode.

Reach and grasp are computed offline from the trajectories (no Arena ``TrackedEventCfg``):

* **reach**: the first step at which an end effector is within ``reach_distance_m`` of the object;
* **grasp**: the first step at which a hand's closure crosses ``closure_threshold`` and the object then
  moves with that hand (``GraspRule``). Its ``GraspRecord`` is that hand's position and the object's
  at that step.

Every other subtask (lift, place, and composite extras such as closing the door) is complete when its
predicate appears in Arena's ``progress.events`` (``traces.subtasks``, VERIFY(V-progress)).
``phase_map`` turns the task's subtask names into router phases, in router phase order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from router.prereq.behavior_trace import PHASES, GraspRecord, SubtaskProgress

from vla_isaac.traces.layout import EventMatch, GraspRule


@dataclass(frozen=True)
class Grasp:
    step: int
    hand: str
    record: GraspRecord


def reach_step(ee: Mapping[str, Sequence[Sequence[float]]], obj: Sequence[Sequence[float]],
               distance_m: float) -> Optional[int]:
    """First step at which any hand is within ``distance_m`` of the object."""
    n = min([len(obj)] + [len(p) for p in ee.values()])
    for t in range(n):
        if any(math.dist(p[t][:3], obj[t][:3]) <= distance_m for p in ee.values()):
            return t
    return None


def grasp_event(closures: Mapping[str, Sequence[float]], ee: Mapping[str, Sequence[Sequence[float]]],
                obj: Sequence[Sequence[float]], rule: GraspRule) -> Optional[Grasp]:
    """First closure crossing (in step order; hands in config order within a step) after which the
    object moves with the hand. A crossing whose window runs past the episode end does not count."""
    n = min([len(obj)] + [len(c) for c in closures.values()] + [len(p) for p in ee.values()])
    for t in range(n):
        for hand, c in closures.items():
            if c[t] < rule.closure_threshold or (t > 0 and c[t - 1] >= rule.closure_threshold):
                continue
            end = t + rule.window
            if end >= n:
                continue
            hand_pos = ee[hand]
            moved = math.dist(obj[end][:3], obj[t][:3])
            relative = abs(math.dist(hand_pos[end][:3], obj[end][:3]) - math.dist(hand_pos[t][:3], obj[t][:3]))
            if moved >= rule.min_object_motion_m and relative <= rule.max_relative_motion_m:
                return Grasp(t, hand, GraspRecord(t, tuple(hand_pos[t][:3]), tuple(obj[t][:3])))
    return None


def fired(events: Sequence[Mapping[str, Any]], match: EventMatch) -> Optional[int]:
    """Episode step of the first ``progress.events`` entry for ``match``; None if it never fired."""
    steps = [int(e["step"]) for e in events
             if e.get("criteria_name") == match.criteria_name and int(e.get("predicate_index", -1)) == match.predicate_index]
    return min(steps) if steps else None


def subtask_progress(completed: Mapping[str, bool], phase_map: Mapping[str, str]) -> tuple[SubtaskProgress, ...]:
    """Router ``SubtaskProgress`` records for the task's subtasks, in router phase order."""
    by_phase = {phase_map[name]: done for name, done in completed.items()}
    return tuple(SubtaskProgress(p, by_phase[p]) for p in PHASES if p in by_phase)
