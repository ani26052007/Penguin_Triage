"""Subtask events of one episode.

Reach and grasp are computed offline from the trajectories (no Arena ``TrackedEventCfg``):

* **reach**: the first step at which an end effector is within ``reach_distance_m`` of the object;
* **grasp**: the first closure crossing, in step order, of a hand that is within ``reach_distance_m``
  of the object at that step, after which the object moves with that hand (``GraspRule``). Two hands
  crossing at the same step are taken nearer hand first. Its ``GraspRecord`` is that hand's position
  and the object's at that step.

The **grasping hand** is chosen per episode (``choose_hand``), never fixed in config: the hand of the
grasp; without one, the hand of the first closure crossing within reach (an attempt that did not move
the object); without that, the hand that came nearest to the object (no grasp at all). The choice and
its basis go into the behaviour trace's notes, and that hand's closure is the episode's gripper width.

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

GRASP, ATTEMPT, NEAREST = "grasp", "attempt", "nearest"


@dataclass(frozen=True)
class Grasp:
    step: int
    hand: str
    record: GraspRecord


@dataclass(frozen=True)
class HandChoice:
    """The episode's grasping hand. ``basis``: ``grasp`` (the hand of the grasp), ``attempt`` (the hand of
    the first closure crossing within reach; the object did not move with it) or ``nearest`` (no closure
    crossing within reach; the hand that came nearest to the object, used only for the gripper width)."""
    hand: str
    basis: str
    grasp: Optional[Grasp] = None

    @property
    def grasp_hand(self) -> Optional[str]:
        """The hand that closed on the object; None when no hand closed within reach."""
        return None if self.basis == NEAREST else self.hand

    def notes(self) -> dict[str, Any]:
        return {"grasp_hand": self.grasp_hand, "width_hand": self.hand, "hand_basis": self.basis}


def reach_step(ee: Mapping[str, Sequence[Sequence[float]]], obj: Sequence[Sequence[float]],
               distance_m: float) -> Optional[int]:
    """First step at which any hand is within ``distance_m`` of the object."""
    n = min([len(obj)] + [len(p) for p in ee.values()])
    for t in range(n):
        if any(math.dist(p[t][:3], obj[t][:3]) <= distance_m for p in ee.values()):
            return t
    return None


def _length(closures: Mapping[str, Sequence[float]], ee: Mapping[str, Sequence[Sequence[float]]],
            obj: Sequence[Sequence[float]]) -> int:
    return min([len(obj)] + [len(c) for c in closures.values()] + [len(p) for p in ee.values()])


def closure_crossings(closures: Mapping[str, Sequence[float]], ee: Mapping[str, Sequence[Sequence[float]]],
                      obj: Sequence[Sequence[float]], rule: GraspRule, reach_distance_m: float
                      ) -> list[tuple[int, str]]:
    """(step, hand) of every closure crossing while that hand is within ``reach_distance_m`` of the
    object, in step order; at one step, nearer hand first (then config order)."""
    n = _length(closures, ee, obj)
    found = []
    for order, (hand, c) in enumerate(closures.items()):
        for t in range(n):
            if c[t] < rule.closure_threshold or (t > 0 and c[t - 1] >= rule.closure_threshold):
                continue
            d = math.dist(ee[hand][t][:3], obj[t][:3])
            if d <= reach_distance_m:
                found.append((t, d, order, hand))
    return [(t, hand) for t, _, _, hand in sorted(found)]


def _moves_with(hand_pos: Sequence[Sequence[float]], obj: Sequence[Sequence[float]], t: int, rule: GraspRule,
                n: int) -> bool:
    end = t + rule.window
    if end >= n:                                # the window runs past the episode end
        return False
    moved = math.dist(obj[end][:3], obj[t][:3])
    relative = abs(math.dist(hand_pos[end][:3], obj[end][:3]) - math.dist(hand_pos[t][:3], obj[t][:3]))
    return moved >= rule.min_object_motion_m and relative <= rule.max_relative_motion_m


def grasp_event(closures: Mapping[str, Sequence[float]], ee: Mapping[str, Sequence[Sequence[float]]],
                obj: Sequence[Sequence[float]], rule: GraspRule, reach_distance_m: float) -> Optional[Grasp]:
    """The first closure crossing within reach (``closure_crossings`` order) after which the object
    moves with the hand; None if there is none."""
    n = _length(closures, ee, obj)
    for t, hand in closure_crossings(closures, ee, obj, rule, reach_distance_m):
        hand_pos = ee[hand]
        if _moves_with(hand_pos, obj, t, rule, n):
            return Grasp(t, hand, GraspRecord(t, tuple(hand_pos[t][:3]), tuple(obj[t][:3])))
    return None


def choose_hand(closures: Mapping[str, Sequence[float]], ee: Mapping[str, Sequence[Sequence[float]]],
                obj: Sequence[Sequence[float]], rule: GraspRule, reach_distance_m: float) -> HandChoice:
    """The episode's grasping hand (module docstring), with the grasp when there is one."""
    grasp = grasp_event(closures, ee, obj, rule, reach_distance_m)
    if grasp is not None:
        return HandChoice(grasp.hand, GRASP, grasp)
    crossings = closure_crossings(closures, ee, obj, rule, reach_distance_m)
    if crossings:
        return HandChoice(crossings[0][1], ATTEMPT)
    n = _length(closures, ee, obj)
    if n == 0:
        raise ValueError("no steps to choose a hand from")
    nearest = min(closures, key=lambda h: min(math.dist(ee[h][t][:3], obj[t][:3]) for t in range(n)))
    return HandChoice(nearest, NEAREST)


def fired(events: Sequence[Mapping[str, Any]], match: EventMatch) -> Optional[int]:
    """Episode step of the first ``progress.events`` entry for ``match``; None if it never fired."""
    steps = [int(e["step"]) for e in events
             if e.get("criteria_name") == match.criteria_name and int(e.get("predicate_index", -1)) == match.predicate_index]
    return min(steps) if steps else None


def subtask_progress(completed: Mapping[str, bool], phase_map: Mapping[str, str]) -> tuple[SubtaskProgress, ...]:
    """Router ``SubtaskProgress`` records for the task's subtasks, in router phase order."""
    by_phase = {phase_map[name]: done for name, done in completed.items()}
    return tuple(SubtaskProgress(p, by_phase[p]) for p in PHASES if p in by_phase)
