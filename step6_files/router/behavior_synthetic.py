"""Synthetic behavior traces for C_behavior (plan.md 3.2.14: test without Isaac).

The mock side of ``router.behavior``, in the style of
``router.stage1.synthetic``: an episode is generated from a seed and a
``BehaviorFaults`` record; ``BehaviorFaults()`` is a healthy grasp, and each
field plants one known fault. Deterministic: ``random.Random`` seeded from
(seed, mode).

``FLOORS`` are test settings for these synthetic traces, not calibrated values.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from router.behavior import M_BIAS, M_SCATTER, M_WRONG, behavior_metrics
from router.prereq.behavior_trace import (PHASES, BehaviorTrace, GraspRecord, InMemoryBehaviorSource,
                                          SubtaskProgress)
from router.schemas import Mode
from router.stage1.nominal import NominalStats

FLOORS: Mapping[str, float] = {M_BIAS: 0.003, M_SCATTER: 0.003, M_WRONG: 0.05}
GRASP_NOISE = 0.01                       # healthy per-axis scatter of the gripper-close position (m)
TARGET = (0.40, 0.10, 0.02)
DISTRACTOR = (0.65, 0.10, 0.02)


@dataclass(frozen=True)
class BehaviorFaults:
    offset: tuple[float, ...] = ()       # consistent shift of the end effector (frame / extrinsic bug)
    scatter: float = 0.0                 # extra per-axis noise (perception failure)
    wrong_object: float = 0.0            # probability of closing on the distractor
    fail_phase: Optional[str] = None     # failing episodes stop at this subtask; None = no subtask log
    no_grasp: bool = False               # the gripper never closes


def make_behavior_trace(seed: int, mode: Mode = Mode.CLEAN, faults: BehaviorFaults = BehaviorFaults(), *,
                        success: bool = False) -> BehaviorTrace:
    rng = random.Random(f"behavior|{seed}|{Mode(mode).value}")
    grasp = None
    if not faults.no_grasp:
        goal = DISTRACTOR if rng.random() < faults.wrong_object else TARGET
        sd = GRASP_NOISE + faults.scatter
        shift = faults.offset or (0.0,) * 3
        ee = tuple(g + s + rng.gauss(0, sd) for g, s in zip(goal, shift))
        grasp = GraspRecord(step=40, ee_pos=ee, target_pos=TARGET, distractor_pos=(DISTRACTOR,))
    subtasks: tuple[SubtaskProgress, ...] = ()
    if faults.fail_phase is not None:
        stop = PHASES.index(faults.fail_phase)
        subtasks = tuple(SubtaskProgress(p, success or i < stop) for i, p in enumerate(PHASES))
    return BehaviorTrace(seed, mode, success, grasp, subtasks)


def make_behavior_traces(n: int, mode: Mode = Mode.CLEAN, faults: BehaviorFaults = BehaviorFaults(), *,
                         base_seed: int = 0, success: bool = False) -> list[BehaviorTrace]:
    return [make_behavior_trace(base_seed + i, mode, faults, success=success) for i in range(n)]


def make_behavior_source(clean: BehaviorFaults = BehaviorFaults(), incident: Optional[BehaviorFaults] = None, *,
                         n: int = 8) -> InMemoryBehaviorSource:
    """Failing episodes under refs ``"clean"`` and ``"incident"``."""
    incident = clean if incident is None else incident
    return InMemoryBehaviorSource({"clean": make_behavior_traces(n, Mode.CLEAN, clean),
                                   "incident": make_behavior_traces(n, Mode.INCIDENT, incident)})


def make_behavior_nominal(*, group_size: int = 8, n_groups: int = 40, fire_z: float = 4.0) -> NominalStats:
    """Group-level nominal built from healthy grasps, with the module's own metric function."""
    samples: dict[str, list[float]] = {M_BIAS: [], M_SCATTER: [], M_WRONG: []}
    for k in range(n_groups):
        m = behavior_metrics(make_behavior_traces(group_size, base_seed=50_000 + 100 * k))
        for name in samples:
            samples[name].append(m[name])
    return NominalStats.from_samples(samples, fire_z=fire_z, min_std=FLOORS)
