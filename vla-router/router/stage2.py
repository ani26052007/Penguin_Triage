"""Stage 2: branch split on the nominal scene with the current code (plan.md 3.2.5).

    within the nominal band (RESTORED)  -> Branch C
    clearly failed (NO_EFFECT)          -> Branch G
    degraded (PARTIAL)                  -> Branch G first, flagged multi-cause
    INCONCLUSIVE after extension        -> branch unknown

Which branches run:
    C          -> C only
    G          -> G only
    degraded   -> G and C. G's code verdict still wins by precedence in
                  ``decide``; the C restores are what let a partial vision
                  restore set "likely second cause" (plan 6.5 "Combined").
    unknown    -> G and C, because ``decide`` then needs both G2 and C1.
"""

from __future__ import annotations

from enum import Enum

from router import vocab as V
from router.replay import ReplayRunner
from router.schemas import Incident, Mode, Outcome, ReplayResult, SceneSpec


class Branch(str, Enum):
    C = "C"
    G = "G"
    G_DEGRADED = "G_degraded"
    UNKNOWN = "unknown"


def nominal_scene(incident: Incident) -> SceneSpec:
    return incident.nominal_scene or SceneSpec(incident.scene.nominal)


def run_stage2(runner: ReplayRunner, incident: Incident) -> ReplayResult:
    return runner.run(V.R_STAGE2_NOMINAL, nominal_scene(incident), incident.code_ref, incident.policy,
                      Mode.CLEAN)


def branch_of(stage2: ReplayResult) -> Branch:
    return {Outcome.RESTORED: Branch.C, Outcome.NO_EFFECT: Branch.G,
            Outcome.PARTIAL: Branch.G_DEGRADED}.get(stage2.outcome, Branch.UNKNOWN)


def branches_to_run(branch: Branch) -> tuple[bool, bool]:
    """(run Branch G, run Branch C)."""
    return {Branch.C: (False, True), Branch.G: (True, False),
            Branch.G_DEGRADED: (True, True), Branch.UNKNOWN: (True, True)}[branch]
