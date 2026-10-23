"""Stage 0: reproduce the failure in clean mode (plan.md 3.2.4).

Runs the incident's scene and code with the policy in CLEAN mode on the
shared seed set (10 seeds, extended to 20 when inconclusive).

Whether to stop is not decided here with a second rule: the orchestrator asks
``decide`` with only this replay and stops when that is row 1. So "no
significant failure" has one definition, including its exception (variable
staleness evidence means a clean pass is an infra signal, not a stop).
"""

from __future__ import annotations

from router import vocab as V
from router.replay import ReplayRunner
from router.schemas import Incident, Mode, Outcome, ReplayResult


def run_stage0(runner: ReplayRunner, incident: Incident) -> ReplayResult:
    return runner.run(V.R_STAGE0, incident.scene, incident.code_ref, incident.policy, Mode.CLEAN)


def failure_only_in_incident_mode(stage0: ReplayResult, incident: Incident) -> bool:
    """Clean mode passes but the incident was observed in incident mode: plan
    3.2.4 says record it and let Branch G's infra check (G1) handle it."""
    return stage0.outcome is Outcome.RESTORED and incident.observed_mode is Mode.INCIDENT
