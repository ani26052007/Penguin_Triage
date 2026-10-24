"""Branch G: global failure (plan.md 3.2.6). All replays run; none short-circuits.

G1  infra: the policy on the incident scene in INCIDENT mode. The clean side
    of the comparison is stage0. Only run when the incident was observed in
    incident mode (otherwise there is nothing to compare).
G2  simulator: reference controller on the nominal scene, RAW actions.
G3  pipeline isolation: reference controller on the nominal scene, actions
    through the current WRAPPER and controller.
G4  contract evidence comes from Stage 1, which ``router.route`` runs on every
    incident.
"""

from __future__ import annotations

from router import vocab as V
from router.replay import ReplayRunner
from router.schemas import ActionPipeline, ActorKind, ActorRef, Incident, Mode, ReplayResult
from router.stage2 import nominal_scene


def reference_actor(checkpoint: str, pipeline: ActionPipeline) -> ActorRef:
    return ActorRef(ActorKind.REFERENCE, checkpoint, pipeline)


def run_g1(runner: ReplayRunner, incident: Incident) -> list[ReplayResult]:
    if incident.observed_mode is not Mode.INCIDENT:
        return []
    return [runner.run(V.R_G1, incident.scene, incident.code_ref, incident.policy, Mode.INCIDENT)]


def run_branch_g(runner: ReplayRunner, incident: Incident, reference_checkpoint: str,
                 include_g1: bool = True) -> list[ReplayResult]:
    nominal = nominal_scene(incident)
    out = run_g1(runner, incident) if include_g1 else []
    out.append(runner.run(V.R_G2, nominal, incident.code_ref,
                          reference_actor(reference_checkpoint, ActionPipeline.RAW)))
    out.append(runner.run(V.R_G3, nominal, incident.code_ref,
                          reference_actor(reference_checkpoint, ActionPipeline.WRAPPER)))
    return out
