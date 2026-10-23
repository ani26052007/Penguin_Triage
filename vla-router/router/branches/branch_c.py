"""Branch C: condition-specific failure (plan.md 3.2.7). All replays run.

C1          solvability: reference controller on the incident (shifted) scene,
            RAW actions.
C_restore   one per declared factor: the shifted scene with only that factor
            set back to nominal. Every factor is run, whatever the earlier
            ones returned (no early stop across hypotheses, 3.2.1 #2).
C_all       all factors restored, run only when the scene has two or more
            factors and no single restore came back RESTORED (plan: "If no
            single factor restores success, run one replay with all factors
            restored").
C_render    the shifted scene with the training data's render settings, run
            only when the evidence says the eval render settings differ
            (``render_settings_differ``, a Stage 1 config-contract output).
C_behavior  logs-only analysis; not built in this step (its evidence is passed
            to the router as extra evidence).
"""

from __future__ import annotations

from typing import Sequence

from router import vocab as V
from router.branches.branch_g import reference_actor
from router.replay import ReplayRunner
from router.schemas import ActionPipeline, CodeRef, Evidence, Incident, Outcome, ReplayResult


def render_fixed_code(incident: Incident, render_config_key: str) -> CodeRef | None:
    """The current code with the eval config's render settings replaced by the
    training spec's. None if the training spec does not expose them.

    TODO(verify): the real eval-config key(s) that hold render settings, and
    how the training spec's render settings map onto them (plan 3.2.4: a small
    model proposes the key mapping, hand-checked once per policy).
    """
    train = incident.training_spec.render_settings
    if train is None:
        return None
    cfg = dict(incident.code_ref.eval_config)
    cfg[render_config_key] = train
    return CodeRef(incident.code_ref.code_hash, cfg)


def run_branch_c(runner: ReplayRunner, incident: Incident, reference_checkpoint: str,
                 evidence: Sequence[Evidence], render_config_key: str) -> tuple[list[ReplayResult], list[str]]:
    scene, code, policy = incident.scene, incident.code_ref, incident.policy
    notes: list[str] = []
    out = [runner.run(V.R_C1, scene, code, reference_actor(reference_checkpoint, ActionPipeline.RAW))]

    restores = [runner.run(V.restore_id(f), scene.restore(f), code, policy) for f in scene.factor_names]
    out += restores
    if len(scene.factors) >= 2 and not any(r.outcome is Outcome.RESTORED for r in restores):
        out.append(runner.run(V.R_C_ALL, scene.restore_all(), code, policy))

    if any(e.signal == V.SIG_RENDER_SETTINGS_DIFFER for e in evidence):
        fixed = render_fixed_code(incident, render_config_key)
        if fixed is None:
            notes.append("render settings differ but the training spec has none; C_render not run")
        else:
            out.append(runner.run(V.R_C_RENDER, scene, fixed, policy))
    return out, notes
