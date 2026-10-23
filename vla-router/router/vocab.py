"""Names shared between the checks that emit evidence / replays and ``decide``.

``decide`` is a pure function of (evidence list, replay table). The strings
below are the contract between it and the Stage 1 checks, the replay runners
and the behavior analysis (later steps). Change one here, not in two places.

Evidence signals (``Evidence.signal``). A check emits a record only when it
fires; absence of a signal means "checked, nothing found".

Replay ids (``ReplayResult.replay_id``) and how ``outcome`` must be classified
by the caller with ``ladder.stats`` (anchors: failure rate, nominal band):

  stage0          shifted (incident) scene, current code, policy, CLEAN mode.
                  Anchors: failure = the incident's failure rate, band = the
                  reference band for that condition. RESTORED = no real drop.
  stage2_nominal  nominal scene, current code, policy, CLEAN mode.
                  RESTORED = within the nominal band (Branch C),
                  NO_EFFECT = clearly failed (Branch G),
                  PARTIAL = degraded (Branch G first, multi-cause).
  G1              shifted scene, current code, policy, INCIDENT mode (the
                  incident-mode side of the infra check; the clean side is
                  stage0). Recorded for the report; ``decide`` does not read
                  it, since row 3 rests on stage0 plus staleness evidence.
  G2              reference controller, nominal scene, RAW actions.
  G3              reference controller, nominal scene, through the WRAPPER.
  C1              reference controller, shifted scene, RAW actions.
  C_restore:<f>   shifted scene with only factor <f> set back to nominal.
  C_all           shifted scene with every factor restored (only needed when
                  the scene has two or more factors).
  C_render        shifted scene with the training data's render settings.
  For G2, G3 and C1, RESTORED means "the reference controller succeeded".
"""

from __future__ import annotations

# --- evidence signals ------------------------------------------------------ #
SIG_CONTRACT_VIOLATION = "contract_violation"        # cfg / obs / action contract, not render settings
SIG_RENDER_SETTINGS_DIFFER = "render_settings_differ"  # eval render != training render (gates C_render)
SIG_RENDER_PHYSICS_BROKEN = "render_physics_broken"  # black frames, NaN state, interpenetration
SIG_REFERENCE_UNQUALIFIED = "reference_unqualified"  # reference controller not qualified on this condition
SIG_STALENESS_VARIABLE = "staleness_variable"        # load-dependent delay (server staleness)
SIG_STALENESS_CONSTANT = "staleness_constant"        # constant delay also in clean mode (code latency)
SIG_IMPLIED_TARGET_OFFSET = "implied_target_offset"  # consistent offset: frame / extrinsic bug
SIG_IMPLIED_TARGET_OK = "implied_target_ok"          # small scatter around the right object
SIG_IMPLIED_TARGET_WRONG = "implied_target_wrong"    # large scatter or wrong object
SIG_FAILURE_PHASE_PREFIX = "failure_phase:"          # failure_phase:reach|grasp|lift|place

PHASES_ACTION_GAP = ("grasp", "lift", "place")

# --- replay ids ------------------------------------------------------------ #
R_STAGE0 = "stage0"
R_STAGE2_NOMINAL = "stage2_nominal"
R_G1 = "G1"
R_G2 = "G2"
R_G3 = "G3"
R_C1 = "C1"
R_C_ALL = "C_all"
R_C_RENDER = "C_render"
_RESTORE_PREFIX = "C_restore:"


def restore_id(factor: str) -> str:
    return f"{_RESTORE_PREFIX}{factor}"


def failure_phase(phase: str) -> str:
    return f"{SIG_FAILURE_PHASE_PREFIX}{phase}"
