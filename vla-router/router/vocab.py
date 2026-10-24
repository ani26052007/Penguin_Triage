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

from dataclasses import dataclass

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
PHASES = ("reach",) + PHASES_ACTION_GAP

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


# --- unchecked signals (DECISIONS.md, Fix F1) -------------------------------- #
# A check that did not run leaves its signals unchecked: absence of the signal
# then means "not looked at", not "checked, nothing found". ``route`` builds the
# unchecked set from this map; ``decide`` reads it through ROW_SIGNALS.

# Every signal name a check can emit; ``decide(unchecked=...)`` accepts only these.
ALL_SIGNALS: frozenset[str] = frozenset({
    SIG_CONTRACT_VIOLATION, SIG_RENDER_SETTINGS_DIFFER, SIG_RENDER_PHYSICS_BROKEN, SIG_REFERENCE_UNQUALIFIED,
    SIG_STALENESS_VARIABLE, SIG_STALENESS_CONSTANT, SIG_IMPLIED_TARGET_OFFSET, SIG_IMPLIED_TARGET_OK,
    SIG_IMPLIED_TARGET_WRONG, *(failure_phase(p) for p in PHASES)})

@dataclass(frozen=True)
class PartialSkip:
    """A check that ran but could not look at part of its inputs: ``reason`` for the notes,
    ``signals`` the unlooked-at part would have emitted. Only those signals become unchecked
    (a skipped sub-check is not a skipped source), so absence of them is not "checked, nothing
    found" while the rest of the check still counts."""
    reason: str
    signals: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "signals", frozenset(self.signals))
        unknown = self.signals - ALL_SIGNALS
        if unknown or not self.signals:
            raise ValueError(f"PartialSkip needs known signals, got {sorted(self.signals)}")


SIGNALS_BY_SOURCE: dict[str, frozenset[str]] = {
    "contract_cfg": frozenset({SIG_CONTRACT_VIOLATION, SIG_RENDER_SETTINGS_DIFFER}),
    "contract_obs": frozenset({SIG_CONTRACT_VIOLATION}),
    "contract_action": frozenset({SIG_CONTRACT_VIOLATION}),
    "staleness": frozenset({SIG_STALENESS_CONSTANT, SIG_STALENESS_VARIABLE}),
    "sanity": frozenset({SIG_RENDER_PHYSICS_BROKEN}),
    "behavior": frozenset({SIG_IMPLIED_TARGET_OFFSET, SIG_IMPLIED_TARGET_OK, SIG_IMPLIED_TARGET_WRONG,
                           *(failure_phase(p) for p in PHASES)}),
}

# Row -> the signals whose absence that row's result depends on. If one of them
# is unchecked, the result that rests on its absence becomes UNKNOWN: FALSE for
# rows 2-9 (the signal could have made the row TRUE), TRUE for row 1 (the
# signal can only make row 1 FALSE: the inverse rule). Only the evidence-based
# results flip; a result fixed by the replays stays. The default of
# ``DecideConfig.row_signals``: relaxing a signal is a config change.
ROW_SIGNALS: dict[int, frozenset[str]] = {
    1: frozenset({SIG_STALENESS_VARIABLE}),
    2: frozenset({SIG_RENDER_PHYSICS_BROKEN}),
    3: frozenset({SIG_STALENESS_VARIABLE}),
    4: frozenset({SIG_CONTRACT_VIOLATION, SIG_STALENESS_CONSTANT, SIG_IMPLIED_TARGET_OFFSET}),
    5: frozenset({SIG_RENDER_SETTINGS_DIFFER}),
    8: frozenset({SIG_IMPLIED_TARGET_OK, *(failure_phase(p) for p in PHASES_ACTION_GAP)}),
    9: frozenset({SIG_IMPLIED_TARGET_WRONG}),
}
