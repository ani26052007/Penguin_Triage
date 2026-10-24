"""Tests for router.decide (plan.md 3.2.8): every row, precedence conflicts,
partial restores, conflicting evidence, noisy borderline tables.

No simulator: replays are built with fixed outcomes, or from success counts via
ladder.stats for the borderline tests.
"""

from __future__ import annotations

import itertools
import random

import pytest

from ladder.stats import replay_outcome
from router import vocab as V
from router.decide import (DESTINATION_BY_ROW, LABEL_BY_ROW, DecideConfig, decide,
                           inconclusive_replays)
from router.schemas import (ActorKind, ActorRef, CodeRef, Destination, EpisodeResult, Evidence,
                            Factor, Mode, Outcome, ReplayResult, SceneSpec, VerdictLabel)

R, P, N, I = Outcome.RESTORED, Outcome.PARTIAL, Outcome.NO_EFFECT, Outcome.INCONCLUSIVE

LIGHT, MAT = "lighting", "material(mug)"
SHIFTED = SceneSpec("pick_mug", (Factor(LIGHT, "dark_01"), Factor(MAT, "glossy_02")))
SHIFTED_1 = SceneSpec("pick_mug", (Factor(MAT, "glossy_02"),))
NOMINAL = SceneSpec("pick_mug")
CODE = CodeRef("c0")
POLICY = ActorRef(ActorKind.POLICY, "ckpt")
REF = ActorRef(ActorKind.REFERENCE, "ref_v1")


def rep(rid, outcome, scene=SHIFTED):
    return ReplayResult(rid, scene, CODE, POLICY, Mode.CLEAN, (EpisodeResult(0, True),), outcome)


_n = itertools.count()


def ev(signal, strength=0.8, location=None, source="test"):
    return Evidence(f"e{next(_n)}", source, signal, strength, location, signal)


def table(base, over=None, scene=SHIFTED):
    """{replay_id: Outcome} -> list of ReplayResult. ``over`` replaces or (None) removes entries."""
    d = dict(base)
    for k, v in (over or {}).items():
        if v is None:
            d.pop(k, None)
        else:
            d[k] = v
    return [rep(k, v, scene) for k, v in d.items()]


RL, RM = V.restore_id(LIGHT), V.restore_id(MAT)

# Branch C: nominal fine, shifted scene broken, nothing restores, reference fine.
BASE_C = {V.R_STAGE0: N, V.R_STAGE2_NOMINAL: R, V.R_C1: R, RL: N, RM: N, V.R_C_ALL: N}
# Branch G: nominal fails too; reference fine raw and through the wrapper.
BASE_G = {V.R_STAGE0: N, V.R_STAGE2_NOMINAL: N, V.R_G2: R, V.R_G3: R}


def verdict(evidence, base, over=None, **kw):
    return decide(evidence, table(base, over, **kw))


# --------------------------------------------------------------------------- #
# one case per row (the "all 10 rows have tests" gate)
# --------------------------------------------------------------------------- #

ROW_CASES = {
    1: ([], {V.R_STAGE0: R}),
    2: ([ev(V.SIG_RENDER_PHYSICS_BROKEN)], BASE_C),
    3: ([ev(V.SIG_STALENESS_VARIABLE)], {**BASE_C, V.R_STAGE0: R}),
    4: ([ev(V.SIG_CONTRACT_VIOLATION, location="norm_stats")], BASE_C),
    5: ([ev(V.SIG_RENDER_SETTINGS_DIFFER)], {**BASE_C, V.R_C_RENDER: R}),
    6: ([], {**BASE_C, RM: R}),
    7: ([], {**BASE_C, V.R_C_ALL: R}),
    8: ([ev(V.SIG_IMPLIED_TARGET_OK), ev(V.failure_phase("grasp"))], BASE_C),
    9: ([ev(V.SIG_IMPLIED_TARGET_WRONG)], BASE_C),
    10: ([], BASE_C),
}


@pytest.mark.parametrize("row", range(1, 11))
def test_each_row(row):
    evidence, base = ROW_CASES[row]
    v = verdict(evidence, base)
    assert v.row == row
    assert v.label is LABEL_BY_ROW[row]
    assert v.destination is DESTINATION_BY_ROW[row]


def test_label_and_destination_tables_match_plan():
    assert [DESTINATION_BY_ROW[i] for i in range(1, 11)] == [
        Destination.STOP, Destination.REPORT, Destination.REPORT, Destination.HAGENT,
        Destination.CONFIG_FIX, Destination.DATA_LADDER, Destination.DATA_LADDER,
        Destination.REPORT, Destination.REPORT, Destination.REPORT]
    assert len({LABEL_BY_ROW[i] for i in range(1, 11)}) == 10


# --------------------------------------------------------------------------- #
# row 1
# --------------------------------------------------------------------------- #

def test_row1_partial_stage0_is_a_real_drop():
    v = verdict([], BASE_G, {V.R_STAGE0: P})
    assert v.row == 4


def test_row1_blocked_by_variable_staleness_goes_to_infra():
    v = verdict([ev(V.SIG_STALENESS_VARIABLE)], BASE_C, {V.R_STAGE0: R})
    assert v.row == 3 and v.label is VerdictLabel.INFRA


def test_row1_inconclusive_stage0_is_undetermined_and_asks_for_seeds():
    v = verdict([], {V.R_STAGE0: I})
    assert v.row == 10 and V.R_STAGE0 in v.notes


def test_missing_stage0_is_undetermined():
    v = verdict([], {k: o for k, o in BASE_C.items() if k != V.R_STAGE0})
    assert v.row == 10


# --------------------------------------------------------------------------- #
# row 2
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("signal", [V.SIG_RENDER_PHYSICS_BROKEN, V.SIG_REFERENCE_UNQUALIFIED])
def test_row2_evidence(signal):
    v = verdict([ev(signal)], BASE_C)
    assert v.row == 2 and v.evidence_ids


def test_row2_c1_fails_in_branch_c_and_attaches_the_proving_replay():
    v = verdict([], BASE_C, {V.R_C1: N})
    assert v.row == 2
    assert [r.replay_id for r in v.replays] == [V.R_C1]


def test_row2_g2_fails_in_branch_g():
    v = verdict([], BASE_G, {V.R_G2: N})
    assert v.row == 2 and [r.replay_id for r in v.replays] == [V.R_G2]


def test_row2_degraded_reference_counts_as_failed():
    assert verdict([], BASE_C, {V.R_C1: P}).row == 2


def test_row2_beats_code_vision_and_rendering():
    evidence = [ev(V.SIG_RENDER_PHYSICS_BROKEN), ev(V.SIG_CONTRACT_VIOLATION),
                ev(V.SIG_RENDER_SETTINGS_DIFFER)]
    v = verdict(evidence, BASE_C, {RM: R, V.R_C_RENDER: R})
    assert v.row == 2


def test_row2_beats_infra():
    v = verdict([ev(V.SIG_STALENESS_VARIABLE)], BASE_C, {V.R_STAGE0: P, V.R_C1: N})
    assert v.row == 2


def test_row2_reference_replay_needed_for_the_branch_is_missing():
    v = verdict([], BASE_C, {V.R_C1: None})
    assert v.row == 10 and "C1" in v.notes


# --------------------------------------------------------------------------- #
# row 3
# --------------------------------------------------------------------------- #

def test_row3_infra():
    v = verdict([ev(V.SIG_STALENESS_VARIABLE)], BASE_C, {V.R_STAGE0: R})
    assert v.row == 3 and v.destination is Destination.REPORT and not v.likely_second_cause


def test_row3_partial_clean_restore_flags_second_cause():
    v = verdict([ev(V.SIG_STALENESS_VARIABLE)], BASE_C, {V.R_STAGE0: P})
    assert v.row == 3 and v.likely_second_cause


def test_row3_needs_clean_mode_to_restore():
    v = verdict([ev(V.SIG_STALENESS_VARIABLE)], BASE_G)
    assert v.row == 4          # variable staleness alone does not explain a failure that persists in clean mode


def test_row3_beats_code_and_flags_it():
    v = verdict([ev(V.SIG_STALENESS_VARIABLE), ev(V.SIG_STALENESS_CONSTANT, location="controller.delay")],
                BASE_C, {V.R_STAGE0: R})
    assert v.row == 3 and v.likely_second_cause


def test_row3_beats_vision():
    v = verdict([ev(V.SIG_STALENESS_VARIABLE)], BASE_C, {V.R_STAGE0: R, RM: R})
    assert v.row == 3 and not v.likely_second_cause


# --------------------------------------------------------------------------- #
# row 4
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("signal", [V.SIG_CONTRACT_VIOLATION, V.SIG_STALENESS_CONSTANT,
                                    V.SIG_IMPLIED_TARGET_OFFSET])
def test_row4_located_evidence_is_higher_confidence(signal):
    v = verdict([ev(signal, location="somewhere")], BASE_C)
    assert v.row == 4 and v.destination is Destination.HAGENT
    assert not v.by_elimination and v.location == "somewhere"
    assert "higher" in v.notes


def test_row4_g3_pipeline_failure_names_the_action_side():
    v = verdict([], BASE_G, {V.R_G3: N})
    assert v.row == 4 and not v.by_elimination and v.location == "action_side"
    assert {r.replay_id for r in v.replays} == {V.R_G2, V.R_G3}


def test_row4_g3_in_branch_c_is_still_code():
    v = verdict([], BASE_C, {V.R_G2: R, V.R_G3: N})
    assert v.row == 4 and v.location == "action_side"


def test_row4_branch_g_elimination_is_low_confidence():
    v = verdict([], BASE_G)
    assert v.row == 4 and v.by_elimination
    assert v.location == "observation_side_or_policy_loading"
    assert "elimination" in v.notes


def test_row4_located_beats_elimination_in_branch_g():
    v = verdict([ev(V.SIG_CONTRACT_VIOLATION, location="chunk_len")], BASE_G)
    assert not v.by_elimination and v.location == "chunk_len"


def test_row4_location_comes_from_the_strongest_evidence():
    v = verdict([ev(V.SIG_CONTRACT_VIOLATION, 0.4, "weak"), ev(V.SIG_CONTRACT_VIOLATION, 0.9, "strong")], BASE_C)
    assert v.location == "strong"
    assert len(v.evidence_ids) == 2


def test_row4_degraded_stage2_sets_multi_cause_flag():
    v = verdict([], BASE_G, {V.R_STAGE2_NOMINAL: P})
    assert v.row == 4 and v.multi_cause_flag and v.by_elimination


def test_row4_not_chosen_in_branch_c_without_evidence():
    assert verdict([], BASE_C).row == 10


def test_min_strength_filters_weak_evidence():
    weak = [ev(V.SIG_CONTRACT_VIOLATION, 0.1)]
    assert verdict(weak, BASE_C).row == 4
    assert decide(weak, table(BASE_C), DecideConfig(min_strength=0.5)).row == 10


# --------------------------------------------------------------------------- #
# precedence conflicts
# --------------------------------------------------------------------------- #

def test_contract_violation_and_full_vision_restore_is_code_without_second_cause():
    """A condition-specific code bug flips under a factor restore: one cause."""
    v = verdict([ev(V.SIG_CONTRACT_VIOLATION, location="instruction_template")], BASE_C, {RM: R})
    assert v.row == 4 and not v.likely_second_cause


def test_contract_violation_and_partial_vision_restore_flags_second_cause():
    v = verdict([ev(V.SIG_CONTRACT_VIOLATION, location="norm_stats")], BASE_C, {RM: P})
    assert v.row == 4 and v.likely_second_cause


def test_contract_violation_and_partial_all_factors_restore_flags_second_cause():
    v = verdict([ev(V.SIG_CONTRACT_VIOLATION)], BASE_C, {V.R_C_ALL: P})
    assert v.row == 4 and v.likely_second_cause


def test_code_beats_rendering_and_flags_a_partial_render_restore():
    evidence = [ev(V.SIG_CONTRACT_VIOLATION), ev(V.SIG_RENDER_SETTINGS_DIFFER)]
    assert verdict(evidence, BASE_C, {V.R_C_RENDER: R}).row == 4
    assert verdict(evidence, BASE_C, {V.R_C_RENDER: P}).likely_second_cause


def test_combined_code_plus_lighting_is_code_degraded_and_flagged():
    """6.5 'Combined': nominal degraded, factor restore only partial."""
    over = {V.R_STAGE2_NOMINAL: P, V.R_G2: R, V.R_G3: R, RL: P, RM: N}
    v = verdict([ev(V.SIG_CONTRACT_VIOLATION)], BASE_C, over)
    assert v.row == 4 and v.likely_second_cause and v.multi_cause_flag


def test_rendering_beats_vision_without_second_cause():
    v = verdict([ev(V.SIG_RENDER_SETTINGS_DIFFER)], BASE_C, {V.R_C_RENDER: R, RM: R})
    assert v.row == 5 and not v.likely_second_cause


def test_vision_restore_and_infra_and_sim_precedence_chain():
    evidence = [ev(V.SIG_STALENESS_VARIABLE)]
    over = {V.R_STAGE0: R, RM: R}
    assert verdict(evidence, BASE_C, over).row == 3
    assert verdict(evidence, BASE_C, {**over, V.R_C1: N}).row == 2


# --------------------------------------------------------------------------- #
# row 5
# --------------------------------------------------------------------------- #

def test_row5_rendering_mismatch_goes_to_config_fix():
    v = verdict([ev(V.SIG_RENDER_SETTINGS_DIFFER, location="render.samples")], BASE_C, {V.R_C_RENDER: R})
    assert v.row == 5 and v.destination is Destination.CONFIG_FIX and v.location == "render.samples"


def test_row5_partial_restore_flags_second_cause():
    v = verdict([ev(V.SIG_RENDER_SETTINGS_DIFFER)], BASE_C, {V.R_C_RENDER: P})
    assert v.row == 5 and v.likely_second_cause


def test_row5_needs_the_differing_settings_evidence():
    """C_render restoring without a declared difference is not a rendering verdict."""
    v = verdict([], BASE_C, {V.R_C_RENDER: R})
    assert v.row == 10


def test_row5_render_replay_that_does_nothing_falls_through_to_vision():
    v = verdict([ev(V.SIG_RENDER_SETTINGS_DIFFER)], BASE_C, {V.R_C_RENDER: N, RM: R})
    assert v.row == 6


def test_row5_missing_render_replay_is_undetermined():
    v = verdict([ev(V.SIG_RENDER_SETTINGS_DIFFER)], BASE_C, {RM: R})
    assert v.row == 10 and "C_render" in v.notes


# --------------------------------------------------------------------------- #
# rows 6 and 7
# --------------------------------------------------------------------------- #

def test_row6_names_the_factor():
    v = verdict([], BASE_C, {RL: R})
    assert v.row == 6 and v.factor == LIGHT and v.destination is Destination.DATA_LADDER
    assert not v.likely_second_cause


def test_row6_partial_single_restore_flags_second_cause():
    v = verdict([], BASE_C, {RM: P})
    assert v.row == 6 and v.factor == MAT and v.likely_second_cause


def test_row6_two_full_restores_is_conflicting_not_order_dependent():
    v = verdict([], BASE_C, {RL: R, RM: R})
    assert v.row == 10 and "cannot be named" in v.notes


def test_row6_two_partial_restores_is_conflicting():
    assert verdict([], BASE_C, {RL: P, RM: P}).row == 10


def test_row6_with_inconclusive_sibling_is_undetermined():
    v = verdict([], BASE_C, {RL: R, RM: I})
    assert v.row == 10 and RM in v.notes


def test_row6_single_factor_scene_needs_no_all_factors_replay():
    base = {V.R_STAGE0: N, V.R_STAGE2_NOMINAL: R, V.R_C1: R, RM: R}
    v = verdict([], base, scene=SHIFTED_1)
    assert v.row == 6 and v.factor == MAT


def test_row7_only_all_factors_restores():
    v = verdict([], BASE_C, {V.R_C_ALL: R})
    assert v.row == 7 and v.label is VerdictLabel.VISION_INTERACTING
    assert v.factor == f"{LIGHT};{MAT}" and v.destination is Destination.DATA_LADDER


def test_row7_partial_singles_add_up_to_a_full_all_restore():
    v = verdict([], BASE_C, {RL: P, RM: P, V.R_C_ALL: R})
    assert v.row == 7 and not v.likely_second_cause


def test_row7_partial_all_restore_flags_second_cause():
    v = verdict([], BASE_C, {V.R_C_ALL: P})
    assert v.row == 7 and v.likely_second_cause


def test_row7_missing_all_factors_replay_is_undetermined():
    v = verdict([], BASE_C, {V.R_C_ALL: None})
    assert v.row == 10 and "C_all" in v.notes


def test_row6_beats_row7_when_a_single_restores():
    assert verdict([], BASE_C, {RM: R, V.R_C_ALL: R}).row == 6


# --------------------------------------------------------------------------- #
# rows 8, 9, 10
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("phase", ["grasp", "lift", "place"])
def test_row8_action_gap(phase):
    v = verdict([ev(V.SIG_IMPLIED_TARGET_OK), ev(V.failure_phase(phase))], BASE_C)
    assert v.row == 8 and v.destination is Destination.REPORT and phase in v.notes


def test_row8_failure_at_reach_with_right_target_is_not_an_action_gap():
    assert verdict([ev(V.SIG_IMPLIED_TARGET_OK), ev(V.failure_phase("reach"))], BASE_C).row == 10


def test_row8_needs_nothing_to_restore():
    evidence = [ev(V.SIG_IMPLIED_TARGET_OK), ev(V.failure_phase("grasp"))]
    assert verdict(evidence, BASE_C, {RM: R}).row == 6


def test_row8_inconclusive_restore_blocks_the_claim():
    evidence = [ev(V.SIG_IMPLIED_TARGET_OK), ev(V.failure_phase("grasp"))]
    assert verdict(evidence, BASE_C, {RM: I}).row == 10


def test_row9_perception_unresolved():
    v = verdict([ev(V.SIG_IMPLIED_TARGET_WRONG)], BASE_C)
    assert v.row == 9 and v.label is VerdictLabel.PERCEPTION_UNRESOLVED


def test_rows_8_9_conflicting_target_evidence_is_undetermined():
    evidence = [ev(V.SIG_IMPLIED_TARGET_OK), ev(V.SIG_IMPLIED_TARGET_WRONG), ev(V.failure_phase("grasp"))]
    v = verdict(evidence, BASE_C)
    assert v.row == 10 and "both right and wrong" in v.notes


def test_row10_nothing_restores_and_no_behavior_evidence():
    assert verdict([], BASE_C).row == 10


# --------------------------------------------------------------------------- #
# inconclusive / noisy borderline tables
# --------------------------------------------------------------------------- #

FAIL, BAND = 0.15, (0.7, 0.95)


def counts(rid, k, n, scene=SHIFTED):
    out, _ = replay_outcome(k, n, FAIL, BAND)
    return ReplayResult(rid, scene, CODE, POLICY, Mode.CLEAN,
                        tuple(EpisodeResult(i, i < k) for i in range(n)), out)


def borderline_table(k_material, n):
    return [counts(V.R_STAGE0, 3, n), counts(V.R_STAGE2_NOMINAL, int(0.9 * n), n),
            counts(V.R_C1, n, n), counts(RL, 3, n), counts(RM, k_material, n), counts(V.R_C_ALL, k_material, n)]


def test_borderline_restore_is_inconclusive_then_resolves_with_more_seeds():
    t = borderline_table(18, 30)                   # 60%: interval reaches the band, rate below it
    assert {r.replay_id: r.outcome for r in t}[RM] is Outcome.INCONCLUSIVE
    assert inconclusive_replays(t) == (V.R_C_ALL, RM)
    v = decide([], t)
    assert v.row == 10 and RM in v.notes and "extend seeds" in v.notes
    assert decide([], borderline_table(48, 60)).row == 6       # same cell, more seeds, clearly restored


def test_borderline_low_restore_is_partial_not_inconclusive():
    t = borderline_table(12, 30)
    assert {r.replay_id: r.outcome for r in t}[RM] is Outcome.PARTIAL
    v = decide([], t)
    # both single restores are partial? no: RL stays at 3/30, so material is the only candidate
    assert v.row == 6 and v.factor == MAT and v.likely_second_cause


def test_unknown_higher_row_blocks_a_lower_winner():
    """G2 inconclusive might be a sim problem: a code verdict cannot be trusted yet."""
    v = verdict([ev(V.SIG_CONTRACT_VIOLATION)], BASE_G, {V.R_G2: I})
    assert v.row == 10 and "G2" in v.notes


def test_unknown_lower_row_does_not_block_a_higher_winner():
    v = verdict([ev(V.SIG_CONTRACT_VIOLATION)], BASE_C, {RM: I})
    assert v.row == 4


def test_unknown_stage2_blocks_code_by_elimination_but_not_located_code():
    base = {**BASE_C, V.R_STAGE2_NOMINAL: I}
    assert verdict([], base).row == 10
    assert verdict([ev(V.SIG_CONTRACT_VIOLATION)], base).row == 10     # row 2 still needs the branch
    base_ok = {**base, V.R_G2: R}
    assert verdict([ev(V.SIG_CONTRACT_VIOLATION)], base_ok).row == 4


# --------------------------------------------------------------------------- #
# purity and robustness
# --------------------------------------------------------------------------- #

def test_duplicate_evidence_ids_and_replay_ids_are_rejected():
    e = Evidence("x", "s", V.SIG_CONTRACT_VIOLATION, 0.5)
    with pytest.raises(ValueError):
        decide([e, e], table(BASE_C))
    with pytest.raises(ValueError):
        decide([], table(BASE_C) + [rep(V.R_STAGE0, N)])


def test_unclassified_replay_is_rejected():
    raw = ReplayResult(V.R_STAGE0, SHIFTED, CODE, POLICY, Mode.CLEAN, (EpisodeResult(0, True),))
    with pytest.raises(ValueError):
        decide([], [raw])


def test_replay_table_may_be_a_mapping():
    t = {r.replay_id: r for r in table(BASE_C, {RM: R})}
    assert decide([], t).row == 6


def test_verdict_replays_are_the_proving_ones_and_round_trip_json():
    v = verdict([], BASE_C, {RM: R})
    assert [r.replay_id for r in v.replays] == [RM]
    from router.schemas import Verdict
    assert Verdict.from_dict(v.to_dict()) == v


def _random_case(rng):
    """Biased towards plausible tables so that every row is reachable: the
    reference controller mostly works, Stage 2 mostly passes, evidence is rare."""
    def pick(weights):
        return rng.choices([R, P, N, I], weights)[0]
    spec = {V.R_STAGE0: (1, 1, 5, 1), V.R_STAGE2_NOMINAL: (5, 1, 2, 1), V.R_G2: (9, 1, 1, 1),
            V.R_G3: (7, 1, 3, 1), V.R_C1: (9, 1, 1, 1), V.R_C_ALL: (2, 2, 8, 1),
            V.R_C_RENDER: (3, 2, 4, 1), RL: (2, 2, 8, 1), RM: (2, 2, 8, 1)}
    reps = [rep(i, pick(w)) for i, w in spec.items() if rng.random() < 0.9]
    sigs = [V.SIG_CONTRACT_VIOLATION, V.SIG_RENDER_SETTINGS_DIFFER, V.SIG_RENDER_PHYSICS_BROKEN,
            V.SIG_REFERENCE_UNQUALIFIED, V.SIG_STALENESS_VARIABLE, V.SIG_STALENESS_CONSTANT,
            V.SIG_IMPLIED_TARGET_OFFSET, V.SIG_IMPLIED_TARGET_OK, V.SIG_IMPLIED_TARGET_WRONG,
            V.failure_phase("grasp"), V.failure_phase("reach")]
    evs = [Evidence(f"r{i}", "t", s, rng.random(), "loc" if rng.random() < .5 else None)
           for i, s in enumerate(sigs) if rng.random() < 0.2]
    return evs, reps


def test_random_tables_never_crash_are_consistent_and_order_independent():
    rng = random.Random(7)
    seen_rows = set()
    for _ in range(4000):
        evs, reps = _random_case(rng)
        v = decide(evs, reps)
        seen_rows.add(v.row)
        assert v.label is LABEL_BY_ROW[v.row] and v.destination is DESTINATION_BY_ROW[v.row]
        if v.row == 10:
            assert v.label is VerdictLabel.UNDETERMINED
        if v.by_elimination:
            assert v.row == 4
        shuffled_e, shuffled_r = evs[:], reps[:]
        rng.shuffle(shuffled_e)
        rng.shuffle(shuffled_r)
        assert decide(shuffled_e, shuffled_r) == v           # no test-order dependence (3.2.1 #2)
        assert decide(evs, reps) == v                        # deterministic
    assert seen_rows >= {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}      # the generator reaches every row


def test_code_verdict_says_when_the_second_cause_check_was_inconclusive():
    v = verdict([ev(V.SIG_CONTRACT_VIOLATION)], BASE_C, {RL: I})
    assert v.row == 4 and not v.likely_second_cause
    assert "second-cause check incomplete" in v.notes
