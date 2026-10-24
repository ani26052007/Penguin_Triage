"""The full router on MockBackend (plan.md 3.2.4-3.2.7, 3.2.14): Stage 0,
Stage 2, Branch G, Branch C, orchestrator.

"Done when": every one of the 11 mocked fault classes (6.5) gets the right
verdict, and episodes per incident are logged. Two views:
  * QuotaBackend: no sampling noise, so routing logic must be exact;
  * MockBackend over 10 salts: sampling noise at 10-30 seeds may cost a repeat.
Since Fix F1 the evidence comes from ``route()`` running Stage 1 and C_behavior
on each case's synthetic logs; no case carries hand-written evidence.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from ladder.backend import MockRule
from router import vocab as V
from router.behavior_synthetic import BehaviorFaults
from router.prereq.rollout_cache import CachedBackend, RolloutCache
from router.replay import ReplayConfig, ReplayRunner, seed_set
from router.router import RouterInputs, route
from router.schemas import (ActionPipeline, ActorKind, ActorRef, CodeRef, Evidence, Factor, Incident, Mode,
                            Outcome, SceneSpec, TrainingSpec, VerdictLabel)
from router.stage2 import Branch
from scripts.confusion_matrix import FAILURE_RATE, LIGHT, MAT, NOMINAL_BAND, POLICY, _REF_OK
from tests.router_cases import (ACTION_SPEC, CODE_EVAL_RENDER, TRAIN_RENDER, QuotaBackend, RouterCase, default_config,
                                healthy_inputs, log_refs, router_cases)

CASES = {c.name: c for c in router_cases()}
ROUTABLE = list(CASES)
NOISE_FREE = ("vision_glossy", "rendering_mismatch", "sim_asset", "infra_stale_server", "code_global")
CFG = default_config()


def run_case(case: RouterCase, backend=None, cfg=CFG, **inputs_kw):
    return route(case.incident(), backend or case.quota_backend(), cfg, case.inputs(**inputs_kw))


def check_flags(case: RouterCase, v) -> list[str]:
    bad = []
    if case.expected_factor is not None and v.factor != case.expected_factor:
        bad.append(f"factor={v.factor!r}")
    if case.expected_by_elimination is not None and v.by_elimination != case.expected_by_elimination:
        bad.append(f"by_elimination={v.by_elimination}")
    if case.expected_second_cause is not None and v.likely_second_cause != case.expected_second_cause:
        bad.append(f"likely_second_cause={v.likely_second_cause}")
    return bad


# --------------------------------------------------------------------------- #
# the 11 fault classes
# --------------------------------------------------------------------------- #

def test_eleven_classes_cover_every_routable_label():
    assert len(CASES) == 11
    assert {c.expected for c in CASES.values()} == set(VerdictLabel) - {
        VerdictLabel.PERCEPTION_UNRESOLVED, VerdictLabel.UNDETERMINED}


@pytest.mark.parametrize("name", ROUTABLE)
def test_fault_class_routes_exactly_without_sampling_noise(name):
    case = CASES[name]
    res = run_case(case)
    assert res.verdict.label is case.expected, (res.verdict.notes, [(r.replay_id, r.outcome.value) for r in res.replays])
    assert check_flags(case, res.verdict) == []
    assert res.episodes_simulated > 0


@pytest.mark.parametrize("name", ROUTABLE)
def test_fault_class_routes_under_sampling_noise(name):
    case = CASES[name]
    verdicts = [run_case(case, case.backend(f"{name}:{r}")).verdict for r in range(10)]
    correct = [v for v in verdicts if v.label is case.expected]
    if name in NOISE_FREE:
        assert len(correct) == 10, [v.label.value for v in verdicts]
        assert all(check_flags(case, v) == [] for v in correct)
    else:
        assert len(correct) >= 8, [v.label.value for v in verdicts]


def test_code_condition_specific_needs_stage1():
    """Was a strict xfail (ISSUES #1): the wrong instruction string for the
    glossy mug is found by the config contract that route() now runs."""
    case = CASES["code_condition_specific"]
    v = run_case(case).verdict
    assert v.label is VerdictLabel.CODE and v.location == "eval_config.instruction"
    assert "s1.cfg.instruction_templates" in v.evidence_ids


# --------------------------------------------------------------------------- #
# Fix F1 done-when: log-only evidence produced by route() itself
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,row,location", [
    ("infra_stale_server", 3, None),                       # was row 1 (silent stop)
    ("rendering_mismatch", 5, "eval_config.render"),       # was row 6
    ("code_condition_specific", 4, "eval_config.instruction"),   # was row 6
    ("action_gap", 8, None),                               # was row 10
])
def test_log_only_evidence_routes_without_hand_written_evidence(name, row, location):
    res = run_case(CASES[name])
    assert res.verdict.row == row, res.verdict.notes
    assert res.external_evidence == () and res.skipped == ()
    assert {s for s, _ in res.partial} <= {"behavior"}      # a log with no subtask records leaves the phase unjudged
    assert res.evidence == res.stage1_evidence + res.behavior_evidence
    if location:
        assert res.verdict.location == location


def test_stage1_runs_before_the_stage0_stop_and_behavior_only_on_branch_c():
    infra = run_case(CASES["infra_stale_server"])
    assert infra.stages[:2] == ("stage0", "stage1") and "s1.staleness.variable" in infra.verdict.evidence_ids
    g = run_case(CASES["code_global"])
    assert g.branch is Branch.G and "C_behavior" not in g.stages and g.behavior_evidence == ()
    assert "behavior" not in g.sources_run and not any(s == "behavior" for s, _ in g.skipped)
    c = run_case(CASES["action_gap"])
    assert "C_behavior" in c.stages and "behavior" in c.sources_run and c.behavior_evidence


@pytest.mark.parametrize("name", ROUTABLE)
def test_removing_the_logs_never_gives_a_silent_row_1(name):
    """No logs at all: every trace check and C_behavior are skipped, the notes
    say so, and the verdict is row 10 (or row 2 proved by a replay), never row 1."""
    res = route(CASES[name].incident(log_modes=()), CASES[name].quota_backend(), CFG, CASES[name].inputs())
    assert {s for s, _ in res.skipped} >= {"contract_obs", "contract_action", "staleness", "sanity"}
    assert "checks skipped: " in res.verdict.notes and "staleness (" in res.verdict.notes
    assert res.verdict.row in (2, 10), (res.verdict.row, res.verdict.notes)
    if res.verdict.row == 2:
        assert res.verdict.replays                         # a failing reference controller, not evidence


def test_missing_incident_mode_logs_stop_infra_at_row_10_not_row_1():
    case = CASES["infra_stale_server"]
    res = route(case.incident(log_modes=(Mode.CLEAN.value,)), case.quota_backend(), CFG, case.inputs())
    assert [s for s, _ in res.partial] == ["staleness"] and res.unchecked == (V.SIG_STALENESS_VARIABLE,)
    v = res.verdict
    assert v.row == 10 and "row 1 undecided" in v.notes and "checks partly skipped: staleness" in v.notes
    assert res.stages == ("stage0", "stage1")              # row 1 undecided: nothing below it can win


def test_code_condition_specific_without_contract_checks_is_row_10_not_a_vision_gap():
    res = run_case(CASES["code_condition_specific"], stage1=False)
    assert res.verdict.row == 10 and "row 4 undecided: not checked: contract_violation" in res.verdict.notes
    assert "contract_cfg (no check configured" in res.verdict.notes


def test_code_condition_specific_needs_no_traces_for_its_contract_hit():
    res = route(CASES["code_condition_specific"].incident(log_modes=()), CASES["code_condition_specific"].quota_backend(),
                CFG, CASES["code_condition_specific"].inputs())
    assert "s1.cfg.instruction_templates" in [e.id for e in res.stage1_evidence]
    assert "contract_cfg" in res.sources_run


def test_a_clean_mode_incident_without_logs_can_still_close_at_row_1():
    """staleness_variable does not apply when the incident was seen in clean mode."""
    case = CASES["no_fault"]
    inc = Incident(case.name, case.scene, TrainingSpec("p", **ACTION_SPEC), case.code_ref, POLICY, range(10), Mode.CLEAN,
                   log_refs=log_refs(case.name))
    res = route(inc, case.quota_backend(), CFG, case.inputs())
    assert res.verdict.row == 1 and V.SIG_STALENESS_VARIABLE not in res.unchecked


def test_positional_evidence_is_rejected_not_taken_for_inputs():
    case = CASES["action_gap"]
    with pytest.raises(TypeError, match="extra_evidence"):
        route(case.incident(), case.quota_backend(), CFG, ())


# --------------------------------------------------------------------------- #
# partial skips (ISSUES #59)
# --------------------------------------------------------------------------- #

def test_gripper_never_closing_leaves_the_vision_verdict_undecided_not_row_6():
    """C_behavior cannot judge the implied target (no grasp record) or the phase: those signals are
    unchecked, so rows 4, 8 and 9 are UNKNOWN instead of FALSE and the case is row 10, not a vision gap."""
    case = replace(CASES["vision_glossy"], behavior_default=BehaviorFaults(no_grasp=True))
    assert run_case(CASES["vision_glossy"]).verdict.row == 6
    res = run_case(case)
    assert res.verdict.row == 10 and res.verdict.label is VerdictLabel.UNDETERMINED, res.verdict.notes
    assert {V.SIG_IMPLIED_TARGET_OFFSET, V.SIG_IMPLIED_TARGET_OK, V.SIG_IMPLIED_TARGET_WRONG} <= set(res.unchecked)
    assert "behavior" in res.sources_run and res.skipped == ()
    assert {s for s, _ in res.partial} == {"behavior"} and "implied target not judged" in res.verdict.notes


def test_action_contract_without_layout_or_limits_marks_contract_violation_unchecked():
    case = CASES["code_global"]
    inc = replace(case.incident(), training_spec=TrainingSpec("p"))
    res = route(inc, case.quota_backend(), CFG, case.inputs())
    assert {s for s, _ in res.partial} >= {"contract_action"} and V.SIG_CONTRACT_VIOLATION in res.unchecked
    assert "gripper sign not checked" in res.verdict.notes


# --------------------------------------------------------------------------- #
# episodes per incident, cache, seeds
# --------------------------------------------------------------------------- #

def test_episodes_per_incident_are_logged_and_counted(caplog):
    case = CASES["vision_glossy"]
    backend = case.quota_backend()
    with caplog.at_level(logging.INFO, logger="router"):
        res = run_case(case, backend)
    assert res.episodes_simulated == sum(backend.calls.values())
    assert res.episodes_used == sum(r.n for r in res.replays) == sum(res.episodes_by_replay().values())
    line = next(r.getMessage() for r in caplog.records if r.name == "router")
    assert f"episodes_simulated={res.episodes_simulated}" in line
    assert f"episodes_used={res.episodes_used}" in line and "verdict=vision_factor" in line


def test_second_pass_through_the_same_cache_simulates_nothing():
    case = CASES["combined_code_plus_light"]
    cached = CachedBackend(case.quota_backend(), RolloutCache())
    first = run_case(case, cached)
    second = run_case(case, cached)
    assert first.episodes_simulated > 0 and second.episodes_simulated == 0
    assert second.cache_hits == second.episodes_used and second.verdict == first.verdict


def test_identical_configurations_are_simulated_once():
    """code_partial_offset's incident scene is the nominal scene, so stage0 and
    stage2_nominal are the same configuration: the cache runs it once."""
    res = run_case(CASES["code_partial_offset"])
    s0, s2 = res.replay(V.R_STAGE0), res.replay(V.R_STAGE2_NOMINAL)
    assert s0.scene == s2.scene and res.cache_hits >= min(s0.n, s2.n)
    assert res.episodes_simulated == res.episodes_used - res.cache_hits


@pytest.mark.parametrize("name", ROUTABLE)
def test_all_replays_share_one_seed_set(name):
    case = CASES[name]
    res = run_case(case, case.backend(f"{name}:0"))
    assert res.seeds[:10] == tuple(range(10))
    for r in res.replays:
        assert tuple(e.seed for e in r.episodes) == res.seeds[:r.n], r.replay_id


def test_seed_set_keeps_failing_seeds_and_pads():
    assert seed_set([7, 3, 7, 12], 6) == (7, 3, 12, 13, 14, 15)
    assert seed_set([], 3) == (0, 1, 2)
    assert seed_set(range(40), 30) == tuple(range(30))


# --------------------------------------------------------------------------- #
# Stage 0
# --------------------------------------------------------------------------- #

def test_stage0_no_significant_failure_stops_the_router():
    res = run_case(CASES["no_fault"])
    assert res.verdict.row == 1 and res.stages == ("stage0", "stage1") and res.branch is None
    assert [r.replay_id for r in res.replays] == [V.R_STAGE0]
    assert res.episodes_used == 10


def test_stage0_clean_pass_with_variable_staleness_continues_to_infra():
    res = run_case(CASES["infra_stale_server"])
    assert res.verdict.label is VerdictLabel.INFRA
    assert res.replay(V.R_STAGE0).outcome is Outcome.RESTORED
    g1 = res.replay(V.R_G1)
    assert g1 is not None and g1.mode is Mode.INCIDENT and g1.outcome is Outcome.NO_EFFECT
    assert any("incident mode" in n for n in res.notes)


def _incident(scene, observed_mode=Mode.INCIDENT, render=None, code=CodeRef("c0")):
    return Incident("t", scene, TrainingSpec("p", render_settings=render, **ACTION_SPEC), code, POLICY, range(10), observed_mode,
                    log_refs=log_refs("t"))


def test_stage0_inconclusive_extends_to_20_and_no_further():
    shifted = SceneSpec("pick_mug", (Factor(MAT, "glossy_02"),))
    backend = QuotaBackend(_REF_OK + (MockRule(0.6, scene=shifted),), 0.9)
    res = route(_incident(shifted), backend, CFG, healthy_inputs())
    s0 = res.replay(V.R_STAGE0)
    assert s0.n == 20 and s0.outcome is Outcome.INCONCLUSIVE and V.R_STAGE0 in res.extended
    assert res.verdict.row == 10 and "stage0" in res.verdict.notes
    assert res.stages == ("stage0", "stage1") and res.episodes_used == 20     # nothing else can change row 10


def test_other_replays_extend_to_30():
    shifted = SceneSpec("pick_mug", (Factor(MAT, "glossy_02"),))
    backend = QuotaBackend(_REF_OK + (MockRule(0.15, scene=shifted), MockRule(0.6, scene=shifted.restore(MAT))), 0.9)
    res = route(_incident(shifted), backend, CFG, healthy_inputs())
    r = res.replay(V.restore_id(MAT))
    assert r.n == 30 and r.outcome is Outcome.INCONCLUSIVE
    assert res.verdict.row == 10 and "extend seeds" in res.verdict.notes


# --------------------------------------------------------------------------- #
# Stage 2 branch split
# --------------------------------------------------------------------------- #

def _ids(res):
    return [r.replay_id for r in res.replays]


def test_within_band_runs_branch_c_only():
    res = run_case(CASES["vision_glossy"])
    assert res.branch is Branch.C and "branch_C" in res.stages and "branch_G" not in res.stages
    assert V.R_G2 not in _ids(res) and V.R_C1 in _ids(res)


def test_clearly_failed_runs_branch_g_only():
    res = run_case(CASES["code_global"])
    assert res.branch is Branch.G and "branch_G" in res.stages and "branch_C" not in res.stages
    assert _ids(res)[:2] == [V.R_STAGE0, V.R_STAGE2_NOMINAL]
    assert {V.R_G1, V.R_G2, V.R_G3} <= set(_ids(res)) and V.R_C1 not in _ids(res)
    assert res.verdict.location == "action_side" and not res.verdict.by_elimination


def test_degraded_runs_both_branches_and_flags_multi_cause():
    res = run_case(CASES["combined_code_plus_light"])
    assert res.branch is Branch.G_DEGRADED and "branch_G" in res.stages and "branch_C" in res.stages
    v = res.verdict
    assert v.label is VerdictLabel.CODE and v.multi_cause_flag and v.likely_second_cause


def test_unknown_branch_runs_both_branches():
    shifted = SceneSpec("pick_mug", (Factor(MAT, "glossy_02"),))
    backend = QuotaBackend(_REF_OK + (MockRule(0.15, scene=shifted), MockRule(0.6, scene=SceneSpec("pick_mug"))), 0.9)
    res = route(_incident(shifted), backend, CFG, healthy_inputs())
    assert res.branch is Branch.UNKNOWN and {V.R_G2, V.R_G3, V.R_C1} <= set(_ids(res))
    assert res.verdict.row == 10


# --------------------------------------------------------------------------- #
# Branch G
# --------------------------------------------------------------------------- #

def test_branch_g_uses_the_right_actors_and_modes():
    res = run_case(CASES["code_global"])
    g1, g2, g3 = (res.replay(i) for i in (V.R_G1, V.R_G2, V.R_G3))
    assert g1.actor.kind is ActorKind.POLICY and g1.mode is Mode.INCIDENT
    assert g2.actor == ActorRef(ActorKind.REFERENCE, "ref_v1", ActionPipeline.RAW) and g2.mode is Mode.CLEAN
    assert g3.actor == ActorRef(ActorKind.REFERENCE, "ref_v1", ActionPipeline.WRAPPER)
    assert g2.scene == g3.scene == SceneSpec("pick_mug")


def test_g1_skipped_when_the_incident_was_observed_in_clean_mode():
    case = CASES["code_global"]
    inc = Incident(case.name, case.scene, TrainingSpec("p", **ACTION_SPEC), case.code_ref, POLICY, range(10), Mode.CLEAN,
                   log_refs=log_refs(case.name))
    res = route(inc, case.quota_backend(), CFG, case.inputs())
    assert V.R_G1 not in _ids(res) and {V.R_G2, V.R_G3} <= set(_ids(res))
    assert res.verdict.label is VerdictLabel.CODE


# --------------------------------------------------------------------------- #
# Branch C
# --------------------------------------------------------------------------- #

def test_every_factor_restore_runs_even_after_one_restores():
    """No early stop across hypotheses (3.2.1 #2)."""
    shifted = SceneSpec("pick_mug", (Factor(LIGHT, "dark_01"), Factor(MAT, "glossy_02")))
    backend = QuotaBackend(_REF_OK + (MockRule(0.15, factors_present=(MAT,)),), 0.9)
    res = route(_incident(shifted), backend, CFG, healthy_inputs())
    ids = _ids(res)
    assert V.restore_id(LIGHT) in ids and V.restore_id(MAT) in ids and V.R_C1 in ids
    assert V.R_C_ALL not in ids                     # a single factor restored
    assert res.verdict.label is VerdictLabel.VISION_FACTOR and res.verdict.factor == MAT


def test_all_factors_restore_runs_only_when_no_single_restores():
    res = run_case(CASES["vision_interacting"])
    assert V.R_C_ALL in _ids(res)
    assert res.replay(V.R_C_ALL).scene.is_nominal
    one_factor = run_case(CASES["vision_glossy"])
    assert V.R_C_ALL not in _ids(one_factor)        # one factor: C_all is never needed


def test_restore_replays_change_only_that_factor():
    res = run_case(CASES["vision_interacting"])
    for f in (LIGHT, MAT):
        r = res.replay(V.restore_id(f))
        assert f not in r.scene.factor_names and set(r.scene.factor_names) == {LIGHT, MAT} - {f}
        assert r.code_ref == CASES["vision_interacting"].code_ref and r.actor == POLICY


def test_render_replay_runs_only_with_render_evidence():
    case = CASES["rendering_mismatch"]
    res = run_case(case)
    cr = res.replay(V.R_C_RENDER)
    assert cr is not None and cr.code_ref.eval_config["render"] == TRAIN_RENDER and cr.scene == case.scene
    quiet = route(case.incident(), case.quota_backend(), CFG, RouterInputs())
    assert V.R_C_RENDER not in _ids(quiet)


def test_render_evidence_without_training_render_settings_is_noted_and_undetermined():
    """Externally supplied render evidence (the config contract cannot fire
    without training render settings)."""
    case = CASES["rendering_mismatch"]
    inc = Incident(case.name, case.scene, TrainingSpec("p", **ACTION_SPEC), CODE_EVAL_RENDER, POLICY, range(10),
                   log_refs=log_refs(case.name))
    ext = (Evidence("ext_render", "external", V.SIG_RENDER_SETTINGS_DIFFER, 0.9, "render", "supplied"),)
    res = route(inc, case.quota_backend(), CFG, case.inputs(), extra_evidence=ext)
    assert V.R_C_RENDER not in _ids(res)
    assert any("C_render not run" in n for n in res.notes)
    assert res.verdict.row == 10 and "C_render" in res.verdict.notes


def test_extra_evidence_is_passed_to_decide_and_recorded_as_external():
    case = CASES["code_global"]
    ext = (Evidence("ext_contract", "external", V.SIG_CONTRACT_VIOLATION, 0.9, "wrapper.gains", "supplied"),)
    res = route(case.incident(), case.quota_backend(), CFG, case.inputs(), extra_evidence=ext)
    assert res.external_evidence == ext and ext[0] not in res.stage1_evidence + res.behavior_evidence
    assert res.evidence == res.stage1_evidence + ext       # Branch G: no C_behavior
    assert "ext_contract" in res.verdict.evidence_ids and res.verdict.location == "wrapper.gains"


def test_plain_backend_is_wrapped_in_a_cache():
    case = CASES["vision_glossy"]
    res = route(case.incident(), case.quota_backend(), CFG, case.inputs())
    assert res.episodes_simulated == res.episodes_used - res.cache_hits


def test_incident_nominal_scene_round_trips():
    inc = CASES["action_gap"].incident()
    assert Incident.from_dict(inc.to_dict()) == inc
    plain = CASES["vision_glossy"].incident()
    assert Incident.from_dict(plain.to_dict()) == plain and plain.nominal_scene is None


def test_replay_config_validation():
    with pytest.raises(ValueError):
        ReplayConfig(0.8, (0.7, 0.95))
    with pytest.raises(ValueError):
        ReplayConfig(FAILURE_RATE, NOMINAL_BAND, n_initial=10, n_max=20, n_max_stage0=30)
    runner = ReplayRunner(CachedBackend(QuotaBackend((), 0.9), RolloutCache()), tuple(range(30)), CFG.replay)
    runner.run(V.R_STAGE0, SceneSpec("x"), CodeRef("c"), POLICY)
    with pytest.raises(ValueError):
        runner.run(V.R_STAGE0, SceneSpec("x"), CodeRef("c"), POLICY)
