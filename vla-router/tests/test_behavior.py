"""C_behavior (plan.md 3.2.7): implied target and failure phase, on synthetic traces."""

import pytest

from router import vocab as V
from router.behavior import (EV_PHASE, EV_TARGET, M_BIAS, BehaviorConfig, analyze_behavior, behavior_metrics,
                             run_behavior)
from router.behavior_synthetic import (BehaviorFaults, make_behavior_nominal, make_behavior_source,
                                       make_behavior_trace, make_behavior_traces)
from router.decide import decide
from router.prereq.behavior_trace import (BehaviorTrace, GraspRecord, InMemoryBehaviorSource, SubtaskProgress)
from router.prereq.trace import TraceUnavailable
from router.schemas import (ActionPipeline, ActorKind, ActorRef, CodeRef, EpisodeResult, Factor, Mode, Outcome,
                            ReplayResult, SceneSpec, VerdictLabel)
from router.stage1.nominal import MissingNominal, NominalStats
from router.stage1.synthetic import make_incident


@pytest.fixture(scope="module")
def nominal():
    return make_behavior_nominal()


def _signals(evs):
    return sorted(e.signal for e in evs)


def _analyze(nominal, faults, n=8, **kw):
    return analyze_behavior(make_behavior_traces(n, Mode.CLEAN, faults), nominal, BehaviorConfig(**kw))


# ---- implied target ---------------------------------------------------------

def test_healthy_grasp_is_ok(nominal):
    a = _analyze(nominal, BehaviorFaults(fail_phase=None))
    [e] = a.evidence
    assert (e.id, e.signal, e.source) == (EV_TARGET, V.SIG_IMPLIED_TARGET_OK, "behavior")
    assert 0.5 < e.strength <= 1.0


def test_consistent_offset_is_flagged_as_offset_not_wrong(nominal):
    a = _analyze(nominal, BehaviorFaults(offset=(0.0, 0.06, 0.0)))
    [e] = a.evidence
    assert e.signal == V.SIG_IMPLIED_TARGET_OFFSET and e.strength > 0.9
    assert e.location == "grasp.ee_offset"
    assert "(ee - target)" in e.detail and "(+0.0" in e.detail   # the direction is in the evidence, for the report
    assert a.mean_offset[1] == pytest.approx(0.06, abs=0.02)


def test_large_scatter_is_wrong(nominal):
    a = _analyze(nominal, BehaviorFaults(scatter=0.06))
    assert _signals(a.evidence) == [V.SIG_IMPLIED_TARGET_WRONG]


def test_wrong_object_is_wrong(nominal):
    a = _analyze(nominal, BehaviorFaults(wrong_object=0.6))
    assert _signals(a.evidence) == [V.SIG_IMPLIED_TARGET_WRONG]
    assert a.wrong_fraction > 0.3


def test_exactly_one_target_signal_never_ok_and_wrong(nominal):
    for faults in (BehaviorFaults(), BehaviorFaults(offset=(0.05, 0, 0)), BehaviorFaults(scatter=0.05),
                   BehaviorFaults(wrong_object=0.5), BehaviorFaults(offset=(0.05, 0, 0), scatter=0.05)):
        evs = [e for e in _analyze(nominal, faults).evidence if e.id == EV_TARGET]
        assert len(evs) <= 1


def test_too_few_grasps_is_not_judged(nominal):
    a = _analyze(nominal, BehaviorFaults(no_grasp=True, fail_phase="reach"))
    assert not any(e.id == EV_TARGET for e in a.evidence)
    assert any("never closed the gripper" in n for n in a.notes)


def test_only_failing_episodes_are_used(nominal):
    ok = make_behavior_traces(8, Mode.CLEAN, BehaviorFaults(), base_seed=100, success=True)
    bad = make_behavior_traces(8, Mode.CLEAN, BehaviorFaults(offset=(0.0, 0.06, 0.0)), base_seed=0)
    a = analyze_behavior(ok + bad, nominal)
    assert a.n_episodes == 8 and a.evidence[0].signal == V.SIG_IMPLIED_TARGET_OFFSET
    assert analyze_behavior(ok, nominal).evidence == ()


# ---- failure phase ----------------------------------------------------------

@pytest.mark.parametrize("phase", ["reach", "grasp", "lift", "place"])
def test_failure_phase_is_the_first_incomplete_subtask(nominal, phase):
    a = _analyze(nominal, BehaviorFaults(fail_phase=phase))
    [e] = [e for e in a.evidence if e.id == EV_PHASE]
    assert e.signal == V.failure_phase(phase) and e.strength == 1.0


def test_phase_needs_a_clear_majority(nominal):
    mixed = (make_behavior_traces(4, Mode.CLEAN, BehaviorFaults(fail_phase="grasp"))
             + make_behavior_traces(4, Mode.CLEAN, BehaviorFaults(fail_phase="lift"), base_seed=50))
    a = analyze_behavior(mixed, nominal)
    assert not any(e.id == EV_PHASE for e in a.evidence) and any("tie" in n for n in a.notes)
    three_one = (make_behavior_traces(5, Mode.CLEAN, BehaviorFaults(fail_phase="grasp"))
                 + make_behavior_traces(3, Mode.CLEAN, BehaviorFaults(fail_phase="lift"), base_seed=50))
    a = analyze_behavior(three_one, nominal, BehaviorConfig(min_phase_share=0.7))
    assert not any(e.id == EV_PHASE for e in a.evidence) and any("holds" in n for n in a.notes)


def test_no_subtask_log_means_no_phase_evidence(nominal):
    a = _analyze(nominal, BehaviorFaults(fail_phase=None))
    assert not any(e.id == EV_PHASE for e in a.evidence) and any("failure phase not judged" in n for n in a.notes)


def test_failure_phase_unit():
    t = BehaviorTrace(1, "clean", False, None, tuple(SubtaskProgress(p, p in ("reach", "grasp")) for p in
                                                      ("reach", "grasp", "lift", "place")))
    assert t.failure_phase() == "lift"
    assert BehaviorTrace(1, "clean", True, None, t.subtasks).failure_phase() is None
    with pytest.raises(ValueError):
        SubtaskProgress("push", True)


# ---- integration with decide: the signals mean what rows 4 / 8 / 9 expect -----

def _table(scene, stage2=Outcome.RESTORED, restore=Outcome.NO_EFFECT):
    """Branch C, reference solves the shifted scene, no factor restores (the bucket rows 8 and 9 judge)."""
    code, actor = CodeRef("c0"), ActorRef(ActorKind.POLICY, "ckpt_v1")
    ref = ActorRef(ActorKind.REFERENCE, "ref_v1", ActionPipeline.RAW)

    def rep(rid, outcome, sc=None, who=actor):
        return ReplayResult(rid, sc or scene, code, who, Mode.CLEAN, (EpisodeResult(0, outcome is Outcome.RESTORED),),
                            outcome)

    nominal = SceneSpec(scene.nominal)
    out = [rep(V.R_STAGE0, Outcome.NO_EFFECT), rep(V.R_STAGE2_NOMINAL, stage2, nominal),
           rep(V.R_C1, Outcome.RESTORED, who=ref), rep(V.R_G2, Outcome.RESTORED, nominal, ref),
           rep(V.R_G3, Outcome.RESTORED, nominal, ref)]
    out += [rep(V.restore_id(f), restore, scene.restore(f)) for f in scene.factor_names]
    return out


SCENE = SceneSpec("pick_mug", (Factor("lighting", "dark_01"),))


def test_offset_routes_to_code_row_4(nominal):
    evs = _analyze(nominal, BehaviorFaults(offset=(0.0, 0.06, 0.0), fail_phase="grasp")).evidence
    v = decide(list(evs), _table(SCENE))
    assert (v.row, v.label) == (4, VerdictLabel.CODE) and not v.by_elimination
    assert v.location == "grasp.ee_offset" and EV_TARGET in v.evidence_ids


def test_ok_target_and_grasp_phase_routes_to_action_gap_row_8(nominal):
    evs = _analyze(nominal, BehaviorFaults(fail_phase="grasp")).evidence
    assert _signals(evs) == sorted([V.SIG_IMPLIED_TARGET_OK, V.failure_phase("grasp")])
    v = decide(list(evs), _table(SCENE))
    assert (v.row, v.label) == (8, VerdictLabel.ACTION_GAP)


def test_wrong_target_routes_to_perception_unresolved_row_9(nominal):
    evs = _analyze(nominal, BehaviorFaults(scatter=0.06, fail_phase="reach")).evidence
    v = decide(list(evs), _table(SCENE))
    assert (v.row, v.label) == (9, VerdictLabel.PERCEPTION_UNRESOLVED)


def test_reach_failure_with_ok_target_is_not_an_action_gap(nominal):
    """'reach' is not in PHASES_ACTION_GAP: decide must not call it row 8 (undetermined instead)."""
    evs = _analyze(nominal, BehaviorFaults(fail_phase="reach")).evidence
    assert decide(list(evs), _table(SCENE)).row == 10


# ---- run_behavior / sources ---------------------------------------------------

def test_run_behavior_uses_the_logs_of_the_chosen_mode(nominal):
    src = make_behavior_source(clean=BehaviorFaults(offset=(0.0, 0.06, 0.0)), incident=BehaviorFaults(scatter=0.06))
    inc = make_incident()
    assert run_behavior(inc, src, nominal)[0].signal == V.SIG_IMPLIED_TARGET_OFFSET
    assert run_behavior(inc, src, nominal, BehaviorConfig(mode=Mode.INCIDENT))[0].signal == V.SIG_IMPLIED_TARGET_WRONG


def test_missing_logs_raise_not_silently_pass(nominal):
    inc = make_incident(log_refs={})
    with pytest.raises(TraceUnavailable):
        run_behavior(inc, make_behavior_source(), nominal)
    with pytest.raises(TraceUnavailable):
        run_behavior(make_incident(), InMemoryBehaviorSource({}), nominal)


def test_missing_nominal_fails_at_the_call():
    with pytest.raises(MissingNominal):
        analyze_behavior(make_behavior_traces(4), NominalStats({}))


def test_wrong_mode_traces_are_rejected():
    src = InMemoryBehaviorSource({"clean": make_behavior_traces(3, Mode.INCIDENT)})
    with pytest.raises(ValueError):
        src.load(make_incident(), Mode.CLEAN)


def test_records_round_trip():
    t = make_behavior_trace(3, Mode.CLEAN, BehaviorFaults(offset=(0.1, 0, 0), fail_phase="lift"))
    assert BehaviorTrace.from_dict(t.to_dict()) == t
    with pytest.raises(ValueError):
        GraspRecord(1, (0, 0, 0), (0, 0))


def test_metrics_empty_without_grasps():
    assert behavior_metrics(make_behavior_traces(3, faults=BehaviorFaults(no_grasp=True))) == {}
    assert set(behavior_metrics(make_behavior_traces(3))) >= {M_BIAS}


def test_a_part_that_was_not_judged_names_its_unchecked_signals():
    """ISSUES #59: each "not judged" note is also a PartialSkip with the signals it leaves unchecked."""
    from router import vocab as V
    from router.behavior_synthetic import BehaviorFaults, make_behavior_nominal, make_behavior_traces
    from router.behavior import analyze_behavior
    nominal = make_behavior_nominal()
    a = analyze_behavior(make_behavior_traces(6, faults=BehaviorFaults(no_grasp=True)), nominal)
    by_note = {p.reason: p.signals for p in a.partial}
    assert len(a.partial) == len(a.notes) == 2
    assert next(s for r, s in by_note.items() if r.startswith("implied target")) == {
        V.SIG_IMPLIED_TARGET_OFFSET, V.SIG_IMPLIED_TARGET_OK, V.SIG_IMPLIED_TARGET_WRONG}
    assert next(s for r, s in by_note.items() if r.startswith("failure phase")) == {
        V.failure_phase(p) for p in V.PHASES}
    done = analyze_behavior(make_behavior_traces(6, faults=BehaviorFaults(fail_phase="grasp")), nominal)
    assert done.partial == () and done.evidence
    empty = analyze_behavior(make_behavior_traces(6, success=True), nominal)
    assert [p.signals for p in empty.partial] == [V.SIGNALS_BY_SOURCE["behavior"]]
