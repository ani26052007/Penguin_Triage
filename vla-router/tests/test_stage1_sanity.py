"""Stage 1 check 5: render / physics sanity (plan.md 3.2.4)."""

from dataclasses import replace

import pytest

from router import vocab as V
from router.prereq.trace import InMemoryTraceSource, TraceUnavailable
from router.schemas import Mode
from router.stage1.nominal import MissingNominal, NominalStats
from router.stage1.sanity import SanityCheck
from router.stage1.synthetic import Faults, make_incident, make_nominal, make_source, make_traces

NOMINAL = make_nominal()
INCIDENT = make_incident()


def run(faults=Faults(), **kw):
    return SanityCheck(NOMINAL, make_source(faults), **kw).run(INCIDENT)


def ids(found):
    return sorted(e.id for e in found)


def test_clean_traces_have_no_evidence():
    assert run() == []


@pytest.mark.parametrize("faults, eid, location", [
    (Faults(black_render=True), "s1.sanity.black_frames", "render.black_frames"),
    (Faults(saturated_render=True), "s1.sanity.saturated_frames", "render.saturation"),
    (Faults(missing_texture=True), "s1.sanity.missing_texture", "render.missing_texture"),
    (Faults(nan_state=True), "s1.sanity.nan_state", "sim.nan_state"),
    (Faults(interpenetration=True), "s1.sanity.interpenetration", "sim.interpenetration"),
    (Faults(rest_shift=True), "s1.sanity.rest_pose", "sim.rest_pose"),
])
def test_each_planted_fault_is_found_and_only_that_one(faults, eid, location):
    found = run(faults)
    assert eid in ids(found)
    e = next(e for e in found if e.id == eid)
    assert e.source == "sanity" and e.signal == V.SIG_RENDER_PHYSICS_BROKEN
    assert e.location == location and 0.5 <= e.strength <= 1.0


def test_policy_input_faults_do_not_look_like_a_sim_fault():
    # a normalization or channel bug in the policy input is code (row 4), not a broken render (row 2)
    for faults in (Faults(unnormalized=True), Faults(swap_rb=True), Faults(vflip=True), Faults(wrong_resolution=True)):
        assert run(faults) == []


def test_one_bad_episode_of_eight_is_not_a_scene_fault():
    traces = make_traces(7, Mode.CLEAN) + make_traces(1, Mode.CLEAN, Faults(black_render=True), base_seed=500)
    src = InMemoryTraceSource({"clean": traces})
    assert SanityCheck(NOMINAL, src).run(INCIDENT) == []


def test_reads_the_requested_mode():
    src = make_source(clean=Faults(), incident=Faults(nan_state=True))
    assert SanityCheck(NOMINAL, src, mode=Mode.CLEAN).run(INCIDENT) == []
    assert ids(SanityCheck(NOMINAL, src, mode=Mode.INCIDENT).run(INCIDENT)) == ["s1.sanity.nan_state"]


def test_traces_without_render_stats_skip_the_render_metrics():
    traces = make_traces(6, Mode.CLEAN, Faults(nan_state=True))
    stripped = [replace(t, frames=[replace(f, render=None) for f in t.frames]) for t in traces]
    assert ids(SanityCheck(NOMINAL, InMemoryTraceSource({"clean": stripped})).run(INCIDENT)) == ["s1.sanity.nan_state"]


def test_missing_traces_raise_and_missing_nominal_fails_early():
    with pytest.raises(TraceUnavailable):
        SanityCheck(NOMINAL, make_source()).run(make_incident(log_refs={}))
    with pytest.raises(MissingNominal):
        SanityCheck(NominalStats({}), make_source())
