"""Router change 4: a backend without a reference controller through the wrapper (sim side
decision 3) raises ``ReplayUnavailable`` for G3. G3 is then recorded as unavailable, never as a
result; a row 4 reached by elimination carries ``g3_unavailable``; every other
``NotImplementedError`` still ends the incident UNRESOLVED."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ladder.backend import MockBackend, ReplayUnavailable, wrapper_reference
from ladder.loop import LoopStatus
from router import vocab as V
from router.decide import decide
from router.prereq.rollout_cache import CachedBackend, RolloutCache
from router.replay import ReplayRunner, seed_set
from router.report import verdict_text
from router.router import route
from router.schemas import (ActionPipeline, ActorKind, ActorRef, CodeRef, Evidence, Mode, Outcome, ReplayResult,
                            SceneSpec, Verdict, VerdictLabel)
from tests.ladder_cases import PATCHED, ScriptedRepairer, make_loop
from tests.router_cases import QuotaBackend, default_config, router_cases

CASES = {c.name: c for c in router_cases()}
CFG = default_config()
REF_RAW = ActorRef(ActorKind.REFERENCE, "ref_v1", ActionPipeline.RAW)
REF_WRAP = ActorRef(ActorKind.REFERENCE, "ref_v1", ActionPipeline.WRAPPER)
POLICY = ActorRef(ActorKind.POLICY, "ckpt_v1")
CONTRACT = Evidence("ext_contract", "external", V.SIG_CONTRACT_VIOLATION, 0.9, "obs.normalization")


def routed(name: str, unavailable: bool, **kw):
    c = CASES[name]
    return route(c.incident(), c.quota_backend(wrapper_reference_unavailable=unavailable), CFG, c.inputs(), **kw)


def same_verdict(a: Verdict, b: Verdict) -> None:
    assert (a.label, a.row, a.destination, a.location, a.factor, a.by_elimination, a.likely_second_cause,
            a.multi_cause_flag, a.evidence_ids) == (b.label, b.row, b.destination, b.location, b.factor,
                                                    b.by_elimination, b.likely_second_cause, b.multi_cause_flag,
                                                    b.evidence_ids)


# --------------------------------------------------------------------------- #
# the backend side
# --------------------------------------------------------------------------- #

def test_the_mock_refuses_only_the_reference_through_the_wrapper_and_simulates_nothing():
    assert issubclass(ReplayUnavailable, NotImplementedError)
    assert wrapper_reference(REF_WRAP) and not wrapper_reference(REF_RAW) and not wrapper_reference(POLICY)
    b = MockBackend([], 0.5, wrapper_reference_unavailable=True)
    with pytest.raises(ReplayUnavailable, match="G3"):
        b.run(SceneSpec("n"), CodeRef("c0"), REF_WRAP, [0, 1], Mode.CLEAN)
    assert not b.calls
    assert len(b.run(SceneSpec("n"), CodeRef("c0"), REF_RAW, [0, 1], Mode.CLEAN)) == 2
    assert len(MockBackend([], 0.5).run(SceneSpec("n"), CodeRef("c0"), REF_WRAP, [0], Mode.CLEAN)) == 1


def test_a_refused_run_counts_no_cache_misses_and_caches_nothing():
    cache = RolloutCache()
    cached = CachedBackend(MockBackend([], 0.5, wrapper_reference_unavailable=True), cache)
    with pytest.raises(ReplayUnavailable):
        cached.run(SceneSpec("n"), CodeRef("c0"), REF_WRAP, [0, 1, 2], Mode.CLEAN)
    assert (cached.misses, cached.hits, len(list(cache.keys()))) == (0, 0, 0)


def test_the_runner_records_an_unavailable_replay_once_and_never_as_a_result():
    runner = ReplayRunner(CachedBackend(MockBackend([], 0.5), RolloutCache()), seed_set((), 30), CFG.replay)
    runner.mark_unavailable(V.R_G3, "no wrapper reference")
    assert runner.unavailable == {V.R_G3: "no wrapper reference"} and runner.results == []
    with pytest.raises(ValueError, match="already run"):
        runner.mark_unavailable(V.R_G3, "again")
    with pytest.raises(ValueError, match="already run"):
        runner.run(V.R_G3, SceneSpec("n"), CodeRef("c0"), REF_WRAP)


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name,extra", [("combined_code_plus_light", ()), ("code_global", (CONTRACT,))])
def test_branch_g_with_a_located_contract_hit_gets_the_same_verdict_with_g3_unavailable(name, extra):
    full, cut = routed(name, False, extra_evidence=extra), routed(name, True, extra_evidence=extra)
    assert "branch_G" in cut.stages and cut.branch is full.branch
    assert full.replay(V.R_G3) is not None and cut.replay(V.R_G3) is None
    assert cut.unavailable == ((V.R_G3, "mock: no reference controller through the wrapper (G3)"),)
    same_verdict(full.verdict, cut.verdict)
    assert cut.verdict.label is VerdictLabel.CODE and not cut.verdict.by_elimination
    assert not cut.verdict.g3_unavailable and "confidence=higher (located cause)" in cut.verdict.notes
    assert "replays unavailable: G3 (" in cut.verdict.notes
    assert cut.episodes_simulated == full.episodes_simulated - full.replay(V.R_G3).n


@pytest.mark.parametrize("name", ["code_global", "code_partial_offset"])
def test_branch_g_without_a_contract_hit_is_code_by_elimination_low_confidence_and_flagged(name):
    full, cut = routed(name, False), routed(name, True)
    v = cut.verdict
    assert not [e for e in cut.evidence if e.signal in (V.SIG_CONTRACT_VIOLATION, V.SIG_IMPLIED_TARGET_OFFSET)]
    assert (v.label, v.row, v.by_elimination, v.g3_unavailable) == (VerdictLabel.CODE, 4, True, True)
    assert v.location is None                  # "observation side" needs G3 restored; it is not
    assert "confidence=low (elimination only, G3 unavailable)" in v.notes
    assert V.R_G3 not in {r.replay_id for r in v.replays}
    assert "G3 (reference through the wrapper) was unavailable" in verdict_text(v)
    assert Verdict.from_dict(json.loads(json.dumps(v.to_dict()))) == v
    assert not full.verdict.g3_unavailable     # with G3 run, the same incident is not flagged


def test_branch_c_is_unaffected():
    for name, c in CASES.items():
        full = route(c.incident(), c.quota_backend(), CFG, c.inputs())
        if full.branch is None or "branch_G" in full.stages:
            continue
        cut = routed(name, True)
        assert cut.verdict == full.verdict and cut.replays == full.replays and cut.unavailable == ()
        assert not cut.verdict.g3_unavailable


class PlainNotImplemented(QuotaBackend):
    """Raises a plain NotImplementedError (not ReplayUnavailable) for one kind of actor."""

    def __init__(self, *a, refuse=wrapper_reference, exc=NotImplementedError, **kw):
        super().__init__(*a, **kw)
        self.refuse, self.exc = refuse, exc

    def run(self, scene, code_ref, actor, seeds, mode):
        if self.refuse(actor):
            raise self.exc("not built")
        return super().run(scene, code_ref, actor, seeds, mode)


@pytest.mark.parametrize("refuse,exc", [
    (wrapper_reference, NotImplementedError),                                         # G3, but not the marker type
    (lambda a: a.pipeline is ActionPipeline.RAW, NotImplementedError),                # G2
    (lambda a: a.pipeline is ActionPipeline.RAW, ReplayUnavailable),                  # the marker outside G3
], ids=["g3_plain", "g2_plain", "g2_marker"])
def test_any_other_not_implemented_error_still_ends_the_incident_unresolved(refuse, exc):
    c = CASES["code_global"]
    with pytest.raises(exc):
        route(c.incident(), PlainNotImplemented(c.rules, c.default_rate, refuse=refuse, exc=exc), CFG, c.inputs())
    loop, _ = make_loop(c, ScriptedRepairer([PATCHED]),
                        backend=PlainNotImplemented(c.rules, c.default_rate, refuse=refuse, exc=exc))
    res = loop.run(c.incident())
    assert res.status is LoopStatus.UNRESOLVED and res.error_stage == "router"
    assert res.error_type == exc.__name__


def test_the_loop_records_the_flag_per_pass():
    c = CASES["code_global"]
    loop, _ = make_loop(c, ScriptedRepairer([PATCHED]), backend=c.quota_backend(wrapper_reference_unavailable=True))
    res = loop.run(c.incident())
    assert res.passes[0].g3_unavailable and res.passes[0].by_elimination
    assert res.to_dict()["passes"][0]["g3_unavailable"] is True


# --------------------------------------------------------------------------- #
# decide
# --------------------------------------------------------------------------- #

def _r(rid: str, outcome: Outcome, actor=POLICY, scene=SceneSpec("n")) -> ReplayResult:
    return ReplayResult(rid, scene, CodeRef("c0"), actor, Mode.CLEAN, (), outcome)


BRANCH_G = [_r(V.R_STAGE0, Outcome.NO_EFFECT, scene=SceneSpec("n", ())), _r(V.R_STAGE2_NOMINAL, Outcome.NO_EFFECT),
            _r(V.R_G2, Outcome.RESTORED, REF_RAW)]


def test_decide_reads_unavailable_g3_as_neither_pass_nor_fail():
    failing = decide([], BRANCH_G + [_r(V.R_G3, Outcome.NO_EFFECT, REF_WRAP)])
    passing = decide([], BRANCH_G + [_r(V.R_G3, Outcome.RESTORED, REF_WRAP)])
    gone = decide([], BRANCH_G, unavailable={V.R_G3})
    missing = decide([], BRANCH_G)
    assert (failing.location, failing.by_elimination) == ("action_side", False)            # G3 fail: located
    assert passing.location == "observation_side_or_policy_loading" and passing.by_elimination
    assert (gone.row, gone.by_elimination, gone.location, gone.g3_unavailable) == (4, True, None, True)
    assert replace(gone, g3_unavailable=False, notes="") == replace(missing, notes="")
    assert not failing.g3_unavailable and not passing.g3_unavailable and not missing.g3_unavailable


def test_decide_flags_only_a_row_4_by_elimination():
    located = decide([CONTRACT], BRANCH_G, unavailable={V.R_G3})
    assert (located.row, located.by_elimination, located.g3_unavailable) == (4, False, False)
    sim = decide([], BRANCH_G[:2] + [_r(V.R_G2, Outcome.NO_EFFECT, REF_RAW)], unavailable={V.R_G3})
    assert sim.row == 2 and not sim.g3_unavailable


def test_decide_validates_unavailable():
    with pytest.raises(TypeError, match="single string"):
        decide([], BRANCH_G, unavailable="G3")
    with pytest.raises(ValueError, match="may be unavailable"):
        decide([], BRANCH_G[:2], unavailable={V.R_G2})
    with pytest.raises(ValueError, match="both run and unavailable"):
        decide([], BRANCH_G + [_r(V.R_G3, Outcome.RESTORED, REF_WRAP)], unavailable={V.R_G3})


def test_old_verdict_dicts_load_without_the_flag():
    d = decide([], BRANCH_G).to_dict()
    del d["g3_unavailable"]
    assert Verdict.from_dict(d).g3_unavailable is False
