import math

import pytest

from ladder.stats import (
    Proportion, StageDecision, StagedPlan, agreement_rate, classify_outcome, clearly_better,
    compare, fisher_exact, mcnemar_exact, replay_outcome, staged_decision,
    staged_operating_characteristics, wilson_interval,
)
from router.schemas import Outcome

F, BAND = 0.2, (0.7, 0.9)   # failure anchor and nominal band used across tests


# ---------------------------------------------------------------- wilson ----

@pytest.mark.parametrize("k,n,lo,hi", [
    (0, 10, 0.0, 0.2775),     # textbook Wilson 95% values
    (10, 10, 0.7225, 1.0),
    (5, 10, 0.2366, 0.7634),
    (8, 10, 0.4902, 0.9433),
    (2, 10, 0.0567, 0.5098),
])
def test_wilson_known_values(k, n, lo, hi):
    p = wilson_interval(k, n)
    assert p.lo == pytest.approx(lo, abs=1e-4)
    assert p.hi == pytest.approx(hi, abs=1e-4)
    assert p.lo <= p.rate <= p.hi


def test_wilson_extremes_exact_and_empty():
    assert wilson_interval(0, 7).lo == 0.0
    assert wilson_interval(7, 7).hi == 1.0
    e = wilson_interval(0, 0)
    assert (e.lo, e.hi) == (0.0, 1.0)


def test_wilson_narrows_with_n_and_widens_with_confidence():
    w10 = wilson_interval(5, 10)
    w100 = wilson_interval(50, 100)
    assert (w100.hi - w100.lo) < (w10.hi - w10.lo)
    w975 = wilson_interval(5, 10, 0.975)
    assert w975.lo < w10.lo and w975.hi > w10.hi


@pytest.mark.parametrize("k,n,conf", [(-1, 5, 0.95), (6, 5, 0.95), (1, 5, 1.0), (1, 5, 0.0)])
def test_wilson_rejects_bad_input(k, n, conf):
    with pytest.raises(ValueError):
        wilson_interval(k, n, conf)


# ------------------------------------------------------- three-valued ----

@pytest.mark.parametrize("k,n,fail,band,expected", [
    (10, 10, F, BAND, Outcome.RESTORED),
    (8, 10, F, BAND, Outcome.RESTORED),
    (7, 10, F, BAND, Outcome.RESTORED),          # point rate exactly on band low edge
    (2, 10, F, BAND, Outcome.NO_EFFECT),
    (0, 10, F, BAND, Outcome.NO_EFFECT),
    (15, 30, F, BAND, Outcome.PARTIAL),          # (0.33, 0.67): above failure, below band
    (5, 10, F, BAND, Outcome.INCONCLUSIVE),      # rate 0.5 < 0.7 but interval reaches band
    (4, 10, 0.3, (0.6, 0.9), Outcome.INCONCLUSIVE),   # includes failure AND reaches band
    (40, 100, 0.3, (0.6, 0.9), Outcome.PARTIAL),      # same rate, more seeds -> decided
    (0, 30, 0.5, (0.8, 1.0), Outcome.NO_EFFECT),      # entirely below the failure rate
    (30, 30, F, (0.5, 0.6), Outcome.RESTORED),        # entirely above the band
])
def test_classify_known_cases(k, n, fail, band, expected):
    outcome, _ = replay_outcome(k, n, fail, band)
    assert outcome is expected


def test_classify_borderline_intervals_are_inclusive():
    # interval endpoint exactly on the failure rate -> it includes it
    assert classify_outcome(Proportion(7, 10, 0.2, 0.9, 0.95), F, BAND) is Outcome.INCONCLUSIVE
    assert classify_outcome(Proportion(7, 10, 0.2000001, 0.9, 0.95), F, BAND) is Outcome.RESTORED
    # point rate just below vs on the band low edge
    assert classify_outcome(Proportion(69, 100, 0.5, 0.78, 0.95), F, BAND) is Outcome.INCONCLUSIVE
    assert classify_outcome(Proportion(70, 100, 0.5, 0.78, 0.95), F, BAND) is Outcome.RESTORED
    # upper end just below vs exactly on the band low edge
    assert classify_outcome(Proportion(60, 100, 0.5, 0.6999, 0.95), F, BAND) is Outcome.PARTIAL
    assert classify_outcome(Proportion(60, 100, 0.5, 0.7, 0.95), F, BAND) is Outcome.INCONCLUSIVE


def test_inconclusive_resolves_when_seeds_are_extended():
    # same 50% rate: straddles at 10 seeds, clearly partial at 100
    assert replay_outcome(5, 10, F, BAND)[0] is Outcome.INCONCLUSIVE
    assert replay_outcome(50, 100, F, BAND)[0] is Outcome.PARTIAL


@pytest.mark.parametrize("fail,band", [(0.8, (0.7, 0.9)), (0.2, (0.9, 0.7)), (0.2, (0.5, 1.2))])
def test_classify_rejects_bad_anchors(fail, band):
    with pytest.raises(ValueError):
        classify_outcome(wilson_interval(5, 10), fail, band)


# -------------------------------------------------------- exact tests ----

def test_mcnemar_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(8, 0) == pytest.approx(2 / 256)
    assert mcnemar_exact(6, 1) == pytest.approx(2 * 8 / 128)
    assert mcnemar_exact(1, 6) == mcnemar_exact(6, 1)


def test_fisher_known_values():
    assert fisher_exact(3, 4, 1, 4) == pytest.approx(0.4857142857)     # lady tasting tea
    assert fisher_exact(10, 10, 2, 10) == pytest.approx(2 * 45 / math.comb(20, 12))
    assert fisher_exact(5, 10, 5, 10) == pytest.approx(1.0)
    assert fisher_exact(3, 4, 1, 4) == fisher_exact(1, 4, 3, 4)


def test_fisher_matches_scipy_when_available():
    stats = pytest.importorskip("scipy.stats")
    for k1, n1, k2, n2 in [(2, 10, 9, 10), (4, 20, 14, 20), (0, 15, 6, 12), (7, 30, 11, 25)]:
        ref = stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue
        assert fisher_exact(k1, n1, k2, n2) == pytest.approx(ref, rel=1e-9)
    for b, c in [(6, 0), (7, 2), (3, 9)]:
        ref = stats.binomtest(min(b, c), b + c, 0.5).pvalue
        assert mcnemar_exact(b, c) == pytest.approx(ref, rel=1e-9)


# -------------------------------------------- paired vs unpaired switch ----

def _runs(before_succ, after_succ, seeds):
    return ({s: s in before_succ for s in seeds}, {s: s in after_succ for s in seeds})


def test_agreement_rate():
    a = {0: True, 1: False, 2: True, 3: True}
    b = {0: True, 1: True, 2: True, 3: False, 99: True}   # 99 not shared, ignored
    assert agreement_rate(a, b) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        agreement_rate({1: True}, {2: True})


def test_paired_used_when_agreement_high_and_seeds_match():
    seeds = range(20)
    before, after = _runs(set(range(10)), set(range(16)), seeds)   # 10/20 -> 16/20, 6 one-way flips
    c = compare(before, after, agreement=0.95, min_agreement=0.9)
    assert c.method == "paired"
    assert c.discordant == (6, 0)
    assert c.p_value == pytest.approx(0.03125)
    assert c.clearly_better


def test_unpaired_fallback_when_agreement_low_changes_the_answer():
    seeds = range(20)
    before, after = _runs(set(range(10)), set(range(16)), seeds)
    c = compare(before, after, agreement=0.6, min_agreement=0.9)
    assert c.method == "unpaired"
    assert c.discordant is None
    assert c.p_value == pytest.approx(fisher_exact(16, 20, 10, 20))
    assert not c.clearly_better            # same data, but pairing is no longer valid
    assert "agreement" in c.reason


def test_unpaired_when_seed_sets_differ_even_with_high_agreement():
    before = {s: s < 5 for s in range(20)}
    after = {s: s < 25 for s in range(10, 30)}
    c = compare(before, after, agreement=1.0, min_agreement=0.9)
    assert c.method == "unpaired"
    assert c.reason == "seed sets differ"


def test_agreement_exactly_at_threshold_is_paired():
    before, after = _runs({0}, {0, 1}, range(4))
    assert compare(before, after, agreement=0.9, min_agreement=0.9).method == "paired"


def test_clearly_better_needs_direction():
    seeds = range(20)
    before, after = _runs(set(range(16)), set(range(10)), seeds)   # got worse
    c = compare(before, after, agreement=1.0, min_agreement=0.9)
    assert c.significant and c.clearly_worse and not c.clearly_better
    assert not clearly_better(before, after, agreement=1.0, min_agreement=0.9)


# --------------------------------------------------------- staged seeds ----

PLAN = StagedPlan(n_a=10, n_b=50)


def test_staged_plan_splits_alpha():
    assert PLAN.alpha_a == pytest.approx(0.025)
    assert PLAN.alpha_b == pytest.approx(0.025)
    assert PLAN.confidence("A") == pytest.approx(0.975)
    strict_a = StagedPlan(10, 50, stage_a_fraction=0.2)
    assert strict_a.confidence("A") > PLAN.confidence("A")
    for bad in [dict(n_a=10, n_b=10), dict(n_a=10, n_b=50, alpha_total=0), dict(n_a=10, n_b=50, stage_a_fraction=1)]:
        with pytest.raises(ValueError):
            StagedPlan(**bad)


@pytest.mark.parametrize("k,expected", [
    (10, StageDecision.PASS), (8, StageDecision.PASS),
    (5, StageDecision.CONTINUE),           # straddles the band edge -> Stage B
    (2, StageDecision.FAIL), (0, StageDecision.FAIL),
])
def test_stage_a_decisions(k, expected):
    r = staged_decision(k, 10, "A", PLAN, failure_rate=F, nominal_band=BAND)
    assert r.decision is expected
    assert r.proportion.confidence == pytest.approx(0.975)


@pytest.mark.parametrize("k,expected", [
    (40, StageDecision.PASS),              # 0.8
    (25, StageDecision.AMBIGUOUS),         # 0.5: partial -> Rung 2
    (12, StageDecision.FAIL),              # 0.24: interval includes failure
])
def test_stage_b_decisions(k, expected):
    assert staged_decision(k, 50, "B", PLAN, failure_rate=F, nominal_band=BAND).decision is expected


def test_stage_a_interval_is_stricter_than_unadjusted():
    adj = staged_decision(5, 10, "A", PLAN, failure_rate=F, nominal_band=BAND).proportion
    raw = wilson_interval(5, 10, 0.95)
    assert adj.lo < raw.lo and adj.hi > raw.hi


def test_staged_rejects_wrong_seed_count():
    with pytest.raises(ValueError):
        staged_decision(5, 20, "A", PLAN, failure_rate=F, nominal_band=BAND)
    with pytest.raises(ValueError):
        staged_decision(5, 10, "C", PLAN, failure_rate=F, nominal_band=BAND)


@pytest.mark.parametrize("p", [0.0, 0.2, 0.45, 0.8, 1.0])
def test_operating_characteristics_are_probabilities(p):
    oc = staged_operating_characteristics(p, PLAN, failure_rate=F, nominal_band=BAND)
    assert oc.p_pass + oc.p_fail + oc.p_ambiguous == pytest.approx(1.0)
    assert PLAN.n_a <= oc.expected_seeds <= PLAN.n_b


@pytest.mark.parametrize("fail,band", [(0.1, (0.7, 0.9)), (0.2, (0.7, 0.9)), (0.3, (0.6, 0.95))])
def test_wrong_pass_and_wrong_fail_stay_within_alpha_across_both_looks(fail, band):
    """The early-look correction: error rates over Stage A + Stage B combined."""
    at_failure = staged_operating_characteristics(fail, PLAN, failure_rate=fail, nominal_band=band)
    assert at_failure.p_pass <= PLAN.alpha_total
    at_nominal = staged_operating_characteristics(sum(band) / 2, PLAN, failure_rate=fail, nominal_band=band)
    assert at_nominal.p_fail <= PLAN.alpha_total


def test_middle_value_patch_is_rarely_passed():
    """A true-45% patch against a 0.8 nominal should mostly end ambiguous or fail."""
    oc = staged_operating_characteristics(0.45, PLAN, failure_rate=F, nominal_band=BAND)
    assert oc.p_pass < 0.15
    nominal = staged_operating_characteristics(0.8, PLAN, failure_rate=F, nominal_band=BAND)
    assert nominal.p_pass > 0.95
    assert nominal.expected_seeds < 20       # a clear fix mostly stops at Stage A
