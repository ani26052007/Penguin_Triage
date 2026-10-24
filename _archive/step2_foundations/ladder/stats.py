"""Shared statistics for the router and Rung 1 (plan.md 3.2.9, Section 4).

One definition of "clearly better" for the whole system. Pure stdlib, no
simulator dependency.

Contents
--------
- ``wilson_interval``      proportion with a confidence interval
- ``classify_outcome``     restored / partial / no_effect (+ inconclusive = extend seeds)
- ``mcnemar_exact`` / ``fisher_exact``   paired and unpaired two-sided tests
- ``agreement_rate``       replay agreement between two runs of the same seeds
- ``compare`` / ``clearly_better``       picks paired vs unpaired from replay agreement
- ``StagedPlan`` / ``staged_decision``   Rung 1 Stage A / Stage B with an
                                          early-look correction
- ``staged_operating_characteristics``    exact pass / fail / ambiguous
                                          probabilities and expected seeds

Thresholds (nominal band, failure anchor, minimum agreement) are always passed
in by the caller: they are calibrated per policy and task (3.2.9), never
hard-coded here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist
from typing import Mapping, Optional

from router.schemas import Outcome


# --------------------------------------------------------------------------- #
# proportion with a confidence interval
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Proportion:
    k: int
    n: int
    lo: float
    hi: float
    confidence: float

    @property
    def rate(self) -> float:
        return self.k / self.n if self.n else float("nan")

    def contains(self, p: float) -> bool:
        return self.lo <= p <= self.hi


def _z(confidence: float) -> float:
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Proportion:
    """Two-sided Wilson score interval. Behaves well at small n and at 0 or n
    successes, which is the regime of 10-50 seed replays."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return Proportion(0, 0, 0.0, 1.0, confidence)
    z = _z(confidence)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    # Exact endpoints at the extremes (avoid -1e-17 style float noise).
    if k == 0:
        lo = 0.0
    if k == n:
        hi = 1.0
    return Proportion(k, n, lo, hi, confidence)


# --------------------------------------------------------------------------- #
# three-valued outcome
# --------------------------------------------------------------------------- #

def classify_outcome(prop: Proportion, failure_rate: float,
                     nominal_band: tuple[float, float]) -> Outcome:
    """Classify a replay against the two anchors (3.2.9).

    restored:     interval excludes the failure rate, and the point rate is at
                  or above the band's low edge
    partial:      interval excludes the failure rate and sits entirely below
                  the band (hi < band_lo)
    no_effect:    interval includes the failure rate (or lies entirely below it)
    inconclusive: the interval straddles a boundary; extend the seeds. Either
                  it includes the failure rate AND reaches the band, or it
                  excludes the failure rate but the point rate is below the
                  band's low edge while the interval still reaches it.

    Both the failure rate and the band's low edge count as "boundaries" for
    the straddle rule. The literal reading (restored = interval overlaps the
    band) passed about half of all true-45% patches at 10 seeds against a 0.8
    nominal; see DECISIONS.md. A replay above the band counts as restored.
    """
    band_lo, band_hi = nominal_band
    if not (0.0 <= band_lo <= band_hi <= 1.0):
        raise ValueError(f"bad nominal band {nominal_band}")
    if not (0.0 <= failure_rate < band_lo):
        raise ValueError(f"failure rate {failure_rate} must be below the nominal band {nominal_band}")

    includes_failure = prop.contains(failure_rate)
    reaches_band = prop.hi >= band_lo

    if includes_failure and reaches_band:
        return Outcome.INCONCLUSIVE
    if includes_failure or prop.hi < failure_rate:
        return Outcome.NO_EFFECT
    if not reaches_band:
        return Outcome.PARTIAL
    if prop.rate >= band_lo:
        return Outcome.RESTORED
    return Outcome.INCONCLUSIVE


def replay_outcome(k: int, n: int, failure_rate: float, nominal_band: tuple[float, float],
                   confidence: float = 0.95) -> tuple[Outcome, Proportion]:
    prop = wilson_interval(k, n, confidence)
    return classify_outcome(prop, failure_rate, nominal_band), prop


# --------------------------------------------------------------------------- #
# exact two-sided tests
# --------------------------------------------------------------------------- #

def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar test: two-sided binomial test on the discordant pairs.

    b = seeds that fail before and succeed after, c = the reverse.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be >= 0")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact test for two independent proportions.

    Sums the probabilities of all tables with the same margins that are no
    more likely than the observed one.
    """
    for k, n in ((k1, n1), (k2, n2)):
        if n < 0 or k < 0 or k > n:
            raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    total = n1 + n2
    succ = k1 + k2
    if total == 0:
        return 1.0
    denom = math.comb(total, succ)

    def pmf(x: int) -> float:
        return math.comb(n1, x) * math.comb(n2, succ - x) / denom

    p_obs = pmf(k1)
    lo, hi = max(0, succ - n2), min(n1, succ)
    p = sum(q for q in (pmf(x) for x in range(lo, hi + 1)) if q <= p_obs * (1 + 1e-9))
    return min(1.0, p)


# --------------------------------------------------------------------------- #
# paired vs unpaired comparison
# --------------------------------------------------------------------------- #

def agreement_rate(first: Mapping[int, bool], second: Mapping[int, bool]) -> float:
    """Fraction of shared seeds whose outcome matched across two runs of the
    same configuration (the replay-agreement measurement, 3.2.2)."""
    shared = set(first) & set(second)
    if not shared:
        raise ValueError("no shared seeds between the two runs")
    return sum(1 for s in shared if bool(first[s]) == bool(second[s])) / len(shared)


@dataclass(frozen=True)
class Comparison:
    method: str                     # "paired" or "unpaired"
    p_value: float
    rate_before: float
    rate_after: float
    discordant: Optional[tuple[int, int]]   # (fail->success, success->fail), paired only
    alpha: float
    reason: str

    @property
    def diff(self) -> float:
        return self.rate_after - self.rate_before

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def clearly_better(self) -> bool:
        return self.significant and self.diff > 0

    @property
    def clearly_worse(self) -> bool:
        return self.significant and self.diff < 0


def compare(before: Mapping[int, bool], after: Mapping[int, bool], *,
            agreement: float, min_agreement: float, alpha: float = 0.05) -> Comparison:
    """Compare two sets of episodes keyed by seed.

    Uses the paired exact McNemar test when replay agreement is at least
    ``min_agreement`` *and* both runs used the same seeds; otherwise falls back
    to the unpaired Fisher exact test (3.2.9). ``min_agreement`` is a
    calibrated value with no default on purpose.
    """
    if not before or not after:
        raise ValueError("both runs need at least one episode")
    if not (0.0 <= agreement <= 1.0) or not (0.0 <= min_agreement <= 1.0):
        raise ValueError("agreement values must be in [0, 1]")

    rb = sum(map(bool, before.values())) / len(before)
    ra = sum(map(bool, after.values())) / len(after)
    same_seeds = set(before) == set(after)

    if agreement >= min_agreement and same_seeds:
        b = sum(1 for s in before if not before[s] and after[s])
        c = sum(1 for s in before if before[s] and not after[s])
        return Comparison("paired", mcnemar_exact(b, c), rb, ra, (b, c), alpha,
                          f"agreement {agreement:.2f} >= {min_agreement:.2f} and seeds match")

    reason = (f"agreement {agreement:.2f} < {min_agreement:.2f}" if agreement < min_agreement
              else "seed sets differ")
    kb, ka = sum(map(bool, before.values())), sum(map(bool, after.values()))
    return Comparison("unpaired", fisher_exact(ka, len(after), kb, len(before)), rb, ra,
                      None, alpha, reason)


def clearly_better(before: Mapping[int, bool], after: Mapping[int, bool], *,
                   agreement: float, min_agreement: float, alpha: float = 0.05) -> bool:
    """The single shared definition of "clearly better" (3.2.9)."""
    return compare(before, after, agreement=agreement, min_agreement=min_agreement,
                   alpha=alpha).clearly_better


# --------------------------------------------------------------------------- #
# staged-seed decision (Rung 1)
# --------------------------------------------------------------------------- #

class StageDecision(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    CONTINUE = "continue"     # ambiguous after Stage A -> run Stage B
    AMBIGUOUS = "ambiguous"   # ambiguous after Stage B -> Rung 2


@dataclass(frozen=True)
class StagedPlan:
    """Two looks at the data. ``n_b`` is the cumulative seed count after Stage
    B (Stage A seeds are reused, not thrown away).

    Early-look correction: Bonferroni split of the total error budget across
    the two looks. Stage A intervals use confidence 1 - alpha_a and Stage B
    intervals 1 - alpha_b, with alpha_a + alpha_b = alpha_total. A smaller
    ``stage_a_fraction`` makes Stage A stricter (fewer early stops).
    """
    n_a: int
    n_b: int
    alpha_total: float = 0.05
    stage_a_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not (0 < self.n_a < self.n_b):
            raise ValueError("need 0 < n_a < n_b")
        if not (0.0 < self.alpha_total < 1.0):
            raise ValueError("alpha_total must be in (0, 1)")
        if not (0.0 < self.stage_a_fraction < 1.0):
            raise ValueError("stage_a_fraction must be in (0, 1)")

    @property
    def alpha_a(self) -> float:
        return self.alpha_total * self.stage_a_fraction

    @property
    def alpha_b(self) -> float:
        return self.alpha_total * (1.0 - self.stage_a_fraction)

    def confidence(self, stage: str) -> float:
        if stage == "A":
            return 1.0 - self.alpha_a
        if stage == "B":
            return 1.0 - self.alpha_b
        raise ValueError(f"stage must be 'A' or 'B', got {stage!r}")


@dataclass(frozen=True)
class StagedResult:
    decision: StageDecision
    outcome: Outcome
    proportion: Proportion
    stage: str


def staged_decision(k: int, n: int, stage: str, plan: StagedPlan, *,
                    failure_rate: float, nominal_band: tuple[float, float]) -> StagedResult:
    """Rung 1 decision after one stage (Section 4).

    restored -> PASS, no_effect -> FAIL; partial or inconclusive -> CONTINUE
    after Stage A, AMBIGUOUS (Rung 2) after Stage B. A partial fix is not a
    pass: it goes on to more seeds and then to triage.
    """
    expected_n = plan.n_a if stage == "A" else plan.n_b
    if n != expected_n:
        raise ValueError(f"stage {stage} expects n={expected_n} cumulative seeds, got n={n}")
    prop = wilson_interval(k, n, plan.confidence(stage))
    outcome = classify_outcome(prop, failure_rate, nominal_band)
    if outcome is Outcome.RESTORED:
        decision = StageDecision.PASS
    elif outcome is Outcome.NO_EFFECT:
        decision = StageDecision.FAIL
    else:
        decision = StageDecision.CONTINUE if stage == "A" else StageDecision.AMBIGUOUS
    return StagedResult(decision, outcome, prop, stage)


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


@dataclass(frozen=True)
class OperatingCharacteristics:
    p_pass: float
    p_fail: float
    p_ambiguous: float
    p_stop_at_a: float
    expected_seeds: float


def staged_operating_characteristics(p_true: float, plan: StagedPlan, *, failure_rate: float,
                                     nominal_band: tuple[float, float]) -> OperatingCharacteristics:
    """Exact probabilities of each staged outcome if the true success rate is
    ``p_true`` and episodes are independent. Used to report the wrong-pass and
    wrong-fail rates of early stopping (Section 7) and to calibrate the plan."""
    p_pass = p_fail = p_amb = p_stop_a = 0.0
    extra = plan.n_b - plan.n_a
    for ka in range(plan.n_a + 1):
        wa = _binom_pmf(ka, plan.n_a, p_true)
        ra = staged_decision(ka, plan.n_a, "A", plan, failure_rate=failure_rate, nominal_band=nominal_band)
        if ra.decision is StageDecision.PASS:
            p_pass += wa
            p_stop_a += wa
            continue
        if ra.decision is StageDecision.FAIL:
            p_fail += wa
            p_stop_a += wa
            continue
        for ke in range(extra + 1):
            w = wa * _binom_pmf(ke, extra, p_true)
            rb = staged_decision(ka + ke, plan.n_b, "B", plan, failure_rate=failure_rate,
                                 nominal_band=nominal_band)
            if rb.decision is StageDecision.PASS:
                p_pass += w
            elif rb.decision is StageDecision.FAIL:
                p_fail += w
            else:
                p_amb += w
    expected = plan.n_a + (1.0 - p_stop_a) * extra
    return OperatingCharacteristics(p_pass, p_fail, p_amb, p_stop_a, expected)
