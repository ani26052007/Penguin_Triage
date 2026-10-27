"""Phase 5: replay agreement (ISSUES #15) and the clean vs incident mode gap, on MockBackend with known
rates and known agreement."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ladder.backend import MockBackend, MockRule, RolloutBackend
from ladder.stats import agreement_rate, fisher_exact, wilson_interval
from router.prereq.rollout_cache import CachedBackend, RolloutCache
from router.schemas import ActorKind, ActorRef, CodeRef, EpisodeResult, Factor, Mode, SceneSpec

from vla_isaac.experiments.mode_gap import PAIRING, measure_mode_gap
from vla_isaac.experiments.replay_agreement import AgreementError, compare_runs, measure_agreement

CODE = CodeRef("c0")
POLICY = ActorRef(ActorKind.POLICY, "ckpt")
NOMINAL = SceneSpec("gr1_fridge_ranch")
DARK = SceneSpec("gr1_fridge_ranch", (Factor("lighting.intensity", "100.0"),))


class Scripted(RolloutBackend):
    """Exact outcomes per call: ``runs[i]`` is the i-th call's {seed: success}."""

    def __init__(self, *runs, logs=None):
        self.runs, self.calls, self.logs = list(runs), 0, logs or {}

    def run(self, scene, code_ref, actor, seeds, mode):
        out = self.runs[self.calls % len(self.runs)]
        self.calls += 1
        return [EpisodeResult(s, out[s], dict(self.logs)) for s in seeds]


# ---- replay agreement ---------------------------------------------------------------------------------

def test_exact_counts_on_a_scripted_pair_of_runs():
    first = {s: s < 6 for s in range(10)}                    # S S S S S S F F F F
    second = {s: s in (0, 1, 2, 3, 6, 7) for s in range(10)}  # S S S S F F S S F F
    rep = measure_agreement(Scripted(first, second), [NOMINAL], CODE, POLICY, range(10))
    c = rep.cells[0]
    assert (c.both_success, c.both_failure, c.first_fail_second_success, c.first_success_second_fail) == (4, 2, 2, 2)
    assert c.discordant == 4 and c.agreement["k"] == 6 and c.agreement["n"] == 10
    assert c.agreement["rate"] == agreement_rate(first, second) == 0.6
    p = wilson_interval(6, 10, 0.95)                                    # the interval is ladder.stats'
    assert (c.agreement["lower"], c.agreement["upper"]) == (p.lo, p.hi)
    assert c.success["k"] == 12 and c.success["n"] == 20 and (c.first_rate, c.second_rate) == (0.6, 0.6)
    assert c.informative and c.note is None


def test_a_deterministic_mock_agrees_perfectly():
    be = MockBackend([MockRule(0.6)], 0.0, salt="agree")
    rep = measure_agreement(be, [NOMINAL, DARK], CODE, POLICY, range(40))
    assert all(c.agreement["rate"] == 1.0 and c.discordant == 0 for c in rep.cells)
    assert rep.pooled["k"] == rep.pooled["n"] == 80


def test_a_known_replay_noise_gives_a_known_agreement():
    """MockBackend flips a repeated seed with probability ``replay_noise``: expected agreement 0.75."""
    be = MockBackend([MockRule(0.5)], 0.0, replay_noise=0.25, salt="agree")
    rep = measure_agreement(be, [NOMINAL], CODE, POLICY, range(400))
    c = rep.cells[0]
    assert c.agreement["lower"] <= 0.75 <= c.agreement["upper"]
    assert c.both_success + c.both_failure + c.discordant == 400 and c.informative
    first = MockBackend([MockRule(0.5)], 0.0, salt="agree").run(NOMINAL, CODE, POLICY, range(400), Mode.CLEAN)
    assert c.first_rate == sum(e.success for e in first) / 400            # the first run is the plain draw


@pytest.mark.parametrize("rate, only, other", [(0.0, "failures", "successes"), (1.0, "successes", "failures")])
def test_a_policy_that_always_fails_or_succeeds_is_not_shown_as_reproducible_without_a_note(rate, only, other):
    rep = measure_agreement(MockBackend([MockRule(rate)], 0.0, salt="agree"), [NOMINAL], CODE, POLICY, range(20))
    c = rep.cells[0]
    assert c.agreement["rate"] == 1.0
    assert not c.informative and not rep.informative
    assert f"only from {only}" in c.note and f"overall success rate {rate:.2f}" in c.note and other in c.note
    d = rep.to_dict()
    assert d["informative"] is False and d["cells"][0]["note"] == c.note and d["cells"][0]["success"]["rate"] == rate
    head, row = rep.csv().splitlines()
    assert row.split(",")[head.split(",").index("informative")] == "false" and "only from" in row


def test_no_agreement_on_either_outcome_is_noted():
    rep = measure_agreement(Scripted({0: True, 1: False}, {0: False, 1: True}), [NOMINAL], CODE, POLICY, [0, 1])
    assert rep.cells[0].agreement["k"] == 0 and rep.cells[0].note == "no seed agreed on either outcome"


def test_the_cache_is_refused_because_it_hides_non_determinism():
    cached = CachedBackend(MockBackend([], 0.5, replay_noise=0.5), RolloutCache())
    with pytest.raises(AgreementError, match="CachedBackend"):
        measure_agreement(cached, [NOMINAL], CODE, POLICY, range(5))


def test_unpaired_results_are_refused():
    unpaired = Scripted({s: True for s in range(3)}, logs={"paired": False})     # incident mode or Mimic trials
    with pytest.raises(AgreementError, match="not paired"):
        measure_agreement(unpaired, [NOMINAL], CODE, POLICY, range(3))
    a = [EpisodeResult(0, True), EpisodeResult(1, False)]
    with pytest.raises(AgreementError, match="same distinct seeds"):
        compare_runs(NOMINAL, a, list(reversed(a)))


def test_the_report_is_written_as_json_and_csv(tmp_path):
    rep = measure_agreement(MockBackend([MockRule(0.5)], 0.0, replay_noise=0.1), [NOMINAL, DARK], CODE, POLICY,
                            range(30))
    js, cs = rep.write(tmp_path)
    data = json.loads(js.read_text())
    assert len(data["cells"]) == 2 and data["pooled_agreement"]["n"] == 60
    assert {"both_success", "both_failure", "discordant", "success", "note"} <= set(data["cells"][0])
    assert cs.read_text() == rep.csv() and len(cs.read_text().splitlines()) == 3


# ---- mode gap -------------------------------------------------------------------------------------------

def test_a_known_mode_gap_is_found_with_fisher_and_wilson_from_ladder_stats():
    be = MockBackend([MockRule(0.2, mode=Mode.INCIDENT), MockRule(1.0, mode=Mode.CLEAN)], 0.0, salt="gap")
    gap = measure_mode_gap(be, NOMINAL, CODE, POLICY, range(50))
    kc, ki = gap.clean["k"], gap.incident["k"]
    assert kc == 50 and ki < 25 and gap.differ and gap.gap > 0.5
    assert gap.p_value == fisher_exact(kc, 50, ki, 50)
    for side, k in (("clean", kc), ("incident", ki)):
        p = wilson_interval(k, 50, 0.95)
        assert (getattr(gap, side)["lower"], getattr(gap, side)["upper"]) == (p.lo, p.hi)


def test_no_gap_when_both_modes_match():
    gap = measure_mode_gap(MockBackend([MockRule(1.0)], 0.0), NOMINAL, CODE, POLICY, range(30))
    assert gap.gap == 0.0 and gap.p_value == 1.0 and not gap.differ


def test_the_mode_gap_is_always_unpaired():
    class IncidentUnpaired(MockBackend):
        def run(self, scene, code_ref, actor, seeds, mode):
            out = super().run(scene, code_ref, actor, seeds, mode)
            return [replace(e, logs={**e.logs, "paired": Mode(mode) is Mode.CLEAN}) for e in out]

    gap = measure_mode_gap(IncidentUnpaired([MockRule(0.7)], 0.0), NOMINAL, CODE, POLICY, range(20),
                           incident_seeds=range(40))
    d = gap.to_dict()
    assert d["pairing"] == PAIRING == "unpaired" and "D10" in d["pairing_reason"]
    assert d["incident_paired_in_logs"] is False and gap.incident["n"] == 40 and gap.clean["n"] == 20
    assert "fisher_exact" in d["test"]
    plain = measure_mode_gap(MockBackend([MockRule(0.7)], 0.0), NOMINAL, CODE, POLICY, range(20)).to_dict()
    assert plain["pairing"] == "unpaired" and plain["incident_paired_in_logs"] is None   # unpaired regardless


def test_the_mode_gap_is_written(tmp_path):
    gap = measure_mode_gap(MockBackend([MockRule(0.7)], 0.0), NOMINAL, CODE, POLICY, range(10))
    assert json.loads(gap.write(tmp_path).read_text())["scene"] == NOMINAL.key()
    with pytest.raises(ValueError):
        measure_mode_gap(MockBackend([], 0.5), NOMINAL, CODE, POLICY, [])


def test_on_isaac_backend_the_second_run_is_a_real_rerun(tmp_path):
    from tests.helpers import NOMINAL as PILOT_NOMINAL, POLICY as PILOT_POLICY, make_backend

    rig = make_backend(tmp_path)
    rep = measure_agreement(rig.backend, [PILOT_NOMINAL], rig.code(), PILOT_POLICY, range(3))
    assert len(rig.runner.calls) == 2 and rep.cells[0].agreement["n"] == 3     # Arena ran twice, nothing reused
    assert all(inv.spec.num_envs == 1 for inv in rig.runner.calls)             # clean mode
