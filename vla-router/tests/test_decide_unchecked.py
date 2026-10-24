"""``decide`` with unchecked signals (Fix F1; ISSUES #1, Tier 5).

A row may only evaluate FALSE if every check that could make it TRUE ran; a
skip turns that FALSE into UNKNOWN (row 1: TRUE into UNKNOWN). The existing
"UNKNOWN higher row -> row 10" rule does the rest. ``tests/test_decide.py`` is
unchanged and runs with the default empty set; here the empty set is also
compared against a frozen copy of the pre-F1 ``decide``.
"""

from __future__ import annotations

import random

import pytest

from router import vocab as V
from router.decide import DecideConfig, decide, row1_state
from tests.fixtures import decide_pre_f1
from tests.test_decide import BASE_C, BASE_G, RM, N, P, R, _random_case, ev, table

ALL_SIGNALS = sorted(frozenset().union(*V.SIGNALS_BY_SOURCE.values()))


def v(evidence, base, over=None, unchecked=()):
    return decide(evidence, table(base, over), unchecked=unchecked)


# --------------------------------------------------------------------------- #
# the empty set is the old behaviour; random sets only ever push to row 10
# --------------------------------------------------------------------------- #

def test_empty_unchecked_set_reproduces_the_pre_f1_verdict_exactly():
    rng = random.Random(11)
    for _ in range(4000):
        evs, reps = _random_case(rng)
        assert decide(evs, reps) == decide(evs, reps, unchecked=()) == decide_pre_f1.decide(evs, reps)


def test_random_unchecked_sets_leave_the_row_unchanged_or_make_it_10():
    rng = random.Random(12)
    changed, rows = 0, set()
    for _ in range(4000):
        evs, reps = _random_case(rng)
        base = decide(evs, reps)
        unchecked = {s for s in ALL_SIGNALS if rng.random() < 0.3}
        got = decide(evs, reps, unchecked=unchecked)
        assert got.row in (base.row, 10), (base.row, got.row, sorted(unchecked), got.notes)
        if got.row != base.row:
            changed += 1
            rows.add(base.row)
    # the property is actually exercised (a TRUE row 2 never flips: row 1 above it was FALSE)
    assert changed > 100 and rows >= {1, 3, 4, 5, 6, 7}


# --------------------------------------------------------------------------- #
# each skipped source turns its row's FALSE into UNKNOWN
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("source,evidence,base,over,row", [
    # row 2: no sanity evidence, reference fine -> FALSE; would be row 6
    ("sanity", [], BASE_C, {RM: R}, 2),
    # row 4 (Branch C): no located evidence -> FALSE; would be row 6
    ("contract_obs", [], BASE_C, {RM: R}, 4),
    ("contract_action", [], BASE_C, {RM: R}, 4),
    ("contract_cfg", [], BASE_C, {RM: R}, 4),
    ("staleness", [], BASE_C, {RM: R}, 4),                 # staleness_constant
    ("behavior", [], BASE_C, {RM: R}, 4),                  # implied_target_offset
])
def test_skipped_source_makes_its_row_undecided(source, evidence, base, over, row):
    assert v(evidence, base, over).row == 6
    got = v(evidence, base, over, V.SIGNALS_BY_SOURCE[source])
    assert got.row == 10 and f"row {row} undecided: not checked" in got.notes
    assert "row 6 would match but a higher-precedence row is undecided" in got.notes


def test_row1_inverse_rule_unchecked_variable_staleness_turns_true_into_unknown():
    t = table({V.R_STAGE0: R})
    assert decide([], t).row == 1 and row1_state([], t) == "true"
    got = decide([], t, unchecked={V.SIG_STALENESS_VARIABLE})
    assert got.row == 10 and "row 1 undecided: stage0 passes but not checked: staleness_variable" in got.notes
    assert row1_state([], t, unchecked={V.SIG_STALENESS_VARIABLE}) == "unknown"
    assert row1_state([], table({V.R_STAGE0: N})) == "false"
    # other unchecked signals do not touch row 1
    assert row1_state([], t, unchecked={V.SIG_CONTRACT_VIOLATION}) == "true"


def test_row3_flips_only_when_staleness_could_make_it_true():
    sv = {V.SIG_STALENESS_VARIABLE}
    # Stage 0 partly restores in clean mode: variable staleness would make it infra
    got = v([], BASE_C, {V.R_STAGE0: P, RM: R}, sv)
    assert v([], BASE_C, {V.R_STAGE0: P, RM: R}).row == 6
    assert got.row == 10 and "row 3 undecided: not checked: staleness_variable" in got.notes
    # Stage 0 fails in clean mode: row 3 is out whatever staleness says
    assert v([], BASE_C, {RM: R}, sv).row == 6


def test_row5_flips_when_render_settings_were_not_compared():
    got = v([], BASE_C, {RM: R}, {V.SIG_RENDER_SETTINGS_DIFFER})
    assert got.row == 10 and "row 5 undecided: not checked: render_settings_differ" in got.notes


def test_rows_8_and_9_flip_only_when_nothing_restores():
    behavior = V.SIGNALS_BY_SOURCE["behavior"] - {V.SIG_IMPLIED_TARGET_OFFSET}   # keep row 4 decided
    # nothing restores, implied target wrong -> row 9; row 8 unchecked above it -> 10
    wrong = [ev(V.SIG_IMPLIED_TARGET_WRONG)]
    assert v(wrong, BASE_C).row == 9
    got = v(wrong, BASE_C, None, {V.SIG_IMPLIED_TARGET_OK})
    assert got.row == 10 and "row 8 undecided: not checked: implied_target_ok" in got.notes
    # no behavior evidence at all: "no row matched" becomes "rows 8, 9 undecided"
    got = v([], BASE_C, None, behavior)
    assert got.row == 10 and "row 8 undecided" in got.notes and "row 9 undecided" in got.notes
    # a factor restored: rows 8-9 are FALSE by the replays and stay FALSE
    assert v([], BASE_C, {RM: R}, behavior).row == 6


def test_branch_g_elimination_is_true_and_stays_true():
    for unchecked in (V.SIGNALS_BY_SOURCE["contract_obs"], V.SIGNALS_BY_SOURCE["behavior"]):
        got = v([], BASE_G, None, unchecked)
        assert got.row == 4 and got.by_elimination


# --------------------------------------------------------------------------- #
# a TRUE row is never blocked by a skip on a lower row
# --------------------------------------------------------------------------- #

def _lower_only(row: int) -> set[str]:
    """Every signal that only rows below ``row`` depend on."""
    above = set().union(*(V.ROW_SIGNALS.get(r, frozenset()) for r in range(1, row + 1)))
    return set(ALL_SIGNALS) - above


@pytest.mark.parametrize("evidence,base,over,row", [
    ([], BASE_C, {V.R_C1: N}, 2),                                            # reference fails on the shifted scene
    ([ev(V.SIG_STALENESS_VARIABLE)], BASE_C, {V.R_STAGE0: R}, 3),
    ([ev(V.SIG_CONTRACT_VIOLATION, location="norm_stats")], BASE_C, None, 4),
    ([ev(V.SIG_RENDER_SETTINGS_DIFFER)], BASE_C, {V.R_C_RENDER: R}, 5),
    ([], BASE_C, {RM: R}, 6),
    ([ev(V.SIG_IMPLIED_TARGET_OK), ev(V.failure_phase("grasp"))], BASE_C, None, 8),
])
def test_true_row_is_not_blocked_by_skips_on_lower_rows(evidence, base, over, row):
    lower = _lower_only(row)
    assert v(evidence, base, over).row == row
    assert v(evidence, base, over, lower).row == row, sorted(lower)


def test_relaxing_a_row_is_a_config_change():
    t = table(BASE_C, {RM: R})
    unchecked = V.SIGNALS_BY_SOURCE["behavior"]
    assert decide([], t, unchecked=unchecked).row == 10
    relaxed = DecideConfig(row_signals={r: s - unchecked for r, s in V.ROW_SIGNALS.items()})
    assert decide([], t, relaxed, unchecked).row == 6


def test_unchecked_must_be_signal_names():
    with pytest.raises(TypeError):
        decide([], table(BASE_C), unchecked=[1])
