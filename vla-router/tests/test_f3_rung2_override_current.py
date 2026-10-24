"""ISSUES #4, Rung 2 part (Fix F3): the router re-run verdict wins over Rung 2's label.

    rows 4, 10   Rung 2 "code" -> repair again; "unclear" -> UNRESOLVED; "data_gap" is rejected by the
                 gate (becomes unclear) -> UNRESOLVED
    rows 6, 7    the data ladder, whatever Rung 2 says; a label that disagrees is marked overridden
    others       the router's destination is followed; the label is logged only
The model's label is always in ``LoopResult.rung2`` (``model_label``).

(The file name is historical: its first two tests pinned the OLD behaviour, where Rung 2 "code" sent a
row 6 re-run back to hagent and "unclear" dropped it. They now assert the new rule.)
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import ladder.loop as loop_mod
from ladder.loop import LoopStatus
from ladder.rung2 import Rung2Label
from router.decide import DESTINATION_BY_ROW, LABEL_BY_ROW
from tests.ladder_cases import PATCHED, PATCHED_INSTRUCTION, ScriptedRepairer, combined_case, condition_specific_case, loop_config, make_loop
from tests.test_ladder_loop import reply, rung2_fn

LABELS = ["code", "unclear", "data_gap"]


def test_rung2_code_no_longer_overrides_a_row6_vision_rerun():
    # was: rung2:code -> back to hagent (rep.calls == 2). Now the row 6 re-run goes to the data ladder.
    case = condition_specific_case(0.4)              # re-run on the patch: row 6, material(mug) is a vision factor
    fn, model = rung2_fn(reply("code"))
    rep = ScriptedRepairer([PATCHED_INSTRUCTION])
    loop, ladder = make_loop(case, rep, rung2=fn, cfg=loop_config(max_repairs=2))
    res = loop.run(case.incident())
    assert [p.row for p in res.passes] == [4, 6] and rep.calls == 1
    assert res.status is LoopStatus.DATA_HANDOFF and ladder.handoffs[0].factor == "material(mug)"
    assert not ladder.handoffs[0].via_rung2 and len(model.calls) == 1
    assert res.rung2[0].model_label is Rung2Label.CODE and "overridden by router" in res.rung2[0].overridden


def test_rung2_unclear_no_longer_drops_a_row6_vision_rerun():
    # was: UNRESOLVED, nothing handed over. Now the data ladder gets it.
    case = condition_specific_case(0.4)
    fn, _ = rung2_fn(reply("unclear"))
    loop, ladder = make_loop(case, ScriptedRepairer([PATCHED_INSTRUCTION]), rung2=fn)
    res = loop.run(case.incident())
    assert res.passes[1].row == 6 and res.status is LoopStatus.DATA_HANDOFF and len(ladder.handoffs) == 1
    assert res.rung2[0].model_label is Rung2Label.UNCLEAR and "overridden by router" in res.rung2[0].overridden


def test_rung2_data_gap_agreeing_with_a_row6_rerun_is_not_marked_overridden():
    case = condition_specific_case(0.4)
    fn, _ = rung2_fn(reply("data_gap"))
    loop, ladder = make_loop(case, ScriptedRepairer([PATCHED_INSTRUCTION]), rung2=fn)
    res = loop.run(case.incident())
    assert res.status is LoopStatus.DATA_HANDOFF and ladder.handoffs[0].via_rung2 and res.rung2[0].overridden == ""


def run_forced(row: int, label: str):
    """An ambiguous patch whose router re-run is forced to ``row``; Rung 2 answers ``label``."""
    case = combined_case(patched_nominal=0.4)
    fn, model = rung2_fn(reply(label))
    rep = ScriptedRepairer([PATCHED])
    loop, ladder = make_loop(case, rep, rung2=fn, cfg=loop_config(max_repairs=2))
    real = loop_mod.route

    def forced(inc, backend, cfg, inputs):
        rr = real(inc, backend, cfg, inputs)
        if inc.code_ref == PATCHED:
            v = replace(rr.verdict, label=LABEL_BY_ROW[row], destination=DESTINATION_BY_ROW[row], row=row,
                        factor="lighting" if row in (6, 7) else None, replays=())
            rr = replace(rr, verdict=v)
        return rr
    loop_mod.route, saved = forced, real
    try:
        res = loop.run(case.incident())
    finally:
        loop_mod.route = saved
    return res, rep, ladder, model


@pytest.mark.parametrize("row", [4, 10])
@pytest.mark.parametrize("label", LABELS)
def test_rows_4_and_10_use_the_rung2_label(row, label):
    res, rep, ladder, model = run_forced(row, label)
    assert res.rung2[0].model_label is Rung2Label(label) and len(model.calls) == 1 and not ladder.handoffs
    if label == "code":
        assert rep.calls == 2 and res.passes[1].action == "rung2:code"
        assert any("Rung 2" in f for f in rep.requests[1].feedback)
    else:                                            # unclear, or data_gap rejected by the gate
        assert rep.calls == 1 and res.status is LoopStatus.UNRESOLVED and "Rung 2 unclear" in res.reason
        assert res.rung2[0].label is Rung2Label.UNCLEAR


@pytest.mark.parametrize("row", [6, 7])
@pytest.mark.parametrize("label", LABELS)
def test_rows_6_and_7_go_to_the_data_ladder_whatever_rung2_says(row, label):
    res, rep, ladder, model = run_forced(row, label)
    assert res.status is LoopStatus.DATA_HANDOFF and rep.calls == 1 and ladder.handoffs[0].factor == "lighting"
    assert res.rung2[0].model_label is Rung2Label(label)                    # the model's label is logged
    assert "overridden by router" in res.rung2[0].overridden                # (forced replays: data_gap fails the gate too)
    assert res.to_dict()["rung2"][0]["overridden"] == res.rung2[0].overridden


@pytest.mark.parametrize("row", [2, 3, 5, 8, 9])
@pytest.mark.parametrize("label", LABELS)
def test_other_rows_follow_the_router_and_log_the_label_only(row, label):
    res, rep, ladder, model = run_forced(row, label)
    assert res.rung2[0].model_label is Rung2Label(label) and "label logged only" in res.rung2[0].overridden
    assert not ladder.handoffs
    if row == 5:                                     # config_fix: the router sends it to a repair, as without Rung 2
        assert rep.calls == 2 and rep.requests[1].kind.value == "config"
    else:                                            # report only
        assert rep.calls == 1 and res.status is LoopStatus.REPORTED and res.passes[1].action == "report"
