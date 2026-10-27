"""Router change 2: the optional ``post_place`` phase (a composite task's subtask after the place,
such as closing the fridge door in the sim side's pilot).

The required check: a pick-only ``BehaviorTrace`` (no ``post_place`` record) gets the same verdict as
before the change. ``verdict_grid()`` routes a grid of pick-only behaviour faults, at the decide level
(C_behavior evidence and its partial skips on a Branch C table) and through ``route()``; its output
before the change is frozen in ``tests/fixtures/pick_only_verdicts_before_change2.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from router import vocab as V
from router.behavior import analyze_behavior
from router.behavior_synthetic import BehaviorFaults, make_behavior_nominal, make_behavior_traces
from router.decide import decide
from router.prereq import behavior_trace as BT
from router.prereq.behavior_trace import BehaviorTrace, SubtaskProgress
from router.router import route
from router.schemas import Mode, VerdictLabel
from tests.router_cases import default_config, router_cases
from tests.test_behavior import SCENE, _table

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "pick_only_verdicts_before_change2.json"
PICK = ("reach", "grasp", "lift", "place")
TARGETS = {"ok": {}, "offset": {"offset": (0.0, 0.06, 0.0)}, "scatter": {"scatter": 0.06},
           "wrong_object": {"wrong_object": 0.6}, "no_grasp": {"no_grasp": True}}


def _verdict(v) -> dict:
    return {"label": v.label.value, "row": v.row, "location": v.location, "by_elimination": v.by_elimination}


def verdict_grid() -> dict:
    nominal = make_behavior_nominal()
    out: dict = {}
    for tname, target in TARGETS.items():
        for phase in (None,) + PICK:
            traces = make_behavior_traces(8, Mode.CLEAN, BehaviorFaults(fail_phase=phase, **target))
            assert all("post_place" not in {s.name for s in t.subtasks} for t in traces)
            a = analyze_behavior(traces, nominal)
            unchecked = set().union(*(p.signals for p in a.partial)) if a.partial else set()
            v = decide(list(a.evidence), _table(SCENE), unchecked=unchecked)
            out[f"decide/{tname}/{phase}"] = {**_verdict(v), "evidence": sorted(e.signal for e in a.evidence)}
    cfg = default_config()
    for case in router_cases():
        out[f"route/{case.name}"] = _verdict(route(case.incident(), case.quota_backend(), cfg, case.inputs()).verdict)
    gap = next(c for c in router_cases() if c.name == "action_gap")
    for phase in (None,) + PICK:
        case = type(gap)(**{**gap.__dict__, "behavior_default": BehaviorFaults(fail_phase=phase)})
        out[f"route/action_gap/{phase}"] = _verdict(route(case.incident(), case.quota_backend(), cfg,
                                                          case.inputs()).verdict)
    return out


def test_pick_only_traces_get_the_same_verdicts_as_before_the_change():
    assert verdict_grid() == json.loads(GOLDEN.read_text())


def test_the_two_phase_tuples_agree_and_post_place_is_the_only_addition():
    assert BT.PHASES == V.PHASES == ("reach", "grasp", "lift", "place", "post_place")
    assert BT.PICK_PHASES == PICK and BT.OPTIONAL_PHASES == ("post_place",)
    assert V.PHASES_ACTION_GAP == ("grasp", "lift", "place", "post_place")
    assert V.failure_phase("post_place") in V.ALL_SIGNALS
    assert V.failure_phase("post_place") in V.ROW_SIGNALS[8]


def test_post_place_is_optional_a_pick_only_record_never_names_it():
    done = [SubtaskProgress(p, True) for p in PICK]
    assert BehaviorTrace(1, Mode.CLEAN, False, subtasks=done).failure_phase() is None
    assert BehaviorTrace(1, Mode.CLEAN, False, subtasks=done[:3] + [SubtaskProgress("place", False)]
                         ).failure_phase() == "place"


def test_a_door_failure_after_a_good_place_is_the_post_place_phase():
    subtasks = [SubtaskProgress(p, True) for p in PICK] + [SubtaskProgress("post_place", False)]
    assert BehaviorTrace(1, Mode.CLEAN, False, subtasks=subtasks).failure_phase() == "post_place"
    # an earlier failed phase still wins: the episode failed there first
    early = [SubtaskProgress("reach", True), SubtaskProgress("grasp", False)] + subtasks[2:]
    assert BehaviorTrace(1, Mode.CLEAN, False, subtasks=early).failure_phase() == "grasp"


def test_post_place_failures_with_a_right_target_are_an_action_gap_row_8():
    nominal = make_behavior_nominal()
    traces = [BehaviorTrace(t.seed, t.mode, False, t.grasp,
                            tuple(SubtaskProgress(p, True) for p in PICK) + (SubtaskProgress("post_place", False),),
                            t.code_key)
              for t in make_behavior_traces(8, Mode.CLEAN, BehaviorFaults())]
    a = analyze_behavior(traces, nominal)
    assert {e.signal for e in a.evidence} == {V.SIG_IMPLIED_TARGET_OK, V.failure_phase("post_place")}
    v = decide(list(a.evidence), _table(SCENE))
    assert (v.row, v.label) == (8, VerdictLabel.ACTION_GAP) and "failure at post_place" in v.notes


def test_the_synthetic_generator_stays_pick_only_unless_asked_for_post_place():
    [t] = make_behavior_traces(1, Mode.CLEAN, BehaviorFaults(fail_phase="lift"))
    assert [s.name for s in t.subtasks] == list(PICK)
    [d] = make_behavior_traces(1, Mode.CLEAN, BehaviorFaults(fail_phase="post_place"))
    assert [(s.name, s.completed) for s in d.subtasks] == [(p, True) for p in PICK] + [("post_place", False)]
    with pytest.raises(ValueError):
        SubtaskProgress("close_door", True)          # composite names are mapped by the backend, not accepted here
