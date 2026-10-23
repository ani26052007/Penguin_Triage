"""Rung 2 (plan.md Section 4): one big-model call, three labels, and the rows 6-7 gate.

A "data gap" answer must match a router one-factor or all-factors restore (decision rows 6 and 7).
Rung 2 cannot create a vision route on its own: that is enforced in code (``data_gap_allowed``),
whatever the model says.
"""

from __future__ import annotations

import json

import pytest

from hagent.exceptions import ModelError
from hagent.models import ScriptedModel
from ladder import rung2
from ladder.rung2 import Rung2Input, Rung2Label, data_gap_allowed, triage
from router import vocab as V
from router.schemas import (ActorRef, ActorKind, CodeRef, Destination, EpisodeResult, Evidence, Mode, Outcome,
                            ReplayResult, SceneSpec, Verdict, VerdictLabel)

SCENE = SceneSpec("pick_mug")
EV = (Evidence("ev_contract", "stage1.cfg_contract", "contract_violation", 0.8, "norm_stats", "mean differs"),
      Evidence("ev_other", "fixture", "x", 0.4))


def replay(rid: str, k: int, outcome: Outcome) -> ReplayResult:
    eps = tuple(EpisodeResult(i, i < k) for i in range(10))
    return ReplayResult(rid, SCENE, CodeRef("c1"), ActorRef(ActorKind.POLICY, "ckpt"), Mode.CLEAN, eps, outcome)


RESTORE = replay(V.restore_id("lighting"), 9, Outcome.RESTORED)
CALL = replay(V.R_C_ALL, 9, Outcome.RESTORED)
STAGE0 = replay(V.R_STAGE0, 4, Outcome.PARTIAL)


def verdict(row: int, replays=()) -> Verdict:
    label = {4: VerdictLabel.CODE, 6: VerdictLabel.VISION_FACTOR, 7: VerdictLabel.VISION_INTERACTING,
             9: VerdictLabel.PERCEPTION_UNRESOLVED, 10: VerdictLabel.UNDETERMINED}[row]
    dest = {4: Destination.HAGENT, 6: Destination.DATA_LADDER, 7: Destination.DATA_LADDER}.get(row, Destination.REPORT)
    return Verdict(label, dest, row, factor="lighting" if row in (6, 7) else None, replays=tuple(replays))


def inp(v: Verdict) -> Rung2Input:
    return Rung2Input("inc1", v, (STAGE0, RESTORE), EV, "RUNG1 AMBIGUOUS  stage B, n=50")


def reply(label: str, reason: str = "because", ids=("ev_contract",)) -> str:
    return json.dumps({"label": label, "reason": reason, "evidence_ids": list(ids)})


# --------------------------------------------------------------------------- #
# the gate: pure function of the router verdict
# --------------------------------------------------------------------------- #

def test_gate_allows_rows_6_and_7_with_a_restore_attached():
    assert data_gap_allowed(verdict(6, [RESTORE])) == (True, "")
    assert data_gap_allowed(verdict(7, [CALL]))[0]


@pytest.mark.parametrize("row", [4, 9, 10])
def test_gate_rejects_every_other_row(row):
    ok, why = data_gap_allowed(verdict(row, [RESTORE]))
    assert not ok and "rows 6-7" in why


def test_gate_rejects_a_vision_verdict_without_a_restore_replay():
    ok, why = data_gap_allowed(verdict(6, [STAGE0]))
    assert not ok and "restore" in why


# --------------------------------------------------------------------------- #
# required: "data gap" is rejected when no router row 6-7 restore exists
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("v", [verdict(4), verdict(10), verdict(9), verdict(6, [STAGE0])],
                         ids=["code_row", "undetermined_row", "perception_row", "row6_without_restore"])
def test_data_gap_is_rejected_when_no_row_6_7_restore_exists(v):
    model = ScriptedModel([reply("data_gap", "the policy never saw this lighting")])
    res = triage(inp(v), model)
    assert res.model_label is Rung2Label.DATA_GAP            # what the model said is kept for the log
    assert res.label is Rung2Label.UNCLEAR and res.accepted is False
    assert res.rejected_reason.startswith("data_gap rejected")
    assert len(model.calls) == 1 and res.calls == 1          # a gate rejection is not a retry: one call


def test_data_gap_is_accepted_with_a_row_6_restore():
    res = triage(inp(verdict(6, [RESTORE])), ScriptedModel([reply("data_gap")]))
    assert res.label is Rung2Label.DATA_GAP and res.accepted and not res.rejected_reason
    assert res.evidence_ids == ("ev_contract",) and res.source == "model"


# --------------------------------------------------------------------------- #
# the other labels, validation, failure handling
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("label", ["code", "unclear"])
def test_code_and_unclear_pass_through(label):
    res = triage(inp(verdict(4)), ScriptedModel([reply(label)]))
    assert res.label is Rung2Label(label) and res.accepted


def test_bad_reply_gets_one_retry_with_the_reason():
    model = ScriptedModel([json.dumps({"label": "code", "reason": "x", "evidence_ids": ["nope"], "route": "hagent"}),
                           reply("code")])
    res = triage(inp(verdict(4)), model)
    assert res.label is Rung2Label.CODE and res.calls == 2
    assert "rejected" in model.calls[1][-1]["content"] and "route" in model.calls[1][-1]["content"]


@pytest.mark.parametrize("bad", [
    reply("maybe"), reply("code", ids=("ghost",)), reply("code", reason=""), reply("code", reason="x" * 601),
    json.dumps({"label": "code", "reason": "r"}), "not json", json.dumps({"label": "code", "reason": "r",
                                                                         "evidence_ids": [], "destination": "stop"})])
def test_malformed_replies_fall_back_to_unclear_and_never_raise(bad):
    res = triage(inp(verdict(4)), ScriptedModel([bad, bad]))
    assert res.label is Rung2Label.UNCLEAR and res.model_label is None and res.source == "fallback"
    assert not res.accepted and res.calls == 2 and "no usable reply" in res.reason


def test_provider_error_and_no_model_fall_back_to_unclear():
    def boom(messages):
        raise ModelError("503")
    res = triage(inp(verdict(4)), ScriptedModel([boom]))
    assert res.label is Rung2Label.UNCLEAR and res.source == "fallback" and "503" in res.reason
    none = triage(inp(verdict(4)), None)
    assert none.label is Rung2Label.UNCLEAR and none.calls == 0 and none.reason == "no model configured"


def test_cost_and_tokens_are_counted():
    model = ScriptedModel([reply("code")], cost_per_call=0.07, tokens_per_call=(100, 20))
    res = triage(inp(verdict(4)), model)
    assert res.cost == pytest.approx(0.07) and res.tokens == 120


def test_prompt_carries_the_verdict_replays_evidence_and_rung1_but_no_route_choice():
    model = ScriptedModel([reply("code")])
    triage(inp(verdict(6, [RESTORE])), model)
    system, user = (m["content"] for m in model.calls[0])
    assert "vision_factor (decision-table row 6)" in user
    assert f"{V.restore_id('lighting')}: restored (9/10 succeeded)" in user
    assert "ev_contract" in user and "RUNG1 AMBIGUOUS" in user
    assert "rows 6-7" in system and "unclear" in system


def test_extra_blocks_are_added_to_the_prompt():
    i = Rung2Input("inc1", verdict(4), (STAGE0,), EV, "r1", {"Per-subtask progress": "reach 10/10, grasp 3/10"})
    msgs = rung2.build_messages(i)
    assert "## Per-subtask progress\nreach 10/10, grasp 3/10" in msgs[1]["content"]
