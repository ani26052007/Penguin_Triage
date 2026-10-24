"""Factor-search stub for verdict row 9 (plan.md 3.2.12), scripted model."""

import json

import pytest

from hagent.exceptions import ModelError
from hagent.models import ScriptedModel

from router.factor_search import FrameRef, VisionFactorSearch
from router.schemas import Destination, Verdict, VerdictLabel

ROW9 = Verdict(VerdictLabel.PERCEPTION_UNRESOLVED, Destination.REPORT, 9, evidence_ids=("behavior.implied_target",))
NOMINAL = [FrameRef("front@t10", "data:image/png;base64,AAAA"), FrameRef("wrist@t10", "https://x/n2.png")]
FAILING = [FrameRef("front@t10", "data:image/png;base64,BBBB")]


def reply(*factors):
    return json.dumps({"factors": [{"name": n, "description": d} for n, d in factors]})


def test_proposals_come_back_as_unconfirmed_candidates():
    m = ScriptedModel([reply(("lighting", "the failing frames are much darker"),
                             ("material(mug)", "the mug looks glossy"))])
    out = VisionFactorSearch(m).propose(ROW9, NOMINAL, FAILING, tested_factors=["camera_offset"])
    assert out.source == "model" and out.attempts == 1
    assert [c.name for c in out.candidates] == ["lighting", "material(mug)"]


def test_request_carries_nominal_and_failing_frames_and_the_tested_list():
    m = ScriptedModel([reply()])
    VisionFactorSearch(m).propose(ROW9, NOMINAL, FAILING, tested_factors=["camera_offset", "lighting"])
    user = m.calls[0][1]["content"]
    urls = [p["image_url"]["url"] for p in user if p["type"] == "image_url"]
    assert urls == ["data:image/png;base64,AAAA", "https://x/n2.png", "data:image/png;base64,BBBB"]
    assert any("camera_offset, lighting" in p.get("text", "") for p in user)
    assert [p["text"] for p in user if p["type"] == "text"][0].startswith("NOMINAL")


def test_frames_per_side_are_capped():
    m = ScriptedModel([reply()])
    many = [FrameRef(f"f{i}", f"https://x/{i}.png") for i in range(10)]
    VisionFactorSearch(m, max_frames_per_side=2).propose(ROW9, many, many)
    assert sum(p["type"] == "image_url" for p in m.calls[0][1]["content"]) == 4


@pytest.mark.parametrize("row", [1, 4, 6, 8, 10])
def test_only_row_9(row):
    v = Verdict(VerdictLabel.UNDETERMINED, Destination.REPORT, row)
    m = ScriptedModel([reply()])
    with pytest.raises(ValueError, match="row 9"):
        VisionFactorSearch(m).propose(v, NOMINAL, FAILING)
    assert m.calls == []


def test_needs_frames_on_both_sides():
    with pytest.raises(ValueError):
        VisionFactorSearch(ScriptedModel([reply()])).propose(ROW9, [], FAILING)
    with pytest.raises(ValueError):
        VisionFactorSearch(ScriptedModel([reply()])).propose(ROW9, NOMINAL, [])


def test_an_already_tested_factor_is_rejected_then_retried():
    m = ScriptedModel([reply(("lighting", "darker")), reply(("texture(table)", "the table texture changed"))])
    out = VisionFactorSearch(m).propose(ROW9, NOMINAL, FAILING, tested_factors=["lighting"])
    assert out.attempts == 2 and [c.name for c in out.candidates] == ["texture(table)"]


@pytest.mark.parametrize("bad", [
    "not json", reply(("Lighting Is Different", "x")), reply(("a_b", "")), reply(("a", "x"), ("a", "y")),
    reply(*[(f"f{i}", "x") for i in range(4)]), json.dumps({"factors": "lighting"}),
    json.dumps({"factors": [], "route": "vision"}), json.dumps({"factors": [{"name": "a_b"}]}),
])
def test_bad_replies_give_no_proposal_never_a_made_up_one(bad):
    m = ScriptedModel([bad, bad])
    out = VisionFactorSearch(m).propose(ROW9, NOMINAL, FAILING)
    assert out.candidates == () and out.source == "none" and out.attempts == 2 and len(m.calls) == 2 and out.reason


def test_provider_error_gives_no_proposal():
    def boom(_):
        raise ModelError("HTTP 400: image input not supported")
    out = VisionFactorSearch(ScriptedModel([boom])).propose(ROW9, NOMINAL, FAILING)
    assert out.source == "none" and "image input not supported" in out.reason


def test_an_empty_list_is_a_valid_answer():
    out = VisionFactorSearch(ScriptedModel([reply()])).propose(ROW9, NOMINAL, FAILING)
    assert out.source == "model" and out.candidates == ()


def test_the_verdict_is_untouched():
    before = ROW9.to_dict()
    VisionFactorSearch(ScriptedModel([reply(("lighting", "darker"))])).propose(ROW9, NOMINAL, FAILING)
    assert ROW9.to_dict() == before and ROW9.row == 9
