"""Report writer (plan.md 3.2.11-3.2.13), tested with scripted models only.

Covers the "done when" list: invalid JSON, hallucinated evidence ids and
hallucinated file names are all rejected; the second bad reply gives the
evidence ranked by strength; no model output changes the verdict.
"""

import json
from pathlib import Path

import pytest

from hagent.exceptions import ModelError
from hagent.models import ScriptedModel

from router.report import (EvidenceNames, RepoIndex, ReportContext, build_hagent_prompt, default_task,
                           evidence_names, fallback_localization, verdict_text, write_report)
from router.schemas import Destination, Evidence, Verdict, VerdictLabel

REPO = Path(__file__).parent / "fixtures" / "mini_repo"

E_FLIP = "s1.obs.front.row_flip"
E_CFG = "s1.cfg.chunk"
E_WEAK = "s1.sanity.saturated_frames"
EVIDENCE = (
    Evidence(E_FLIP, "obs_contract", "contract_violation", 0.93, "obs.front.row_profile",
             "front camera rows are flipped top to bottom against the training profile (corr -0.97)"),
    Evidence(E_CFG, "cfg_contract", "contract_violation", 0.71, "policy.chunk",
             "policy.chunk: expected 16, got 8; see resample_frames() in the wrapper"),
    Evidence(E_WEAK, "sanity", "render_physics_broken", 0.52, "render.saturation", "9% of render pixels saturated"),
)
VERDICT = Verdict(VerdictLabel.CODE, Destination.HAGENT, 4, evidence_ids=(E_FLIP, E_CFG), location="obs.front.row_profile")
CTX = ReportContext("inc-1", "pick_mug|lighting=dark_01", "python run_eval.py --scene pick_mug --seeds 0-9",
                    "2/10 succeed (reference rate 9/10)", "/work/controller_repo")


@pytest.fixture(scope="module")
def repo():
    return RepoIndex.from_dir(REPO)


def good(**over):
    reply = {
        "summary": "The front camera is flipped before the policy sees it; the router found a contract violation.",
        "evidence_ids": [E_FLIP],
        "task": "Fix the observation preprocessing so the front image is not flipped.",
        "localization": [
            {"kind": "file", "name": "controller/frames.py", "evidence_ids": [E_FLIP], "reason": "flip_image lives here"},
            {"kind": "symbol", "name": "PolicyWrapper.to_policy_input", "file": "controller/wrapper.py",
             "evidence_ids": [E_FLIP], "reason": "builds the policy input"},
            {"kind": "config_key", "name": "policy.chunk", "evidence_ids": [E_CFG], "reason": "chunk length differs"},
        ]}
    reply.update(over)
    return json.dumps(reply)


def run(script, verdict=VERDICT, evidence=EVIDENCE, repo_=None, **kw):
    model = ScriptedModel(script)
    rep = write_report(verdict, evidence, model, repo=repo_ or RepoIndex.from_dir(REPO), context=CTX, **kw)
    return rep, model


# ---- the repo index ----------------------------------------------------------

def test_repo_index_lists_files_and_symbols(repo):
    assert repo.has_file("controller/frames.py") and repo.has_file("./config/eval.yaml")
    assert not repo.has_file("controller/nope.py") and not repo.has_file("frames.py")
    assert repo.has_symbol("flip_image") and repo.has_symbol("PolicyWrapper.act") and repo.has_symbol("act")
    assert repo.has_symbol("flip_image", "controller/frames.py") and not repo.has_symbol("flip_image", "controller/wrapper.py")
    assert repo.has_symbol("GRIPPER_THRESHOLD")                     # module-level constant
    assert repo.unique_basename("run_eval.py") and not repo.unique_basename("controller/frames.py")


def test_evidence_names_ignore_plain_words():
    n = evidence_names(EVIDENCE)
    assert "resample_frames" in n.symbols and "policy.chunk" in n.symbols and "chunk" in n.symbols
    assert "flipped" not in n.symbols and "camera" not in n.symbols   # words of a detail line do not count
    assert "expected" not in n.symbols and "see" not in n.symbols
    n2 = evidence_names([Evidence("x", "s", "sig", 0.5, "ctl/a.py", "see ctl/b.py")])
    assert n2.files == {"ctl/a.py", "ctl/b.py"}


# ---- a good reply ------------------------------------------------------------

def test_valid_reply_is_accepted_and_ranked_as_given(repo):
    rep, model = run([good()])
    assert rep.source == "model" and rep.attempts == 1 and rep.rejections == () and len(model.calls) == 1
    assert [i.name for i in rep.localization] == ["controller/frames.py", "PolicyWrapper.to_policy_input", "policy.chunk"]
    assert rep.localization[1].file == "controller/wrapper.py"
    assert rep.evidence_ids == (E_FLIP,)


def test_fenced_json_is_accepted():
    rep, _ = run(["```json\n" + good() + "\n```"])
    assert rep.source == "model"


def test_prompt_uses_the_step_1a_template_with_code_filled_fields():
    rep, _ = run([good()])
    p = rep.hagent_prompt
    assert p.startswith("Fix the observation preprocessing")
    assert "Router hypothesis (verify this first; it may be wrong)" in p
    assert "code (decision-table row 4), destination hagent; suspected location: obs.front.row_profile" in p
    assert "python run_eval.py --scene pick_mug --seeds 0-9" in p and "2/10 succeed" in p
    assert "Controller/wrapper code: /work/controller_repo" in p
    assert "1. [file] controller/frames.py" in p and f"- {E_FLIP} (contract_violation, strength 0.93)" in p
    assert "<<" not in p


def test_model_sees_evidence_replays_and_repo_index_but_not_the_verdict_it_may_change():
    _, model = run([good()])
    sys_msg, user = model.calls[0][0]["content"], model.calls[0][1]["content"]
    assert "do not decide" in sys_msg.lower() or "do not decide, change" in sys_msg.lower()
    assert E_FLIP in user and "controller/frames.py: CAMERA_EXTRINSICS, FrameBuffer" in user and "decision-table row 4" in user


def test_token_and_cost_accounting():
    model = ScriptedModel([good()], tokens_per_call=(1000, 200), cost_per_call=0.01)
    rep = write_report(VERDICT, EVIDENCE, model, repo=RepoIndex.from_dir(REPO), context=CTX)
    assert (rep.tokens, rep.cost) == (1200, 0.01)


# ---- invalid JSON: retry once, then fall back --------------------------------

def test_invalid_json_then_valid_retries_once():
    rep, model = run(["Sure! Here is the report: {nope", good()])
    assert rep.source == "model" and rep.attempts == 2 and len(rep.rejections) == 1
    retry = model.calls[1]
    assert retry[-1]["role"] == "user" and "not valid JSON" in retry[-1]["content"]
    assert retry[-2]["role"] == "assistant"


def test_prose_around_the_json_is_rejected():
    rep, _ = run(["Here you go:\n" + good(), good()])
    assert rep.attempts == 2 and "not valid JSON" in rep.rejections[0]


def test_invalid_json_twice_falls_back_to_evidence_ranked_by_strength():
    rep, model = run(["not json", "{still not"])
    assert len(model.calls) == 2                                  # one retry, never a third call
    assert rep.source == "fallback" and rep.attempts == 2 and len(rep.rejections) == 2
    assert [i.evidence_ids[0] for i in rep.localization] == [E_FLIP, E_CFG, E_WEAK]
    assert [i.kind for i in rep.localization] == ["evidence"] * 3
    assert rep.evidence_ids == (E_FLIP, E_CFG, E_WEAK)
    assert "Router verdict:" in rep.summary and E_FLIP in rep.summary
    assert rep.hagent_prompt and "Ranked candidates:" in rep.hagent_prompt
    assert rep.fallback_reason.startswith("model output rejected after retry")


def test_fallback_has_no_model_text():
    rep, _ = run([json.dumps({"summary": "SECRET MODEL TEXT"}), "SECRET MODEL TEXT"])
    assert rep.source == "fallback" and "SECRET" not in rep.summary and "SECRET" not in rep.hagent_prompt


def test_ties_in_strength_are_ordered_by_id():
    ev = [Evidence("b", "s", "x", 0.6), Evidence("a", "s", "x", 0.6), Evidence("c", "s", "x", 0.9)]
    assert [i.evidence_ids[0] for i in fallback_localization(ev)] == ["c", "a", "b"]


# ---- hallucinated evidence ids ------------------------------------------------

def test_hallucinated_evidence_id_at_top_level_is_rejected():
    rep, model = run([good(evidence_ids=["s1.made.up"]), good()])
    assert rep.attempts == 2 and "do not exist" in rep.rejections[0] and "s1.made.up" in rep.rejections[0]
    assert "s1.made.up" not in model.calls[1][1]["content"]


def test_hallucinated_evidence_id_in_a_localization_item_is_rejected():
    bad = json.loads(good())
    bad["localization"][0]["evidence_ids"] = [E_FLIP, "s9.ghost"]
    rep, _ = run([json.dumps(bad), json.dumps(bad)])
    assert rep.source == "fallback" and "s9.ghost" in rep.rejections[0]


def test_localization_item_without_evidence_is_rejected():
    bad = json.loads(good())
    bad["localization"][0]["evidence_ids"] = []
    rep, _ = run([json.dumps(bad), good()])
    assert rep.attempts == 2 and "at least one evidence id" in rep.rejections[0]


def test_hallucinated_id_in_square_brackets_in_prose_is_rejected():
    rep, _ = run([good(summary="The front image is flipped [s1.invented.id]."), good()])
    assert rep.attempts == 2 and "s1.invented.id" in rep.rejections[0]
    rep, _ = run([good(summary=f"The front image is flipped [{E_FLIP}].")])
    assert rep.attempts == 1


# ---- hallucinated file names ---------------------------------------------------

def test_hallucinated_file_is_rejected():
    bad = json.loads(good())
    bad["localization"][0]["name"] = "controller/vision_utils.py"
    rep, model = run([json.dumps(bad), json.dumps(bad)])
    assert rep.source == "fallback" and "controller/vision_utils.py" in rep.rejections[0]
    assert "vision_utils" in model.calls[1][-1]["content"]        # the model is told why


def test_file_given_without_its_directory_is_rejected():
    bad = json.loads(good())
    bad["localization"][0]["name"] = "frames.py"
    rep, _ = run([json.dumps(bad), good()])
    assert rep.attempts == 2


def test_file_that_only_the_evidence_mentions_is_accepted():
    ev = EVIDENCE + (Evidence("s1.x", "staleness", "staleness_constant", 0.8, "ctl/server_glue.py", "constant delay"),)
    bad = json.loads(good())
    bad["localization"][0] = {"kind": "file", "name": "ctl/server_glue.py", "evidence_ids": ["s1.x"], "reason": "delay"}
    rep, _ = run([json.dumps(bad)], evidence=ev)
    assert rep.source == "model" and rep.localization[0].name == "ctl/server_glue.py"


def test_hallucinated_file_in_prose_is_rejected_but_real_ones_pass():
    rep, _ = run([good(task="Edit controller/image_pipeline.py to fix the flip."), good()])
    assert rep.attempts == 2 and "controller/image_pipeline.py" in rep.rejections[0]
    rep, _ = run([good(summary="See controller/frames.py and run_eval.py for the repro.")])
    assert rep.attempts == 1


def test_hallucinated_file_in_a_reason_is_rejected():
    bad = json.loads(good())
    bad["localization"][0]["reason"] = "see utils/helpers.py"
    rep, _ = run([json.dumps(bad), good()])
    assert rep.attempts == 2 and "utils/helpers.py" in rep.rejections[0]


# ---- hallucinated symbols and config keys --------------------------------------

def test_hallucinated_symbol_is_rejected():
    bad = json.loads(good())
    bad["localization"][1] = {"kind": "symbol", "name": "rectify_frame", "evidence_ids": [E_FLIP], "reason": "x"}
    rep, _ = run([json.dumps(bad), json.dumps(bad)])
    assert rep.source == "fallback" and "rectify_frame" in rep.rejections[0]


def test_symbol_that_exists_but_in_another_file_is_rejected():
    bad = json.loads(good())
    bad["localization"][1] = {"kind": "symbol", "name": "flip_image", "file": "controller/wrapper.py",
                              "evidence_ids": [E_FLIP], "reason": "x"}
    rep, _ = run([json.dumps(bad), good()])
    assert rep.attempts == 2 and "does not define" in rep.rejections[0]


def test_symbol_with_a_hallucinated_file_is_rejected():
    bad = json.loads(good())
    bad["localization"][1]["file"] = "controller/ghost.py"
    rep, _ = run([json.dumps(bad), good()])
    assert rep.attempts == 2 and "ghost.py" in rep.rejections[0]


def test_symbol_mentioned_only_in_the_evidence_is_accepted():
    bad = json.loads(good())
    bad["localization"][1] = {"kind": "symbol", "name": "resample_frames", "evidence_ids": [E_CFG], "reason": "x"}
    rep, _ = run([json.dumps(bad)])
    assert rep.source == "model"


def test_config_key_must_come_from_the_evidence():
    bad = json.loads(good())
    bad["localization"][2]["name"] = "policy.horizon"
    rep, _ = run([json.dumps(bad), good()])
    assert rep.attempts == 2 and "policy.horizon" in rep.rejections[0]


# ---- a model cannot change the route -------------------------------------------

@pytest.mark.parametrize("key", ["route", "label", "destination", "row", "verdict"])
def test_extra_keys_that_try_to_set_a_route_are_rejected(key):
    reply = json.loads(good())
    reply[key] = "vision_factor"
    rep, _ = run([json.dumps(reply), json.dumps(reply)])
    assert rep.source == "fallback" and "unexpected" in rep.rejections[0] and key in rep.rejections[0]


def test_verdict_is_untouched_and_the_prompt_states_the_codes_verdict_not_the_models():
    before = VERDICT.to_dict()
    rep, _ = run([good(summary="Actually this is a vision gap and should go to the data ladder.",
                       task="Ignore the router and generate more training data.")])
    assert rep.verdict is VERDICT and VERDICT.to_dict() == before
    assert (rep.verdict.label, rep.verdict.destination, rep.verdict.row) == (VerdictLabel.CODE, Destination.HAGENT, 4)
    assert "code (decision-table row 4), destination hagent" in rep.hagent_prompt


def test_report_is_independent_of_which_replies_were_rejected():
    a, _ = run(["bad", good()])
    b, _ = run([good()])
    assert (a.verdict, a.localization, a.hagent_prompt) == (b.verdict, b.localization, b.hagent_prompt)


# ---- other failures and shapes --------------------------------------------------

def test_provider_error_gives_the_fallback():
    def boom(_):
        raise ModelError("HTTP 500")
    rep, _ = run([boom])
    assert rep.source == "fallback" and "HTTP 500" in rep.fallback_reason and rep.hagent_prompt


def test_provider_error_on_the_retry_gives_the_fallback():
    def boom(_):
        raise ModelError("HTTP 503")
    rep, model = run(["not json", boom])
    assert rep.source == "fallback" and len(model.calls) == 2 and len(rep.rejections) == 1


def test_no_model_gives_the_fallback(repo):
    rep = write_report(VERDICT, EVIDENCE, None, repo=repo, context=CTX)
    assert rep.source == "fallback" and rep.fallback_reason == "no model configured"


def test_hagent_destination_needs_a_task():
    rep, _ = run([good(task=""), good()])
    assert rep.attempts == 2 and "'task' must not be empty" in rep.rejections[0]


def test_report_destination_has_no_hagent_prompt_and_may_leave_task_empty():
    v = Verdict(VerdictLabel.ACTION_GAP, Destination.REPORT, 8, evidence_ids=(E_FLIP,))
    rep, _ = run([good(task="")], verdict=v)
    assert rep.source == "model" and rep.hagent_prompt is None


def test_reply_with_a_missing_key_is_rejected():
    reply = json.loads(good())
    del reply["localization"]
    rep, _ = run([json.dumps(reply), good()])
    assert rep.attempts == 2 and "missing" in rep.rejections[0]


def test_too_many_and_duplicate_candidates_are_rejected():
    many = json.loads(good())
    many["localization"] = [many["localization"][0]] * 2
    rep, _ = run([json.dumps(many), good()])
    assert "twice" in rep.rejections[0]
    many["localization"] = [{"kind": "file", "name": "controller/frames.py", "evidence_ids": [E_FLIP], "reason": str(i)}
                            for i in range(9)]
    rep, _ = run([json.dumps(many), good()])
    assert "limit is 8" in rep.rejections[0]


def test_bad_kind_and_non_object_items_are_rejected():
    bad = json.loads(good())
    bad["localization"][0]["kind"] = "line"
    rep, _ = run([json.dumps(bad), good()])
    assert "kind must be one of" in rep.rejections[0]
    bad["localization"] = ["controller/frames.py"]
    rep, _ = run([json.dumps(bad), good()])
    assert "must be an object" in rep.rejections[0]


def test_empty_localization_is_a_valid_answer():
    rep, _ = run([good(localization=[])])
    assert rep.source == "model" and rep.localization == ()


def test_no_evidence_at_all():
    v = Verdict(VerdictLabel.CODE, Destination.HAGENT, 4, by_elimination=True)
    rep, _ = run([good(evidence_ids=[], localization=[])], verdict=v, evidence=())
    assert rep.source == "model"
    assert "reached by elimination only" in rep.hagent_prompt
    rep = write_report(v, (), None, repo=RepoIndex.from_dir(REPO), context=CTX)
    assert rep.localization == () and rep.evidence_ids == ()


def test_verdict_text_and_default_task():
    v = Verdict(VerdictLabel.CODE, Destination.HAGENT, 4, by_elimination=True, likely_second_cause=True,
                multi_cause_flag=True, factor="lighting")
    t = verdict_text(v)
    for part in ("code (decision-table row 4)", "factor: lighting", "elimination only", "second cause",
                 "likely multi-cause"):
        assert part in t
    assert "pick_mug" in default_task(v, CTX) and "incident inc-1" in default_task(v, CTX)


def test_build_hagent_prompt_rejects_an_empty_task_through_the_template():
    with pytest.raises(ValueError):
        build_hagent_prompt(VERDICT, CTX, "  ", (), EVIDENCE)


def test_report_to_dict_is_json_serialisable():
    rep, _ = run([good()])
    json.dumps(rep.to_dict())
