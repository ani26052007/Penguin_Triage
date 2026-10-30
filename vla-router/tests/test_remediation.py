"""Remediation advisor (docs/REMEDIATION.md): deterministic, no model, no change to any decision.

The menu is a pure function of the verdict label (and factor). The report writer appends it as plain text
for verdicts that have one; no model reply can add, drop or reorder an option.
"""

import ast
import dataclasses
import json
import re
from pathlib import Path

import pytest

from hagent.models import ScriptedModel

from router import remediation as R
from router.remediation import RemediationOption, menu_text, remediation_options
from router.report import RepoIndex, ReportContext, write_report
from router.schemas import Destination, Evidence, Verdict, VerdictLabel

REPO_ROOT = Path(__file__).parents[2]
ROUTER = Path(__file__).parents[1] / "router"
MINI_REPO = Path(__file__).parent / "fixtures" / "mini_repo"
CTX = ReportContext("inc-1", "pick_mug|lighting=dark_01", "python run_eval.py", "2/10 succeed", "/work/repo")
E = (Evidence("s1.sanity.x", "sanity", "render_physics_broken", 0.5, "render.x", "something"),)

ROWS = {VerdictLabel.NO_SIGNIFICANT_FAILURE: 1, VerdictLabel.SIM_ASSET: 2, VerdictLabel.INFRA: 3, VerdictLabel.CODE: 4,
        VerdictLabel.RENDERING_MISMATCH: 5, VerdictLabel.VISION_FACTOR: 6, VerdictLabel.VISION_INTERACTING: 7,
        VerdictLabel.ACTION_GAP: 8, VerdictLabel.PERCEPTION_UNRESOLVED: 9, VerdictLabel.UNDETERMINED: 10}
DEST = {1: Destination.STOP, 2: Destination.REPORT, 3: Destination.REPORT, 4: Destination.HAGENT,
        5: Destination.CONFIG_FIX, 6: Destination.DATA_LADDER, 7: Destination.DATA_LADDER, 8: Destination.REPORT,
        9: Destination.REPORT, 10: Destination.REPORT}


def verdict(label, factor=None):
    row = ROWS[label]
    return Verdict(label, DEST[row], row, factor=factor)


def ids(label, factor=None):
    return [o.id for o in remediation_options(verdict(label, factor))]


def test_vision_rows_list_augmentation_rerender_generative_and_finetune_in_that_order():
    for label in (VerdictLabel.VISION_FACTOR, VerdictLabel.VISION_INTERACTING):
        got = remediation_options(verdict(label, "lighting"))
        assert [o.id for o in got] == ["vision-augmentation", "vision-rerender-variation", "vision-rerender-mimic",
                                       "vision-generative-rerender", "vision-finetune"]
        by = {o.id: o for o in got}
        assert "rung_a.py" in by["vision-augmentation"].repo_entry_point
        assert "lighting" in by["vision-rerender-variation"].title and "lighting" in by["vision-rerender-mimic"].title
        assert by["vision-rerender-mimic"].status.startswith(R.BLOCKED_PREFIX)
        assert by["vision-generative-rerender"].status == R.STATUS_NOT_BUILT
        assert by["vision-generative-rerender"].repo_entry_point == R.NOT_IMPLEMENTED
        assert "finetune_launcher.py" in by["vision-finetune"].repo_entry_point


def test_vision_menu_names_a_factor_only_when_there_is_one():
    assert "the failing factor" in remediation_options(verdict(VerdictLabel.VISION_FACTOR))[1].title
    assert ids(VerdictLabel.VISION_FACTOR) == ids(VerdictLabel.VISION_FACTOR, "texture")


def test_action_gap_row_lists_four_options_and_links_the_teleop_reference():
    got = remediation_options(verdict(VerdictLabel.ACTION_GAP))
    assert [o.id for o in got] == ["action-scripted-demos", "action-rl-oracle", "action-teleop-mimic",
                                   "action-own-rollouts"]
    by = {o.id: o for o in got}
    assert by["action-rl-oracle"].status == R.STATUS_NOT_BUILT
    assert by["action-teleop-mimic"].status.startswith(R.BLOCKED_PREFIX)
    assert R.TELEOP_REFERENCE in by["action-teleop-mimic"].prerequisites
    assert "annotated source demonstrations" in by["action-teleop-mimic"].caveat
    assert "untested with Arena's annotation flow" in by["action-teleop-mimic"].caveat
    assert "not expert data" in by["action-own-rollouts"].caveat


def test_perception_unresolved_row_proposes_factors_and_reports_a_new_modality_only():
    got = remediation_options(verdict(VerdictLabel.PERCEPTION_UNRESOLVED))
    assert [o.id for o in got] == ["perception-factor-search", "perception-new-modality"]
    assert got[0].repo_entry_point == "vla-router/router/factor_search.py" and got[0].status == R.STATUS_NOT_RUN
    assert got[1].caveat == "Report only." and got[1].status == R.STATUS_NOT_BUILT


@pytest.mark.parametrize("label", [VerdictLabel.SIM_ASSET, VerdictLabel.INFRA, VerdictLabel.CODE,
                                   VerdictLabel.NO_SIGNIFICANT_FAILURE, VerdictLabel.UNDETERMINED,
                                   VerdictLabel.RENDERING_MISMATCH])
def test_other_labels_have_no_data_or_model_options(label):
    assert remediation_options(verdict(label)) == ()
    assert menu_text(verdict(label)) == ""


def all_options():
    out = []
    for label in VerdictLabel:
        out += remediation_options(verdict(label, "f"))
    return out


def test_every_label_is_covered_by_the_row_table_of_this_test():
    assert set(ROWS) == set(VerdictLabel)


def test_every_repo_entry_point_that_is_a_path_exists():
    checked = 0
    for o in all_options():
        assert o.repo_entry_point == R.NOT_IMPLEMENTED or o.entry_paths(), o.id
        for p in o.entry_paths():
            assert (REPO_ROOT / p).exists(), f"{o.id}: {p}"
            checked += 1
    assert checked >= 8


def test_status_strings_come_from_a_fixed_set():
    for o in all_options():
        assert R.status_is_valid(o.status), o
    assert not R.status_is_valid("works")
    assert not R.status_is_valid(R.BLOCKED_PREFIX)


def test_options_are_frozen_and_complete():
    o = all_options()[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        o.status = "implemented, not run"
    for o in all_options():
        assert all(isinstance(getattr(o, f.name), str) and getattr(o, f.name) for f in dataclasses.fields(o)), o.id


def test_the_function_is_pure():
    v = verdict(VerdictLabel.ACTION_GAP)
    assert remediation_options(v) == remediation_options(v)
    assert menu_text(v) == menu_text(v)


def test_no_option_claims_success_or_a_run():
    banned = re.compile(r"\b(restores?|guarantee[sd]?|proven|fixes? the)\b", re.I)
    for o in all_options():
        assert not banned.search(" ".join(dataclasses.astuple(o))), o.id


# --------------------------------------------------------------------------- #
# the report appends the menu; no model touches it
# --------------------------------------------------------------------------- #

def imports(path: Path) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


def test_the_remediation_module_imports_no_model_and_no_decision_code():
    bad = {m for m in imports(ROUTER / "remediation.py")
           if m.split(".")[0] == "hagent" or m.startswith(("router.llm", "router.report", "router.config_mapper",
                                                           "router.factor_search", "router.decide", "router.router",
                                                           "router.replay", "router.stage", "router.branches"))}
    assert not bad, sorted(bad)


def test_the_report_carries_the_menu_for_a_vision_verdict_without_a_model():
    v = verdict(VerdictLabel.VISION_FACTOR, "lighting")
    rep = write_report(v, E, None, repo=RepoIndex.from_dir(MINI_REPO), context=CTX)
    assert rep.remediation == menu_text(v)
    assert "user's choice; not chosen by a model" in rep.remediation
    assert R.SCOPE_LINE in rep.remediation and rep.to_dict()["remediation"] == rep.remediation
    json.dumps(rep.to_dict())
    assert rep.verdict is v and rep.hagent_prompt is None


def test_a_model_reply_cannot_change_or_add_to_the_menu():
    v = verdict(VerdictLabel.ACTION_GAP)
    good = json.dumps({"summary": "the policy lacks the skill", "evidence_ids": ["s1.sanity.x"], "task": "t",
                       "localization": []})
    wants_menu = json.dumps({"summary": "s", "evidence_ids": [], "task": "t", "localization": [],
                             "remediation": ["only option 4"]})
    for script in ([good], [wants_menu, wants_menu], [wants_menu, good]):
        rep = write_report(v, E, ScriptedModel(script), repo=RepoIndex.from_dir(MINI_REPO), context=CTX)
        assert rep.remediation == menu_text(v)
        assert rep.verdict is v


def test_code_and_stop_verdicts_get_no_menu_text():
    for label in (VerdictLabel.CODE, VerdictLabel.NO_SIGNIFICANT_FAILURE, VerdictLabel.RENDERING_MISMATCH):
        rep = write_report(verdict(label), E, None, repo=RepoIndex.from_dir(MINI_REPO), context=CTX)
        assert rep.remediation == ""


def test_the_menu_text_lists_options_in_order_with_every_field():
    text = menu_text(verdict(VerdictLabel.VISION_FACTOR, "lighting"))
    titles = [o.title for o in remediation_options(verdict(VerdictLabel.VISION_FACTOR, "lighting"))]
    positions = [text.index(t) for t in titles]
    assert positions == sorted(positions)
    for field in ("prerequisites:", "rough cost:", "repo entry point:", "status:", "caveat:"):
        assert text.count(field) == len(titles)
