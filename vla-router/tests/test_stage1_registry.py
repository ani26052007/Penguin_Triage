"""Stage 1 registry, and Stage 1 evidence flowing into the router (plan.md 3.2.4, 3.2.8)."""

import pytest

from ladder.backend import MockBackend, MockRule
from router import vocab as V
from router.decide import decide
from router.replay import ReplayConfig
from router.router import RouterConfig, RouterInputs, route
from router.schemas import ActionPipeline, ActorKind, ActorRef, Factor, SceneSpec, VerdictLabel
from router.stage1.action_contract import ActionContractCheck
from router.stage1.base import Check
from router.stage1.cfg_contract import ConfigContractCheck
from router.stage1.obs_contract import ObservationContractCheck
from router.prereq.trace import InMemoryTraceSource
from router.schemas import Mode
from router.stage1.registry import EVIDENCE_SOURCES, NOT_CONFIGURED, run_stage1
from router.stage1.sanity import SanityCheck
from router.stage1.staleness import StalenessProbe
from router.stage1.synthetic import (CAMERAS, Faults, make_config_mapping, make_eval_config, make_incident,
                                     make_nominal, make_source)

NOMINAL = make_nominal()


def all_checks(src, mapping=None):
    return [ConfigContractCheck(mapping or make_config_mapping()),
            ObservationContractCheck(NOMINAL, src, CAMERAS),
            ActionContractCheck(NOMINAL, src),
            StalenessProbe(NOMINAL, src),
            SanityCheck(NOMINAL, src)]


def test_every_registered_source_has_a_check_and_a_clean_run_is_empty():
    checks = all_checks(make_source())
    assert tuple(c.source for c in checks) == EVIDENCE_SOURCES
    res = run_stage1(make_incident(), checks)
    assert res.evidence == () and res.skipped == () and res.sources_run == EVIDENCE_SOURCES


@pytest.mark.parametrize("faults, source, prefix", [
    (Faults(swap_rb=True), "contract_obs", "s1.obs."),
    (Faults(gripper_inverted=True), "contract_action", "s1.action."),
    (Faults(extra_delay=3), "staleness", "s1.staleness."),
    (Faults(nan_state=True), "sanity", "s1.sanity."),
])
def test_one_planted_trace_fault_lights_exactly_one_source(faults, source, prefix):
    res = run_stage1(make_incident(), all_checks(make_source(faults)))
    assert {e.source for e in res.evidence} == {source}
    assert all(e.id.startswith(prefix) for e in res.evidence)


def test_planted_config_fault_lights_only_the_config_source():
    cfg = make_eval_config()
    cfg["policy"]["replan"] = 16
    res = run_stage1(make_incident(eval_config=cfg), all_checks(make_source()))
    assert [(e.source, e.id) for e in res.evidence] == [("contract_cfg", "s1.cfg.replan_steps")]


def test_missing_traces_skip_the_trace_checks_but_not_the_config_check():
    res = run_stage1(make_incident(log_refs={}), all_checks(make_source()))
    assert res.evidence == () and [s for s, _ in res.skipped] == list(EVIDENCE_SOURCES[1:])
    assert res.sources_run == ("contract_cfg",)           # "nothing found" is not claimed for the skipped ones


def test_sources_run_lists_only_the_checks_that_ran():
    """Tier 5 (Fix F1): it used to list all five sources whatever was passed."""
    res = run_stage1(make_incident(), [ConfigContractCheck(make_config_mapping()), SanityCheck(NOMINAL, make_source())])
    assert res.sources_run == ("contract_cfg", "sanity")
    assert res.skipped == tuple((s, NOT_CONFIGURED) for s in ("contract_obs", "contract_action", "staleness"))
    empty = run_stage1(make_incident(), [])
    assert empty.sources_run == () and [s for s, _ in empty.skipped] == list(EVIDENCE_SOURCES)


def test_staleness_partial_skip_is_recorded_and_the_source_still_counts_as_run():
    src = InMemoryTraceSource({"clean": make_source().load(make_incident(), Mode.CLEAN)})
    res = run_stage1(make_incident(), all_checks(src))
    assert [s for s, _ in res.partial] == ["staleness"] and res.partial[0][1].signals == {"staleness_variable"}
    assert "staleness" in res.sources_run and res.skipped == ()


def test_registry_rejects_bad_check_sets():
    class Rogue(Check):
        source = "not_registered"
        def run(self, incident): return []
    class Liar(Check):
        source = "sanity"
        def run(self, incident):
            return ConfigContractCheck(make_config_mapping()).run(make_incident(eval_config=bad_cfg))
    bad_cfg = make_eval_config()
    bad_cfg["policy"]["replan"] = 16
    with pytest.raises(ValueError, match="unregistered"):
        run_stage1(make_incident(), [Rogue()])
    with pytest.raises(ValueError, match="share an evidence source"):
        run_stage1(make_incident(), [SanityCheck(NOMINAL, make_source()), SanityCheck(NOMINAL, make_source())])
    with pytest.raises(ValueError, match="another source"):
        run_stage1(make_incident(), [Liar()])


def test_evidence_ids_are_unique_across_a_run():
    res = run_stage1(make_incident(), all_checks(make_source(Faults(swap_rb=True, extra_delay=3, nan_state=True,
                                                                    gripper_inverted=True))))
    ids = [e.id for e in res.evidence]
    assert len(ids) == len(set(ids)) >= 4


# ---- into the router ------------------------------------------------------ #

LOW, OK = 0.15, 0.9
POLICY = ActorRef(ActorKind.POLICY, "ckpt_v1")
REF_RAW = ActorRef(ActorKind.REFERENCE, "ref_v1", ActionPipeline.RAW)
REF_WRAP = ActorRef(ActorKind.REFERENCE, "ref_v1", ActionPipeline.WRAPPER)
MAT = "material(mug)"
CONFIG = RouterConfig(ReplayConfig(failure_rate=0.15, nominal_band=(0.70, 0.95)), reference_checkpoint="ref_v1")


def _backend(salt):
    # the failure shows only with the glossy-mug factor present; reference controller fine either way
    return MockBackend((MockRule(0.97, actor=REF_RAW), MockRule(0.97, actor=REF_WRAP),
                        MockRule(LOW, factors_present=(MAT,))), OK, salt=salt)


def test_instruction_template_bug_found_by_stage1_routes_to_code_at_that_location():
    scene = SceneSpec("pick_mug", (Factor(MAT, "glossy_02"),))
    cfg = make_eval_config()
    cfg["env"]["instruction"] = "grab the mug"
    incident = make_incident(eval_config=cfg, scene=scene)
    stage1 = run_stage1(incident, all_checks(make_source(code_key=incident.code_ref.key())))
    assert [e.id for e in stage1.evidence] == ["s1.cfg.instruction_templates"]

    # without Stage 1 the replays alone point at the factor; with the checks unconfigured that is row 10
    base = route(incident, _backend("a"), CONFIG)
    assert base.verdict.row == 10 and "row 4 undecided: not checked: contract_violation" in base.verdict.notes

    res = route(incident, _backend("a"), CONFIG, RouterInputs(tuple(all_checks(make_source(code_key=incident.code_ref.key())))))
    assert [e.id for e in res.stage1_evidence] == ["s1.cfg.instruction_templates"]
    assert res.verdict.label is VerdictLabel.CODE
    assert res.verdict.location == "eval_config.env.instruction"
    assert "s1.cfg.instruction_templates" in res.verdict.evidence_ids


def test_stage1_evidence_is_accepted_by_decide_without_changes_to_it():
    stage1 = run_stage1(make_incident(), all_checks(make_source(Faults(extra_delay=3))))
    assert stage1.evidence and all(e.signal in V.__dict__.values() for e in stage1.evidence)
    decide(stage1.evidence, [])      # evidence alone is accepted; the replays decide the rest
