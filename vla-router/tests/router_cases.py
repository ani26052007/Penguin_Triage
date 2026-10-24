"""The 11 mocked fault classes (plan.md 6.5) as router incidents.

Built from the Step 3 fixture (``scripts.confusion_matrix.build_cases``), but
with NO hand-written evidence: ``route()`` runs Stage 1 and C_behavior itself
on synthetic logs (``RouterCase.inputs()``). Each case plants its fault where
the real system would log it:

* infra_stale_server: incident-mode traces with a load-dependent delay
  (``Faults.server_delay_max``); clean-mode traces are healthy.
* rendering_mismatch: the eval config's ``render`` differs from the training
  spec's render settings (config contract, no traces needed).
* code_condition_specific: the eval config's instruction string for the glossy
  mug does not match the training templates (config contract; plan 6.4 "wrong
  instruction string for one object").
* combined_code_plus_light: policy inputs left unnormalised by code ``c0``
  (observation contract), on top of the lighting shift.
* action_gap: failing episodes close on the right object and stop at grasp
  (C_behavior).

Every other case logs healthy traces. Traces are keyed on the code under test
(``CodeKeyedTraceSource``), so a fix that removes the fault removes it from
the logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from ladder.backend import MockBackend, MockRule
from router.behavior_synthetic import BehaviorFaults, CodeKeyedBehaviorSource, make_behavior_nominal
from router.replay import ReplayConfig
from router.router import RouterConfig, RouterInputs
from router.schemas import CodeRef, EpisodeResult, Incident, Mode, SceneSpec, TrainingSpec, VerdictLabel
from router.stage1.action_contract import ActionContractCheck
from router.stage1.cfg_contract import ConfigContractCheck, ConfigMapping
from router.stage1.obs_contract import ObservationContractCheck
from router.stage1.sanity import SanityCheck
from router.stage1.staleness import StalenessProbe
from router.stage1.synthetic import CAMERAS, CodeKeyedTraceSource, Faults, make_nominal, make_training_spec
from scripts.confusion_matrix import (FAILURE_RATE, LIGHT, NOMINAL_BAND, POLICY, _REF_OK,
                                      build_cases)

REFERENCE_CHECKPOINT = "ref_v1"          # matches REF_RAW / REF_WRAP in the Step 3 fixture
TRAIN_RENDER = {"preset": "train_like"}
CODE_EVAL_RENDER = CodeRef("c0", {"render": {"preset": "eval_fast"}})
CODE_TRAIN_RENDER = CodeRef("c0", {"render": TRAIN_RENDER})
INSTRUCTION_TEMPLATES = {"pick": "pick up the {object}"}
GOOD_INSTRUCTIONS = {"default": "pick up the mug", "glossy": "pick up the mug"}
CODE_BAD_INSTRUCTION = CodeRef("c0", {"instruction": {"default": "pick up the mug", "glossy": "grab the shiny thing"}})
CODE_GOOD_INSTRUCTION = CodeRef("c1", {"instruction": GOOD_INSTRUCTIONS})

STAGE1_NOMINAL = make_nominal()
BEHAVIOR_NOMINAL = make_behavior_nominal()
CONFIG_MAPPING = ConfigMapping(keys={"render_settings": "render", "instruction_templates": "instruction"})
LOG_REFS = (Mode.CLEAN.value, Mode.INCIDENT.value)


@dataclass(frozen=True)
class RouterCase:
    """``trace_faults`` / ``behavior_faults``: planted log faults by code key;
    ``trace_file_faults`` / ``behavior_file_faults``: (file predicate, faults)
    for codes known only by their files (a real repair agent's patch; needs
    ``workspaces``); ``trace_default`` / ``behavior_default`` for any other
    code (None: that code has no logs)."""
    name: str
    expected: VerdictLabel
    scene: SceneSpec
    rules: tuple[MockRule, ...]
    default_rate: float
    code_ref: CodeRef = CodeRef("c0")
    render_settings: Optional[dict] = None
    instruction_templates: Optional[dict] = None
    nominal_scene: Optional[SceneSpec] = None
    expected_factor: Optional[str] = None
    expected_second_cause: Optional[bool] = None
    expected_by_elimination: Optional[bool] = None
    expected_location: Optional[str] = None
    failing_seeds: tuple[int, ...] = field(default_factory=lambda: tuple(range(10)))
    trace_faults: Mapping[str, Faults] = field(default_factory=dict)
    trace_default: Optional[Faults] = Faults()
    behavior_faults: Mapping[str, BehaviorFaults] = field(default_factory=dict)
    behavior_default: Optional[BehaviorFaults] = BehaviorFaults()
    trace_file_faults: tuple = ()
    behavior_file_faults: tuple = ()

    def backend(self, salt: str, workspaces=None) -> MockBackend:
        return MockBackend(self.rules, self.default_rate, salt=salt, workspaces=workspaces)

    def quota_backend(self, workspaces=None) -> "QuotaBackend":
        return QuotaBackend(self.rules, self.default_rate, workspaces=workspaces)

    def incident(self, log_modes: tuple[str, ...] = LOG_REFS) -> Incident:
        """``log_modes``: the inference modes whose logs exist (``()``: no logs at all)."""
        spec = TrainingSpec("mock_policy", render_settings=self.render_settings,
                            instruction_templates=self.instruction_templates, **ACTION_SPEC)
        return Incident(self.name, self.scene, spec, self.code_ref, POLICY, self.failing_seeds, Mode.INCIDENT,
                        log_refs=log_refs(self.name, log_modes), nominal_scene=self.nominal_scene)

    def inputs(self, *, stage1: bool = True, behavior: bool = True, workspaces=None) -> RouterInputs:
        """The full Stage 1 check set and the C_behavior source on this case's
        synthetic logs; ``stage1=False`` / ``behavior=False`` leave them out."""
        src = CodeKeyedTraceSource(self.trace_faults, self.trace_default, file_faults=self.trace_file_faults,
                                   workspaces=workspaces if self.trace_file_faults else None)
        checks = (ConfigContractCheck(CONFIG_MAPPING), ObservationContractCheck(STAGE1_NOMINAL, src, CAMERAS),
                  ActionContractCheck(STAGE1_NOMINAL, src), StalenessProbe(STAGE1_NOMINAL, src),
                  SanityCheck(STAGE1_NOMINAL, src)) if stage1 else ()
        if not behavior:
            return RouterInputs(checks)
        bsrc = CodeKeyedBehaviorSource(self.behavior_faults, self.behavior_default,
                                       file_faults=self.behavior_file_faults,
                                       workspaces=workspaces if self.behavior_file_faults else None)
        return RouterInputs(checks, bsrc, BEHAVIOR_NOMINAL)


# The action layout and limits the synthetic logs follow. Without them the action contract checks
# none of dimensions / range / gripper sign (a partial skip of contract_violation), and row 4 could
# never be ruled out.
ACTION_SPEC = dict(norm_stats={"action": make_training_spec().norm_stats["action"]},
                   action_layout=make_training_spec().action_layout)


def log_refs(name: str, modes: tuple[str, ...] = LOG_REFS) -> dict[str, str]:
    return {m: f"{name}/{m}" for m in modes}


def healthy_inputs() -> RouterInputs:
    """Stage 1 and C_behavior over healthy logs for any code: for incidents
    whose fault is only in the replays (the incident needs ``log_refs``)."""
    return RouterCase("healthy", VerdictLabel.UNDETERMINED, SceneSpec("x"), (), 0.0).inputs()


def router_cases() -> list[RouterCase]:
    out = []
    for c in build_cases():
        kw = dict(name=c.name, expected=c.expected, scene=c.scene, rules=c.rules, default_rate=c.default_rate,
                  code_ref=c.code_ref, nominal_scene=c.stage2_scene,
                  expected_factor=c.expected_factor, expected_second_cause=c.expected_second_cause,
                  expected_by_elimination=c.expected_by_elimination)
        if c.name == "rendering_mismatch":
            kw.update(code_ref=CODE_EVAL_RENDER, render_settings=TRAIN_RENDER, expected_location="eval_config.render",
                      rules=_REF_OK + (MockRule(0.9, code=CODE_TRAIN_RENDER), MockRule(0.15, factors_present=(LIGHT,))))
        elif c.name == "infra_stale_server":
            kw.update(trace_default=Faults(server_delay_max=6))
        elif c.name == "code_condition_specific":
            kw.update(code_ref=CODE_BAD_INSTRUCTION, instruction_templates=INSTRUCTION_TEMPLATES,
                      expected_location="eval_config.instruction")
        elif c.name == "combined_code_plus_light":
            kw.update(trace_faults={c.code_ref.key(): Faults(unnormalized=True)}, expected_location="obs.normalization")
        elif c.name == "action_gap":
            kw.update(behavior_default=BehaviorFaults(fail_phase="grasp"))
        out.append(RouterCase(**kw))
    return out


class QuotaBackend(MockBackend):
    """MockBackend without sampling noise: seed ``s`` succeeds iff
    ``s % 10 < round(rate * 10)``, so every block of 10 consecutive seeds has
    exactly that many successes. Used to check routing logic exactly; the
    salted ``MockBackend`` is used to measure noise."""

    def run(self, scene, code_ref, actor, seeds, mode):
        mode = Mode(mode)
        rate = self.rate_for(scene, code_ref, actor, mode)
        cfg = self.config_key(scene, code_ref, actor, mode)
        quota = round(rate * 10)
        out = []
        for seed in seeds:
            self.calls[cfg] += 1
            out.append(EpisodeResult(int(seed), int(seed) % 10 < quota, {"backend": "quota", "rate": rate}))
        return out


def default_config(**replay_kw) -> RouterConfig:
    return RouterConfig(ReplayConfig(FAILURE_RATE, NOMINAL_BAND, **replay_kw), REFERENCE_CHECKPOINT)
