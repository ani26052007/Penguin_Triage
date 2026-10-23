"""The 11 mocked fault classes (plan.md 6.5) as router incidents.

Built from the Step 3 fixture (``scripts.confusion_matrix.build_cases``) with
three differences, because here the router runs the replays itself:

* rendering_mismatch: the render settings live in the training spec and in the
  eval config under ``render``, so the router can build the C_render code ref.
* action_gap: the Step 3 ``stage2_scene`` becomes ``Incident.nominal_scene``.
* code_condition_specific: no evidence. Its only routing signal is a Stage 1
  contract violation, and Stage 1 is out of scope for this step, so this class
  is an expected failure (user decision) until Stage 1 exists.

Every other class gets the Step 3 hand-written evidence as ``extra_evidence``
(the stand-in for Stage 1 and C_behavior records, user decision).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ladder.backend import MockBackend, MockRule
from router.replay import ReplayConfig
from router.router import RouterConfig
from router.schemas import (CodeRef, EpisodeResult, Evidence, Incident, Mode, SceneSpec, TrainingSpec,
                            VerdictLabel)
from scripts.confusion_matrix import (FAILURE_RATE, LIGHT, NOMINAL_BAND, POLICY, _REF_OK,
                                      build_cases)

REFERENCE_CHECKPOINT = "ref_v1"          # matches REF_RAW / REF_WRAP in the Step 3 fixture
TRAIN_RENDER = {"preset": "train_like"}
CODE_EVAL_RENDER = CodeRef("c0", {"render": {"preset": "eval_fast"}})
CODE_TRAIN_RENDER = CodeRef("c0", {"render": TRAIN_RENDER})

STAGE1_ONLY = {"code_condition_specific"}


@dataclass(frozen=True)
class RouterCase:
    name: str
    expected: VerdictLabel
    scene: SceneSpec
    rules: tuple[MockRule, ...]
    default_rate: float
    evidence: tuple[Evidence, ...] = ()
    code_ref: CodeRef = CodeRef("c0")
    render_settings: Optional[dict] = None
    nominal_scene: Optional[SceneSpec] = None
    expected_factor: Optional[str] = None
    expected_second_cause: Optional[bool] = None
    expected_by_elimination: Optional[bool] = None
    failing_seeds: tuple[int, ...] = field(default_factory=lambda: tuple(range(10)))

    def backend(self, salt: str) -> MockBackend:
        return MockBackend(self.rules, self.default_rate, salt=salt)

    def quota_backend(self) -> "QuotaBackend":
        return QuotaBackend(self.rules, self.default_rate)

    def incident(self) -> Incident:
        return Incident(self.name, self.scene, TrainingSpec("mock_policy", render_settings=self.render_settings),
                        self.code_ref, POLICY, self.failing_seeds, Mode.INCIDENT,
                        nominal_scene=self.nominal_scene)


def router_cases() -> list[RouterCase]:
    out = []
    for c in build_cases():
        kw = dict(name=c.name, expected=c.expected, scene=c.scene, rules=c.rules, default_rate=c.default_rate,
                  evidence=c.evidence, code_ref=c.code_ref, nominal_scene=c.stage2_scene,
                  expected_factor=c.expected_factor, expected_second_cause=c.expected_second_cause,
                  expected_by_elimination=c.expected_by_elimination)
        if c.name == "rendering_mismatch":
            kw.update(code_ref=CODE_EVAL_RENDER, render_settings=TRAIN_RENDER,
                      rules=_REF_OK + (MockRule(0.9, code=CODE_TRAIN_RENDER), MockRule(0.15, factors_present=(LIGHT,))))
        if c.name in STAGE1_ONLY:
            kw.update(evidence=())
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
