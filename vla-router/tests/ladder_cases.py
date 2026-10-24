"""Fixtures for the Step 7 loop tests: mocked incidents with a *fixed* system.

A fix is a new ``CodeRef``; the mock rules are keyed on it, so "the patch worked" is a rule row, not
a code path. The evidence is not a fixture: ``route()`` runs Stage 1 and C_behavior on synthetic
logs keyed on the code under test (``RouterCase.trace_faults``), so a fix removes the planted fault
from the logs and the checks stop finding it (Fix F1). ``ScriptedRepairer`` returns a prepared
``CodeRef`` per call and counts calls.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

from ladder.backend import MockRule, RolloutBackend
from ladder.loop import IncidentLoop, LoopConfig, StubDataLadder
from ladder.repair import RepairOutcome, RepairRequest
from ladder.rung0 import Rung0Result
from ladder.rung1 import Rung1Config, staged_plan_for
from router.schemas import CodeRef, EpisodeResult
from router.stage1.synthetic import Faults
from scripts.confusion_matrix import FAILURE_RATE, LIGHT, MAT, NOMINAL_BAND, _REF_OK
from tests.router_cases import CODE_GOOD_INSTRUCTION, RouterCase, default_config, router_cases

CASES = {c.name: c for c in router_cases()}
PATCHED = CodeRef("c1")
BAD_PATCH = CodeRef("c_bad")
PATCHED_INSTRUCTION = CODE_GOOD_INSTRUCTION      # c1 with the instruction string fixed
UNNORMALIZED = Faults(unnormalized=True)          # the combined case's code bug, as the logs show it


def rung1_config(seeds: int = 50, agreement: float = 1.0, min_agreement: float = 0.9) -> Rung1Config:
    return Rung1Config(staged_plan_for(seeds), FAILURE_RATE, NOMINAL_BAND, agreement, min_agreement)


def loop_config(**kw) -> LoopConfig:
    return LoopConfig(default_config(), rung1_config(), **kw)


def combined_case(severe: bool = False, patched_nominal: float = 0.9, extra_codes: Sequence[CodeRef] = (),
                  fault_persists: bool = False) -> RouterCase:
    """Code bug (unnormalised policy inputs: observation contract) + lighting shift, with the patched
    code ``c1``:

    original ``c0``: nominal 0.35 (degraded; ``severe``: 0.10, clearly failed -> Branch G, so the
                     second-cause flag is never raised), lit scene 0.10.
    patched  ``c1``: nominal 0.9 (the code bug is gone), lit scene 0.35 (lighting alone still hurts,
                     a bit less than with the bug on top).
    ``patched_nominal`` lowers the patched nominal rate (an ambiguous Rung 1); ``extra_codes`` are further
    fixes (``c2``...) that restore the nominal scene fully.

    Logs: ``c0`` and ``BAD_PATCH`` log unnormalised inputs; ``c1`` and the extra codes log healthy
    ones, unless ``fault_persists`` (a fix that only masks the symptom: the logs keep the fault).
    Any other code has no logs.
    """
    base = CASES["combined_code_plus_light"]
    ref, rest = base.rules[:len(_REF_OK)], base.rules[len(_REF_OK):]
    more = tuple(m for c in extra_codes for m in (MockRule(0.35, code=c, factors_present=(LIGHT,)), MockRule(0.9, code=c)))
    patched = (MockRule(0.35, code=PATCHED, factors_present=(LIGHT,)), MockRule(patched_nominal, code=PATCHED)) + more
    fixed = UNNORMALIZED if fault_persists else Faults()
    faults = {base.code_ref.key(): UNNORMALIZED, BAD_PATCH.key(): UNNORMALIZED,
              **{c.key(): fixed for c in (PATCHED, *extra_codes)}}
    return replace(base, name="combined_severe" if severe else base.name, rules=ref + patched + rest,
                   default_rate=0.10 if severe else base.default_rate, trace_faults=faults, trace_default=None)


def condition_specific_case(patched_rate: float) -> RouterCase:
    """Condition-specific code bug (wrong instruction string for the glossy mug, found by the config
    contract), Branch C. The fix is ``PATCHED_INSTRUCTION``. After it the glossy scene is only partly
    recovered (``patched_rate``) while the nominal scene is fine, so the router re-run names the
    material as a vision factor."""
    base = CASES["code_condition_specific"]
    ref, rest = base.rules[:len(_REF_OK)], base.rules[len(_REF_OK):]
    return replace(base, rules=ref + (MockRule(patched_rate, code=PATCHED_INSTRUCTION, factors_present=(MAT,)),) + rest)


class FnBackend(RolloutBackend):
    """Success decided by ``fn(scene_key, code_key, seed) -> bool``. Counts simulated episodes per
    (scene, code, seed) so tests can check that no episode is run twice."""

    def __init__(self, fn: Callable[[str, str, int], bool]):
        self.fn, self.runs = fn, Counter()

    def run(self, scene, code_ref, actor, seeds, mode):
        out = []
        for s in seeds:
            self.runs[(scene.key(), code_ref.key(), actor.key(), int(s))] += 1
            out.append(EpisodeResult(int(s), bool(self.fn(scene.key(), code_ref.key(), int(s)))))
        return out


def rendering_case() -> RouterCase:
    return CASES["rendering_mismatch"]


@dataclass
class ScriptedRepairer:
    """Returns ``codes[i]`` on the i-th call (the last one repeats). Records every request."""
    codes: Sequence[CodeRef]
    requests: list[RepairRequest] = field(default_factory=list)

    @property
    def calls(self) -> int:
        return len(self.requests)

    def repair(self, request: RepairRequest) -> RepairOutcome:
        self.requests.append(request)
        code = self.codes[min(len(self.requests) - 1, len(self.codes) - 1)]
        diff = f"--- a/x\n+++ b/x\n+{code.code_hash}\n"
        return RepairOutcome(code, diff=diff, notes="scripted", code_diff=diff)


def passing_rung0(req: RepairRequest, out: RepairOutcome) -> Rung0Result:
    return Rung0Result(True, ())


def make_loop(case: RouterCase, repairer, *, rung0=passing_rung0, rung2=None, ladder: Optional[StubDataLadder] = None,
              cfg: Optional[LoopConfig] = None, backend=None, workspaces=None) -> tuple[IncidentLoop, StubDataLadder]:
    """``workspaces``: the backend's file rules and the logs' file faults read the code through it."""
    ladder = ladder or StubDataLadder()
    loop = IncidentLoop(backend or case.quota_backend(workspaces), cfg or loop_config(), repairer=repairer,
                        inputs=case.inputs(workspaces=workspaces), data_ladder=ladder, rung0=rung0, rung2=rung2)
    return loop, ladder
