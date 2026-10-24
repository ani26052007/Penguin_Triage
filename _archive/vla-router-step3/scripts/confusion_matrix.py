"""Confusion matrix of ``decide`` over the 11 mocked fault classes (plan.md 6.5).

    python -m scripts.confusion_matrix [--seeds 30] [--repeats 10]

For each fault class a ``MockBackend`` is configured with that fault's success
rates; the script runs the replays against it, classifies each with
``ladder.stats``, hands the table plus the case's evidence to ``decide`` and
tallies the verdict labels. Each repeat uses a different mock salt, so the
matrix shows how sampling noise at ``--seeds`` episodes per replay moves the
verdicts, including how often the router abstains.

Test fixture, not the router. Two things are stand-ins until later steps:
  * ``collect_replays`` runs a fixed superset of replays (no Stage 0 early
    stop, no extend-seeds loop, no cache). The real runners are step 6.
  * Each case's evidence list is written by hand. The real Stage 1 checks and
    behavior analysis are step 6, so the evidence here is what they would
    plausibly emit for that fault, not a measurement.

Unless a case says otherwise every replay runs in clean mode; the infra case
fails only in INCIDENT mode, which no replay uses, as in a real stale-server
fault.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ladder.backend import MockBackend, MockRule
from ladder.stats import replay_outcome
from router import vocab as V
from router.decide import decide
from router.schemas import (ActionPipeline, ActorKind, ActorRef, CodeRef, Evidence, Factor, Mode,
                            ReplayResult, SceneSpec, Verdict, VerdictLabel)

FAILURE_RATE = 0.15            # failure anchor (failure-map shifted rate)
NOMINAL_BAND = (0.70, 0.95)    # nominal band; both are per-policy calibration values in the real router

POLICY = ActorRef(ActorKind.POLICY, "ckpt_v1")
REF_RAW = ActorRef(ActorKind.REFERENCE, "ref_v1", ActionPipeline.RAW)
REF_WRAP = ActorRef(ActorKind.REFERENCE, "ref_v1", ActionPipeline.WRAPPER)

LIGHT, MAT = "lighting", "material(mug)"
F_LIGHT, F_MAT = Factor(LIGHT, "dark_01"), Factor(MAT, "glossy_02")
NOMINAL = SceneSpec("pick_mug")

CODE = CodeRef("c0")
CODE_EVAL = CodeRef("c0", {"render": "eval_fast"})
CODE_TRAIN = CodeRef("c0", {"render": "train_like"})

_REF_OK = (MockRule(0.97, actor=REF_RAW), MockRule(0.97, actor=REF_WRAP))


@dataclass(frozen=True)
class FaultCase:
    name: str
    expected: VerdictLabel
    scene: SceneSpec                                   # the incident (shifted) scene
    rules: tuple[MockRule, ...]
    default_rate: float
    evidence: tuple[Evidence, ...] = ()
    code_ref: CodeRef = CODE
    render_fixed_code: Optional[CodeRef] = None        # set => C_render replay is run
    stage2_scene: Optional[SceneSpec] = None           # default: the incident scene's nominal
    expected_factor: Optional[str] = None
    expected_second_cause: Optional[bool] = None       # None = not checked
    expected_by_elimination: Optional[bool] = None
    note: str = ""

    def make_backend(self, salt: str) -> MockBackend:
        return MockBackend(self.rules, self.default_rate, salt=salt)


def _e(i: str, signal: str, strength: float = 0.8, location: Optional[str] = None) -> Evidence:
    return Evidence(i, "fixture", signal, strength, location, f"hand-written for the mock: {signal}")


def build_cases() -> list[FaultCase]:
    one, two = SceneSpec("pick_mug", (F_MAT,)), SceneSpec("pick_mug", (F_LIGHT, F_MAT))
    lit = SceneSpec("pick_mug", (F_LIGHT,))
    gap_shifted = SceneSpec("pick_unseen_geom", (Factor(LIGHT, "dim_01"),))
    OK, LOW = 0.9, 0.15
    return [
        FaultCase("no_fault", VerdictLabel.NO_SIGNIFICANT_FAILURE, NOMINAL, _REF_OK, OK),
        FaultCase("vision_glossy", VerdictLabel.VISION_FACTOR, one,
                  _REF_OK + (MockRule(LOW, factors_present=(MAT,)),), OK, expected_factor=MAT),
        FaultCase("vision_interacting", VerdictLabel.VISION_INTERACTING, two,
                  _REF_OK + (MockRule(0.10, factors_present=(LIGHT, MAT)),
                             MockRule(0.35, factors_present=(LIGHT,)), MockRule(0.35, factors_present=(MAT,))), OK,
                  note="either factor alone is only a partial restore (additive reading of row 7)"),
        FaultCase("rendering_mismatch", VerdictLabel.RENDERING_MISMATCH, lit,
                  _REF_OK + (MockRule(OK, code=CODE_TRAIN), MockRule(LOW, factors_present=(LIGHT,))), OK,
                  evidence=(_e("ev_render", V.SIG_RENDER_SETTINGS_DIFFER, 0.9, "render.samples"),),
                  code_ref=CODE_EVAL, render_fixed_code=CODE_TRAIN, expected_second_cause=False),
        FaultCase("sim_asset", VerdictLabel.SIM_ASSET, one,
                  (MockRule(0.10, actor=REF_RAW, factors_present=(MAT,)),) + _REF_OK
                  + (MockRule(0.10, factors_present=(MAT,)),), OK,
                  note="broken collision mesh: the reference controller fails on the shifted scene (C1)"),
        FaultCase("infra_stale_server", VerdictLabel.INFRA, lit,
                  _REF_OK + (MockRule(LOW, mode=Mode.INCIDENT),), OK,
                  evidence=(_e("ev_stale", V.SIG_STALENESS_VARIABLE, 0.85),)),
        FaultCase("action_gap", VerdictLabel.ACTION_GAP, gap_shifted,
                  _REF_OK + (MockRule(LOW, scene=gap_shifted), MockRule(LOW, scene=gap_shifted.restore_all())), OK,
                  evidence=(_e("ev_target", V.SIG_IMPLIED_TARGET_OK, 0.8),
                            _e("ev_phase", V.failure_phase("grasp"), 0.9)),
                  stage2_scene=SceneSpec("pick_mug"),
                  note="the unseen geometry is an undeclared change, so no declared factor restores it"),
        FaultCase("code_global", VerdictLabel.CODE, lit,
                  (MockRule(0.97, actor=REF_RAW), MockRule(0.10, actor=REF_WRAP)), 0.10,
                  expected_by_elimination=False, note="wrapper bug: reference fine raw, broken through the wrapper (G3)"),
        FaultCase("code_condition_specific", VerdictLabel.CODE, one,
                  _REF_OK + (MockRule(LOW, factors_present=(MAT,)),), OK,
                  evidence=(_e("ev_contract", V.SIG_CONTRACT_VIOLATION, 0.8, "instruction_template"),),
                  expected_by_elimination=False, expected_second_cause=False),
        FaultCase("code_partial_offset", VerdictLabel.CODE, NOMINAL, _REF_OK, 0.35,
                  expected_by_elimination=True, expected_second_cause=False,
                  note="mild normalization offset, no Stage 1 hit: code by elimination, flagged degraded"),
        FaultCase("combined_code_plus_light", VerdictLabel.CODE, lit,
                  _REF_OK + (MockRule(0.10, factors_present=(LIGHT,)),), 0.35,
                  evidence=(_e("ev_contract", V.SIG_CONTRACT_VIOLATION, 0.7, "norm_stats"),),
                  expected_by_elimination=False, expected_second_cause=True),
    ]


# --------------------------------------------------------------------------- #
# replay fixture (NOT the step-6 runner)
# --------------------------------------------------------------------------- #

def _replay(backend: MockBackend, rid: str, scene: SceneSpec, code: CodeRef, actor: ActorRef,
            seeds: Sequence[int], mode: Mode = Mode.CLEAN) -> ReplayResult:
    eps = backend.run(scene, code, actor, seeds, mode)
    outcome, prop = replay_outcome(sum(e.success for e in eps), len(eps), FAILURE_RATE, NOMINAL_BAND)
    return ReplayResult(rid, scene, code, actor, mode, tuple(eps), outcome, prop.lo, prop.hi)


def collect_replays(case: FaultCase, backend: MockBackend, n_seeds: int) -> list[ReplayResult]:
    seeds = list(range(n_seeds))
    shifted, code = case.scene, case.code_ref
    nominal = case.stage2_scene or SceneSpec(shifted.nominal)
    out = [
        _replay(backend, V.R_STAGE0, shifted, code, POLICY, seeds),
        _replay(backend, V.R_STAGE2_NOMINAL, nominal, code, POLICY, seeds),
        _replay(backend, V.R_G2, nominal, code, REF_RAW, seeds),
        _replay(backend, V.R_G3, nominal, code, REF_WRAP, seeds),
        _replay(backend, V.R_C1, shifted, code, REF_RAW, seeds),
    ]
    for f in shifted.factor_names:
        out.append(_replay(backend, V.restore_id(f), shifted.restore(f), code, POLICY, seeds))
    if len(shifted.factors) >= 2:
        out.append(_replay(backend, V.R_C_ALL, shifted.restore_all(), code, POLICY, seeds))
    if case.render_fixed_code is not None:
        out.append(_replay(backend, V.R_C_RENDER, shifted, case.render_fixed_code, POLICY, seeds))
    return out


# --------------------------------------------------------------------------- #
# matrix
# --------------------------------------------------------------------------- #

@dataclass
class CaseResult:
    case: FaultCase
    verdicts: list[Verdict] = field(default_factory=list)

    def labels(self) -> Counter:
        return Counter(v.label for v in self.verdicts)

    @property
    def correct(self) -> int:
        return sum(v.label is self.case.expected for v in self.verdicts)

    def _count(self, pred) -> tuple[int, int]:
        """(flag mismatches, rows checked) among the correct-label verdicts."""
        good = [v for v in self.verdicts if v.label is self.case.expected]
        return sum(not pred(v) for v in good), len(good)

    def flag_mismatches(self) -> list[str]:
        c, out = self.case, []
        checks = [("likely_second_cause", c.expected_second_cause, lambda v: v.likely_second_cause),
                  ("by_elimination", c.expected_by_elimination, lambda v: v.by_elimination),
                  ("factor", c.expected_factor, lambda v: v.factor)]
        for name, want, get in checks:
            if want is None:
                continue
            bad = sum(get(v) != want for v in self.verdicts if v.label is c.expected)
            if bad:
                out.append(f"{name}!={want!r} in {bad}/{self.correct}")
        return out


def run_matrix(cases: Sequence[FaultCase], n_seeds: int = 30, repeats: int = 10) -> list[CaseResult]:
    results = []
    for case in cases:
        res = CaseResult(case)
        for r in range(repeats):
            backend = case.make_backend(f"{case.name}:{r}")
            res.verdicts.append(decide(case.evidence, collect_replays(case, backend, n_seeds)))
        results.append(res)
    return results


def summarize(results: Sequence[CaseResult]) -> dict:
    total = sum(len(r.verdicts) for r in results)
    correct = sum(r.correct for r in results)
    undet = sum(r.labels()[VerdictLabel.UNDETERMINED] for r in results)
    committed = total - undet
    committed_correct = sum(r.correct for r in results if r.case.expected is not VerdictLabel.UNDETERMINED)
    return {"total": total, "correct": correct, "undetermined": undet,
            "accuracy": correct / total if total else float("nan"),
            "undetermined_rate": undet / total if total else float("nan"),
            "accuracy_when_committing": committed_correct / committed if committed else float("nan")}


def format_matrix(results: Sequence[CaseResult]) -> str:
    used = [l for l in VerdictLabel if any(r.labels()[l] for r in results)]
    for r in results:                       # keep the expected column even if it never fired
        if r.case.expected not in used:
            used.append(r.case.expected)
    used.sort(key=list(VerdictLabel).index)
    name_w = max(len(r.case.name) for r in results)
    col_w = [max(len(l.value), 4) for l in used]
    head = f"{'fault class':<{name_w}}  {'expected':<22}" + "".join(f"  {l.value:>{w}}" for l, w in zip(used, col_w))
    lines = [head, "-" * len(head)]
    for r in results:
        cells = []
        for l, w in zip(used, col_w):
            n = r.labels()[l]
            mark = "*" if l is r.case.expected else " "
            cells.append(f"  {(str(n) if n else '.') + mark:>{w}}")
        lines.append(f"{r.case.name:<{name_w}}  {r.case.expected.value:<22}" + "".join(cells))
    s = summarize(results)
    lines += ["", "(* = expected label; counts are over repeats)",
              f"verdicts: {s['total']}   correct: {s['correct']} ({s['accuracy']:.1%})",
              f"undetermined rate: {s['undetermined_rate']:.1%}   accuracy when committing: {s['accuracy_when_committing']:.1%}"]
    notes = [f"  {r.case.name}: {', '.join(m)}" for r in results if (m := r.flag_mismatches())]
    if notes:
        lines += ["", "flag / factor mismatches among correct labels:"] + notes
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", type=int, default=30, help="episodes per replay (default 30)")
    ap.add_argument("--repeats", type=int, default=10, help="mock salts per fault class (default 10)")
    args = ap.parse_args(argv)
    results = run_matrix(build_cases(), args.seeds, args.repeats)
    print(format_matrix(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
