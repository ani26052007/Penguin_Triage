"""Rung 1: staged simulator validation of a patch (plan.md Section 4).

    run_rung1(backend, incident, after, branch, cfg) -> Rung1Result

Which scene is validated is keyed on the router's BRANCH, not on its flags
(flags miss undetected second causes, and a patch that cannot fix the second
cause would then loop back to the repair agent forever):

    Branch G (incl. degraded)         target = the nominal scene
    Branch C with a code/config fix   target = the incident (shifted) scene

Both scenes always run, before (the code under repair) and after (the patch).
The target decides pass / fail through ``ladder.stats.staged_decision`` (Stage A,
then Stage B on the same seeds, Bonferroni-split alpha). The other scene is a
regression check:

    pass                target restored, other scene restored (or not broken before)
    pass_with_residual  target restored, other scene not restored but no worse
                        (improved or unchanged) -> router re-run, never back to
                        the repair agent: the residual is not what the patch touches
    fail                target still at the broken baseline, or ANY regression
                        -> back to the repair agent
    ambiguous           target still ambiguous after Stage B -> Rung 2

"Regression" on the other scene = clearly worse than its own baseline by the
shared ``ladder.stats.compare`` test, or restored before and clearly below the
band after. Runs are CLEAN mode, so server batching cannot contaminate them
(plan: "inference settings that do not introduce stale actions").

CLI (registered as a hagent custom command, see ``ladder/bin/run_rung1``):

    run_rung1 --seeds 50 [--config FILE]     # FILE or $RUNG1_CONFIG, JSON, see ``load_config``

No simulator is assumed: the backend is a ``RolloutBackend``; the CLI builds a
``MockBackend`` from JSON rules, and ``IsaacBackend`` raises until it exists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence

from ladder.backend import IsaacBackend, MockBackend, MockRule, RolloutBackend
from ladder.workspace import FileContains, WorkspaceError, WorkspaceManager, code_ref_of
from ladder.stats import (Comparison, Proportion, StageDecision, StagedPlan, classify_outcome, compare,
                          staged_decision, wilson_interval)
from router.prereq.rollout_cache import CachedBackend, RolloutCache
from router.replay import ReplayConfig, seed_set
from router.schemas import (ActorRef, CodeRef, EpisodeResult, Incident, Mode, Outcome, SceneSpec)
from router.stage2 import Branch, nominal_scene


class Rung1Outcome(str, Enum):
    PASS = "pass"
    PASS_WITH_RESIDUAL = "pass_with_residual"
    FAIL = "fail"
    AMBIGUOUS = "ambiguous"


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Rung1Config:
    """Anchors come from the router's calibration (``ReplayConfig``). ``agreement`` is the
    measured replay agreement and ``min_agreement`` the paired/unpaired cut-off (3.2.9);
    both are calibration outputs, so there is no default for either."""
    plan: StagedPlan
    failure_rate: float
    nominal_band: tuple[float, float]
    agreement: float
    min_agreement: float

    def __post_init__(self) -> None:
        lo, hi = self.nominal_band
        if not (0.0 <= self.failure_rate < lo <= hi <= 1.0):
            raise ValueError(f"need 0 <= failure_rate < band_lo <= band_hi <= 1, got {self.failure_rate}, "
                             f"{self.nominal_band}")
        if not (0.0 <= self.agreement <= 1.0 and 0.0 <= self.min_agreement <= 1.0):
            raise ValueError("agreement values must be in [0, 1]")

    @classmethod
    def from_replay(cls, replay: ReplayConfig, plan: StagedPlan, *, agreement: float,
                    min_agreement: float) -> "Rung1Config":
        return cls(plan, replay.failure_rate, replay.nominal_band, agreement, min_agreement)


def staged_plan_for(seeds: int, *, n_a: int = 10, alpha_total: float = 0.05,
                    stage_a_fraction: float = 0.5) -> StagedPlan:
    """``seeds`` is the cumulative Stage B count. Stage A is ``n_a`` (plan: 10) but at most half of it."""
    if seeds < 2:
        raise ValueError("--seeds must be at least 2")
    return StagedPlan(max(1, min(n_a, seeds // 2)), seeds, alpha_total, stage_a_fraction)


def rung1_target(branch: Optional[Branch]) -> str:
    """"nominal" or "incident". Branch UNKNOWN has no defined target (Stage 2 was
    inconclusive), so it raises rather than guess; the loop reports it unresolved."""
    if branch in (Branch.G, Branch.G_DEGRADED):
        return "nominal"
    if branch is Branch.C:
        return "incident"
    raise ValueError(f"no Rung 1 target for branch {branch!r}")


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SceneCheck:
    """One scene, before the patch vs after it, on the same seeds."""
    role: str                      # target | other
    which: str                     # nominal | incident
    scene_key: str
    n: int
    k_before: int
    k_after: int
    prop_before: Proportion
    prop_after: Proportion
    outcome_before: Outcome
    outcome_after: Outcome
    comparison: Comparison

    @property
    def improved(self) -> bool:
        return self.k_after > self.k_before

    @property
    def regression(self) -> bool:
        return self.comparison.clearly_worse or (
            self.outcome_before is Outcome.RESTORED and self.outcome_after in (Outcome.PARTIAL, Outcome.NO_EFFECT))

    @property
    def state(self) -> str:
        """restored | residual | regression (for the non-target scene)."""
        if self.regression:
            return "regression"
        if self.outcome_after is Outcome.RESTORED or self.outcome_before is Outcome.RESTORED:
            return "restored"
        return "residual"

    def to_dict(self) -> dict:
        return {"role": self.role, "which": self.which, "scene": self.scene_key, "n": self.n,
                "k_before": self.k_before, "k_after": self.k_after,
                "ci_before": [self.prop_before.lo, self.prop_before.hi],
                "ci_after": [self.prop_after.lo, self.prop_after.hi],
                "outcome_before": self.outcome_before.value, "outcome_after": self.outcome_after.value,
                "method": self.comparison.method, "p_value": self.comparison.p_value,
                "improved": self.improved, "state": self.state}


@dataclass(frozen=True)
class Rung1Result:
    outcome: Rung1Outcome
    stage: str                                 # "A" or "B": where the decision was made
    n: int                                     # cumulative seeds used per configuration
    branch: Branch
    target: SceneCheck
    other: Optional[SceneCheck]                # None when the incident scene is the nominal scene
    target_decision: StageDecision
    top_error_mode: str
    notes: tuple[str, ...] = ()

    @property
    def back_to_repair(self) -> bool:
        return self.outcome is Rung1Outcome.FAIL

    @property
    def to_router(self) -> bool:
        return self.outcome in (Rung1Outcome.PASS, Rung1Outcome.PASS_WITH_RESIDUAL)

    @property
    def to_rung2(self) -> bool:
        return self.outcome is Rung1Outcome.AMBIGUOUS

    def compact(self) -> str:
        """What the agent (or Rung 2) reads: pass rate, interval, decision, top error mode."""
        def line(c: SceneCheck) -> str:
            return (f"{c.role} [{c.which}] {c.k_after}/{c.n}={c.prop_after.rate:.2f} "
                    f"CI[{c.prop_after.lo:.2f},{c.prop_after.hi:.2f}] {c.outcome_after.value} "
                    f"(before {c.k_before}/{c.n}={c.prop_before.rate:.2f})")
        lines = [f"RUNG1 {self.outcome.value.upper()}  stage {self.stage}, n={self.n}, branch {self.branch.value}, "
                 f"target decision {self.target_decision.value}", line(self.target)]
        if self.other is not None:
            extra = {"restored": "restored", "regression": "REGRESSION",
                     "residual": "improved, not restored" if self.other.improved else "unchanged, not restored"
                     }[self.other.state]
            lines.append(line(self.other) + f"  -> {extra}")
        lines.append(f"top error mode: {self.top_error_mode}")
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"outcome": self.outcome.value, "stage": self.stage, "n": self.n, "branch": self.branch.value,
                "target": self.target.to_dict(), "other": self.other.to_dict() if self.other else None,
                "target_decision": self.target_decision.value, "top_error_mode": self.top_error_mode,
                "notes": list(self.notes)}


def top_error_mode(episodes: Sequence[EpisodeResult]) -> str:
    """Most common failure mode among failed episodes, from ``logs["failure_phase"]`` or
    ``logs["error_mode"]``. TODO(verify): the real episode-log field names (3.2.2 instrumentation)."""
    failed = [e for e in episodes if not e.success]
    if not failed:
        return "none (no failed episodes)"
    modes = Counter(str(m) for e in failed for m in [e.logs.get("failure_phase") or e.logs.get("error_mode")] if m)
    if not modes:
        return f"unknown ({len(failed)} failed, no failure mode in the episode logs)"
    mode, c = sorted(modes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return f"{mode} ({c}/{len(failed)} failed)"


# --------------------------------------------------------------------------- #
# the staged run
# --------------------------------------------------------------------------- #

def run_rung1(backend: RolloutBackend, incident: Incident, after: CodeRef, branch: Branch, cfg: Rung1Config, *,
              before: Optional[CodeRef] = None) -> Rung1Result:
    """Validate ``after`` against ``before`` (default: the incident's code). Seeds are the router's
    (the incident's failing seeds first), so before-runs are cache hits when ``backend`` is the
    router's ``CachedBackend``, and every comparison is paired."""
    before = before or incident.code_ref
    if before.key() == after.key():
        raise ValueError("the patched code is identical to the code under repair; nothing to validate")
    which_target = rung1_target(branch)
    scenes = {"incident": incident.scene, "nominal": nominal_scene(incident)}
    which_other = "nominal" if which_target == "incident" else "incident"
    other_scene = scenes[which_other] if scenes[which_other].key() != scenes[which_target].key() else None
    plan = cfg.plan
    cached = backend if isinstance(backend, CachedBackend) else CachedBackend(backend, RolloutCache())
    seeds = seed_set(incident.failing_seeds, plan.n_b)

    def episodes(scene: SceneSpec, code: CodeRef, n: int) -> list[EpisodeResult]:
        return cached.run(scene, code, incident.policy, seeds[:n], Mode.CLEAN)

    def check(role: str, which: str, scene: SceneSpec, n: int, stage: str) -> tuple[SceneCheck, list[EpisodeResult]]:
        eb, ea = episodes(scene, before, n), episodes(scene, after, n)
        kb, ka = sum(e.success for e in eb), sum(e.success for e in ea)
        conf = plan.confidence(stage)
        pb, pa = wilson_interval(kb, n, conf), wilson_interval(ka, n, conf)
        alpha = plan.alpha_a if stage == "A" else plan.alpha_b
        cmp = compare({e.seed: e.success for e in eb}, {e.seed: e.success for e in ea},
                      agreement=cfg.agreement, min_agreement=cfg.min_agreement, alpha=alpha)
        return SceneCheck(role, which, scene.key(), n, kb, ka, pb, pa,
                          classify_outcome(pb, cfg.failure_rate, cfg.nominal_band),
                          classify_outcome(pa, cfg.failure_rate, cfg.nominal_band), cmp), ea

    def stage(name: str, n: int, final: bool) -> Optional[Rung1Result]:
        """The result if this look decides; None if Stage A is ambiguous (go on to Stage B)."""
        tgt, tgt_after = check("target", which_target, scenes[which_target], n, name)
        oth = check("other", which_other, other_scene, n, name)[0] if other_scene is not None else None
        sd = staged_decision(tgt.k_after, n, name, plan, failure_rate=cfg.failure_rate,
                             nominal_band=cfg.nominal_band)
        notes = []
        if oth is not None and oth.regression:
            notes.append(f"regression on the {oth.which} scene: {oth.k_before}/{n} -> {oth.k_after}/{n} "
                         f"({oth.comparison.method} p={oth.comparison.p_value:.3f})")
        if sd.decision is StageDecision.FAIL:
            outcome = Rung1Outcome.FAIL
            notes.append("target still at the broken baseline")
        elif oth is not None and oth.regression:
            outcome = Rung1Outcome.FAIL
        elif sd.decision is StageDecision.PASS:
            outcome = (Rung1Outcome.PASS if oth is None or oth.state == "restored"
                       else Rung1Outcome.PASS_WITH_RESIDUAL)
        elif final:
            outcome = Rung1Outcome.AMBIGUOUS
        else:
            return None
        return Rung1Result(outcome, name, n, branch, tgt, oth, sd.decision, top_error_mode(tgt_after), tuple(notes))

    return stage("A", plan.n_a, False) or stage("B", plan.n_b, True)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def rule_from_dict(d: Mapping) -> MockRule:
    return MockRule(rate=float(d["rate"]),
                    scene=SceneSpec.from_dict(d["scene"]) if d.get("scene") else None,
                    factors_present=tuple(d.get("factors_present", ())),
                    code=CodeRef.from_dict(d["code"]) if d.get("code") else None,
                    actor=ActorRef.from_dict(d["actor"]) if d.get("actor") else None,
                    mode=Mode(d["mode"]) if d.get("mode") else None,
                    files=FileContains.from_dict(d["files"]) if d.get("files") else None)


def load_config(path: str, seeds: int, cwd: str) -> tuple[RolloutBackend, Incident, CodeRef, Branch, Rung1Config]:
    """JSON config (all keys required unless marked optional):

        {"incident": <Incident.to_dict()>, "branch": "G" | "G_degraded" | "C",
         "after": <CodeRef dict>,                   # optional: default = the code in the working directory
                                                    #   (ladder.workspace.code_ref_of: same recipe as the repairer)
         "workspace": {"store": DIR, "bases": {label: DIR}},  # optional: needed by "files" rules; the
                                                    #   default "after" is then also stored so it can be built,
                                                    #   and the incident's code must be a workspace name
                                                    #   (<label>@<tree12>[+<diff12>])
         "thresholds": {"failure_rate": .., "nominal_band": [lo, hi], "agreement": .., "min_agreement": ..,
                        "n_a": 10, "alpha_total": 0.05, "stage_a_fraction": 0.5},    # last three optional
         "backend": {"kind": "mock", "default_rate": .., "rules": [<rule dict>...], "salt": "x"}
                    | {"kind": "isaac"}}           # isaac raises NotImplementedError until built
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    incident = Incident.from_dict(raw["incident"])
    th = raw["thresholds"]
    plan = staged_plan_for(seeds, n_a=int(th.get("n_a", 10)), alpha_total=float(th.get("alpha_total", 0.05)),
                           stage_a_fraction=float(th.get("stage_a_fraction", 0.5)))
    cfg = Rung1Config(plan, float(th["failure_rate"]), tuple(th["nominal_band"]), float(th["agreement"]),
                      float(th["min_agreement"]))
    ws = raw.get("workspace")
    workspaces = WorkspaceManager(ws["store"], ws["bases"]) if ws else None
    # TODO(F2b): the eval config of "after" comes from the working copy, not from the incident (ISSUES #2).
    if raw.get("after"):
        after = CodeRef.from_dict(raw["after"])
    elif workspaces is not None:
        after = workspaces.adopt(cwd, incident.code_ref.eval_config)
    else:
        after = code_ref_of(cwd, incident.code_ref.eval_config)
    b = raw["backend"]
    if b["kind"] == "mock":
        backend: RolloutBackend = MockBackend([rule_from_dict(r) for r in b.get("rules", [])],
                                              float(b["default_rate"]), salt=b.get("salt", "mock"),
                                              workspaces=workspaces)
    elif b["kind"] == "isaac":
        backend = IsaacBackend()
    else:
        raise ValueError(f"unknown backend kind {b['kind']!r}")
    return backend, incident, after, Branch(raw["branch"]), cfg


EXIT = {Rung1Outcome.PASS: 0, Rung1Outcome.PASS_WITH_RESIDUAL: 0, Rung1Outcome.FAIL: 1, Rung1Outcome.AMBIGUOUS: 3}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="run_rung1", description="Staged simulator validation of the current patch.")
    ap.add_argument("--seeds", type=int, default=50, help="cumulative seeds after Stage B (default 50)")
    ap.add_argument("--config", default=os.environ.get("RUNG1_CONFIG"), help="JSON config (default $RUNG1_CONFIG)")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON instead of the compact text")
    args = ap.parse_args(argv)
    if not args.config:
        ap.error("no config: pass --config or set RUNG1_CONFIG")
    try:
        backend, incident, after, branch, cfg = load_config(args.config, args.seeds, os.getcwd())
        result = run_rung1(backend, incident, after, branch, cfg)
    except (ValueError, KeyError, OSError, NotImplementedError, WorkspaceError) as e:
        print(f"run_rung1: error: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True) if args.json else result.compact())
    return EXIT[result.outcome]


if __name__ == "__main__":
    raise SystemExit(main())
