"""The incident loop (plan.md 3.2.11 and Section 4).

    IncidentLoop(backend, cfg, repairer=..., inputs=..., data_ladder=...).run(incident)

    incident -> router -> destination --+-- hagent / config_fix: Repairer -> Rung 0 -> Rung 1
                  ^                     |        fail                -> Repairer again (feedback)
                  |                     |        pass / residual     -> router re-run on the fixed system
                  |                     |        ambiguous           -> router re-run -> Rung 2
                  |                     +-- data_ladder: hand-off (factor, failing seeds, nominal scene)
                  |                     +-- report / stop: terminal
                  +---- re-run ---------+

After ANY fix the incident goes back through the router (verification that the cause is gone,
second-cause detection, and the input to Rung 2). Code is fixed before vision: the decision
table's precedence returns a code verdict first (row 4 before rows 6-7), and the loop acts on
exactly one verdict per pass, so a vision hand-off can only happen after the code route has
been closed. Passes are capped (default 3); a cap hit is "unresolved" with everything attached.

Rung 1 scene choice is keyed on the router's branch (``ladder.rung1``); a ``pass_with_residual``
goes to the router and NEVER back to the repair agent, which is what stops a patch that cannot
fix the second cause from looping the repair agent forever.

Stand-ins (all behind interfaces, none a model in the routing path):
  * ``Repairer``            real one: ``ladder.repair.HagentRepairer``.
  * ``RouterInputs``        the Stage 1 checks and the C_behavior source. ``route()`` runs them
                            itself on every pass, on the incident with the current code, so
                            evidence of a fixed fault disappears because the checks no longer
                            find it in that code's logs.
  * ``DataLadder``          stub; the real pipeline is Section 5.
  * Rung 0 / Rung 2 hooks   optional callables; ``None`` skips Rung 0 and makes Rung 2 answer
                            "unclear".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional, Protocol

from ladder.backend import RolloutBackend
from ladder.repair import RepairKind, RepairOutcome, RepairRequest, Repairer
from ladder.rung0 import Rung0Result
from ladder.rung1 import Rung1Config, Rung1Outcome, Rung1Result, run_rung1
from ladder.rung2 import Rung2Input, Rung2Label, Rung2Result
from router.prereq.rollout_cache import CachedBackend, RolloutCache
from router.router import RouterConfig, RouterInputs, RouterResult, route
from router.schemas import CodeRef, Destination, Evidence, Incident, SceneSpec, Verdict
from router.stage2 import Branch, nominal_scene

log = logging.getLogger("ladder.loop")

CODE_OR_UNDETERMINED_ROWS = (4, 10)   # the rows where a Rung 2 label is used
RESTORE_ROWS = (6, 7)                 # vision verdicts: the data ladder, whatever Rung 2 says

Rung0Hook = Callable[[RepairRequest, RepairOutcome], Rung0Result]
Rung2Fn = Callable[[Rung2Input], Rung2Result]


class LoopStatus(str, Enum):
    CLOSED = "closed"              # router re-run (or first run) found no significant failure
    DATA_HANDOFF = "data_handoff"  # vision gap handed to the data ladder
    REPORTED = "reported"          # report-only destinations (rows 2, 3, 8, 9)
    UNRESOLVED = "unresolved"      # row 10, caps hit, Rung 2 unclear, undefined Rung 1 target


@dataclass(frozen=True)
class LoopConfig:
    router: RouterConfig
    rung1: Rung1Config
    max_passes: int = 3            # router passes per incident (plan: two or three)
    max_repairs: int = 3           # repair attempts per incident, over all passes
    max_rung2_calls: int = 1       # plan: one big-model call per incident
    max_cost: float = 0.0          # total cost of the incident (repairs, reviews, Rung 2); 0 = off
    max_big_calls: int = 0         # hagent's big calls and reviews + Rung 0 reviews + Rung 2; 0 = off

    def __post_init__(self) -> None:
        if self.max_passes < 1 or self.max_repairs < 1 or self.max_rung2_calls < 0:
            raise ValueError("max_passes and max_repairs must be >= 1, max_rung2_calls >= 0")
        if self.max_cost < 0 or self.max_big_calls < 0:
            raise ValueError("max_cost and max_big_calls must be >= 0 (0 = off)")


# --------------------------------------------------------------------------- #
# data-ladder hand-off (stub)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DataHandoff:
    """What the data ladder receives (3.2.11): the factor (it picks the rung), the failing seeds,
    the scenes to build nominal rollouts from, and the verdict. ``partial`` is set when the
    restore only partly recovered success. The held-out conditions are not defined yet (Section 5)."""
    incident_id: str
    factor: str
    scene: SceneSpec
    nominal_scene: SceneSpec
    failing_seeds: tuple[int, ...]
    code_ref: CodeRef              # the system to build the data against (after any code fix)
    verdict: Verdict
    partial: bool = False
    via_rung2: bool = False


class DataLadder(Protocol):
    def handoff(self, h: DataHandoff) -> str: ...


class StubDataLadder:
    """Records hand-offs; builds no data."""

    def __init__(self) -> None:
        self.handoffs: list[DataHandoff] = []

    def handoff(self, h: DataHandoff) -> str:
        self.handoffs.append(h)
        return f"stub: recorded hand-off for factor {h.factor!r}"


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #

@dataclass
class PassRecord:
    index: int
    code_key: str
    label: str
    row: int
    destination: str
    branch: Optional[str]
    factor: Optional[str]
    likely_second_cause: bool
    multi_cause_flag: bool
    by_elimination: bool
    episodes_simulated: int
    cache_hits: int
    action: str = ""


@dataclass
class RepairRecord:
    """``diff``: what this attempt changed, against ``code_before``. ``code_diff``: the whole change
    of ``code_after`` against its base, which ``WorkspaceManager.materialize`` rebuilds."""
    attempt: int
    kind: str
    code_before: str
    code_after: str
    result: str          # no_change | unrebuildable_patch | rung0_failed | rung1_fail | rung1_<outcome>
    rung0: Optional[Rung0Result] = None
    rung1: Optional[Rung1Result] = None
    diff: str = ""
    code_diff: str = ""
    reviews: tuple[dict, ...] = ()                       # hagent's reviewer log for this attempt
    pass_probabilities: tuple[Optional[float], ...] = ()   # its reviewers' pass probabilities (plan 4.3)


@dataclass(frozen=True)
class LoopResult:
    """``final_diff`` is the whole change of ``final_code`` against its base ``base`` (the incident
    code's own diff when no fix was accepted; "" when the loop has no workspaces):
    ``WorkspaceManager.register(final_code, final_diff)`` then ``materialize(final_code)`` rebuilds
    the final files from this record and the base checkout alone. The diffs are the lossless
    ``str`` view of bytes (``ladder.workspace.as_text``): serialise with ``json`` (ASCII escapes)."""
    incident_id: str
    status: LoopStatus
    reason: str
    passes: tuple[PassRecord, ...]
    repairs: tuple[RepairRecord, ...]
    rung2: tuple[Rung2Result, ...]
    final_code: CodeRef
    final: Optional[RouterResult]                 # the last router result: verdict, replay table, evidence
    handoff: Optional[DataHandoff]
    big_calls: int
    cost: float
    final_diff: str = ""
    base: str = ""                                # base name (label@tree12) of the incident code
    error_stage: str = ""                         # router | repairer | rung0 | rung1 | rung2, when an exception ended it
    error_type: str = ""
    error_message: str = ""
    rung2_pending: bool = False                   # the pass cap hit while a Rung 2 call was still due
    warnings: tuple[str, ...] = ()                # e.g. a cost cap that could never fire (prices are 0)

    @property
    def destinations(self) -> tuple[str, ...]:
        return tuple(p.destination for p in self.passes)

    def to_dict(self) -> dict:
        return {"incident_id": self.incident_id, "status": self.status.value, "reason": self.reason,
                "passes": [p.__dict__ for p in self.passes],
                "repairs": [{"attempt": r.attempt, "kind": r.kind, "code_before": r.code_before,
                             "code_after": r.code_after, "result": r.result, "diff": r.diff,
                             "code_diff": r.code_diff, "reviews": list(r.reviews),
                             "pass_probabilities": list(r.pass_probabilities),
                             "rung0": r.rung0.to_dict() if r.rung0 else None,
                             "rung1": r.rung1.to_dict() if r.rung1 else None} for r in self.repairs],
                "rung2": [r.to_dict() for r in self.rung2], "final_code": self.final_code.to_dict(),
                "final_diff": self.final_diff, "base": self.base,
                "final_verdict": self.final.verdict.to_dict() if self.final else None,
                "handoff_factor": self.handoff.factor if self.handoff else None,
                "big_calls": self.big_calls, "cost": self.cost,
                "error_stage": self.error_stage, "error_type": self.error_type,
                "error_message": self.error_message, "rung2_pending": self.rung2_pending,
                "warnings": list(self.warnings)}


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #

_REBUILD_ADVICE = {
    "binary": "it edits or adds a binary file, which a text diff cannot rebuild. Do not edit or add binary files.",
    "encoding": "it changes text next to bytes that are not valid UTF-8, and the patch text lost those bytes. "
                "Leave the non-UTF-8 lines (and the lines around them) alone.",
    "symlink": "it adds a symbolic link that is absolute or points outside the repository. Do not create such links.",
    "does_not_apply": "it does not apply to a clean copy of the code. Make plain edits to the files; do not "
                      "change git settings or history.",
    "mismatch": "it does not give the files you left in the working copy (for example because changes were "
                "committed). Do not commit, stash or reset; leave your changes uncommitted.",
}


def rebuild_feedback(attempt: int, cause: str, error: str) -> str:
    """The feedback for an attempt whose patch could not be rebuilt, by ``UnrebuildablePatch.cause``."""
    advice = _REBUILD_ADVICE.get(cause, _REBUILD_ADVICE["does_not_apply"])
    return (f"attempt {attempt}: the patch could not be rebuilt on a clean copy of the code, so it was not "
            f"validated: {advice} ({error[:300]})")


class IncidentLoop:
    """``workspaces`` (a ``ladder.workspace.WorkspaceManager``; default: the repairer's
    ``workspaces``, if it has one) names the incident code's base and gives its cumulative diff, so
    ``LoopResult.base`` / ``final_diff`` rebuild the final code even when the incident started on
    patched code. Without one both stay "" (scripted repairers whose codes are not workspace names)."""

    def __init__(self, backend: RolloutBackend, cfg: LoopConfig, *, repairer: Repairer,
                 inputs: RouterInputs, data_ladder: DataLadder,
                 rung0: Optional[Rung0Hook] = None, rung2: Optional[Rung2Fn] = None, workspaces=None):
        self.backend = backend if isinstance(backend, CachedBackend) else CachedBackend(backend, RolloutCache())
        self.cfg, self.repairer, self.inputs = cfg, repairer, inputs
        self.data_ladder, self.rung0, self.rung2 = data_ladder, rung0, rung2
        self.workspaces = workspaces if workspaces is not None else getattr(repairer, "workspaces", None)
        self.warnings: tuple[str, ...] = ()
        warn = getattr(repairer, "pricing_warnings", None)
        if cfg.max_cost > 0 and warn is not None:
            self.warnings = tuple(warn(cfg.max_cost))
            for w in self.warnings:
                log.warning("%s", w)

    def run(self, incident: Incident) -> LoopResult:
        """Never raises for an error in a stage (ISSUES #10): the incident ends UNRESOLVED with the error
        recorded. ``KeyboardInterrupt`` and ``SystemExit`` propagate."""
        return _Run(self, incident).execute()


class _Run:
    """State of one incident."""

    def __init__(self, loop: IncidentLoop, incident: Incident):
        self.L, self.cfg, self.incident = loop, loop.cfg, incident
        self.code = incident.code_ref
        ws = loop.workspaces
        self.base = ws.base_name(incident.code_ref) if ws is not None else ""
        self.code_diff = ws.diff_of(incident.code_ref) if ws is not None else ""   # whole change of self.code
        self.passes: list[PassRecord] = []
        self.repairs: list[RepairRecord] = []
        self.rung2s: list[Rung2Result] = []
        self.feedback: list[str] = []
        self.pending_rung1: Optional[str] = None      # Rung 1 text when it was ambiguous: Rung 2 reads the next pass
        self.big_calls, self.cost = 0, 0.0
        self.last: Optional[RouterResult] = None
        self.stage = "router"                          # the stage running now, for an error record
        self.in_flight: Optional[RepairRecord] = None  # the repair attempt being validated
        self.error: tuple[str, str, str] = ("", "", "")
        self.rung2_pending = False

    # -- helpers ----------------------------------------------------------- #
    def budget_stop(self) -> str:
        """Why no further big-model or repair spend is allowed ("" when it is)."""
        c = self.cfg
        if c.max_cost > 0 and self.cost >= c.max_cost:
            return f"cost cap (${c.max_cost}) reached: spent ${self.cost:.4f}"
        if c.max_big_calls > 0 and self.big_calls >= c.max_big_calls:
            return f"big-call cap ({c.max_big_calls}) reached: {self.big_calls} calls"
        return ""

    def finish(self, status: LoopStatus, reason: str, handoff: Optional[DataHandoff] = None) -> LoopResult:
        log.info("incident %s: %s (%s) passes=%d repairs=%d", self.incident.incident_id, status.value, reason,
                 len(self.passes), len(self.repairs))
        return LoopResult(self.incident.incident_id, status, reason, tuple(self.passes), tuple(self.repairs),
                          tuple(self.rung2s), self.code, self.last, handoff, self.big_calls, self.cost,
                          self.code_diff, self.base, *self.error, self.rung2_pending, self.L.warnings)

    def execute(self) -> LoopResult:
        try:
            return self._execute()
        except Exception as e:     # not BaseException: KeyboardInterrupt / SystemExit propagate
            stage = self.stage
            self.error = (stage, type(e).__name__, str(e))
            if self.in_flight is not None and not self.in_flight.result:
                self.in_flight.result = f"{stage}_error"
            log.exception("incident %s: %s raised", self.incident.incident_id, stage)
            return self.finish(LoopStatus.UNRESOLVED, f"error in {stage}: {type(e).__name__}: {e}")

    def _execute(self) -> LoopResult:
        while True:
            if len(self.passes) >= self.cfg.max_passes:
                if self.pending_rung1 is not None:
                    self.rung2_pending = True
                    return self.finish(LoopStatus.UNRESOLVED, f"pass cap ({self.cfg.max_passes}) reached while a "
                                       f"Rung 2 call was pending (Rung 1 was ambiguous): Rung 2 never ran")
                return self.finish(LoopStatus.UNRESOLVED, f"pass cap ({self.cfg.max_passes}) reached")
            cur = replace(self.incident, code_ref=self.code)
            self.stage = "router"
            rr = route(cur, self.L.backend, self.cfg.router, self.L.inputs)
            evidence = rr.evidence
            self.last = rr
            v = rr.verdict
            rec = PassRecord(len(self.passes) + 1, self.code.key(), v.label.value, v.row, v.destination.value,
                             rr.branch.value if rr.branch else None, v.factor, v.likely_second_cause,
                             v.multi_cause_flag, v.by_elimination, rr.episodes_simulated, rr.cache_hits)
            self.passes.append(rec)

            # 1. the verdict after an ambiguous Rung 1 goes to Rung 2 first
            if self.pending_rung1 is not None:
                text, self.pending_rung1 = self.pending_rung1, None
                if v.destination is Destination.STOP:
                    rec.action = "close"
                    return self.finish(LoopStatus.CLOSED, "router re-run found no significant failure; "
                                                          "Rung 2 not needed")
                stop = self.budget_stop()
                if stop:
                    rec.action = "budget"
                    return self.finish(LoopStatus.UNRESOLVED, f"{stop}; Rung 2 not called")
                r2 = self.rung2(cur, rr, text)
                # The router re-run verdict wins (ISSUES #4, F3): Rung 2's label is used only on rows
                # 4 and 10; rows 6-7 go to the data ladder whatever it says; any other row follows the
                # router and the label is logged only. The label stays in the result either way.
                if v.row in CODE_OR_UNDETERMINED_ROWS:
                    if r2.label is Rung2Label.CODE:
                        rec.action = "rung2:code"
                        self.feedback.append(f"Rung 2 (big model) says a code problem remains: {r2.reason}")
                        stop = self.repair(RepairKind.CODE, cur, rr, evidence, rec)
                        if stop:
                            return stop
                        continue
                    rec.action = "rung2:unclear"      # unclear, or a data_gap claim the gate turned into unclear
                    return self.finish(LoopStatus.UNRESOLVED, f"Rung 2 unclear: {r2.rejected_reason or r2.reason}")
                if v.row in RESTORE_ROWS:
                    rec.action = "rung2:data_gap" if r2.label is Rung2Label.DATA_GAP else "data_handoff"
                    if r2.label is not Rung2Label.DATA_GAP:
                        self.override(r2, f"overridden by router: re-run is row {v.row} ({v.label.value}), "
                                          f"handed to the data ladder")
                    return self.handoff(cur, rr, via_rung2=r2.label is Rung2Label.DATA_GAP)
                self.override(r2, f"overridden by router: re-run is row {v.row} ({v.label.value}); "
                                  f"the router's destination {v.destination.value!r} is followed, label logged only")

            # 2. otherwise act on the router's destination
            dest = v.destination
            if dest is Destination.STOP:
                rec.action = "close"
                return self.finish(LoopStatus.CLOSED, "no significant failure" if len(self.passes) == 1
                                   else "fixed: the router re-run found no significant failure")
            if dest is Destination.REPORT:
                rec.action = "report"
                status = LoopStatus.UNRESOLVED if v.row == 10 else LoopStatus.REPORTED
                return self.finish(status, f"row {v.row} {v.label.value}: report only")
            if dest is Destination.DATA_LADDER:
                rec.action = "data_handoff"
                return self.handoff(cur, rr, via_rung2=False)
            rec.action = f"repair:{dest.value}"
            kind = RepairKind.CODE if dest is Destination.HAGENT else RepairKind.CONFIG
            stop = self.repair(kind, cur, rr, evidence, rec)
            if stop:
                return stop

    # -- destinations ------------------------------------------------------- #
    def handoff(self, cur: Incident, rr: RouterResult, *, via_rung2: bool) -> LoopResult:
        v = rr.verdict
        h = DataHandoff(cur.incident_id, v.factor or "", cur.scene, nominal_scene(cur), cur.failing_seeds,
                        cur.code_ref, v, partial=v.likely_second_cause, via_rung2=via_rung2)
        status = self.L.data_ladder.handoff(h)
        log.info("data ladder: %s", status)
        return self.finish(LoopStatus.DATA_HANDOFF, f"factor {h.factor!r} handed to the data ladder ({status})", h)

    def override(self, res: Rung2Result, note: str) -> None:
        self.rung2s[-1] = replace(res, overridden=note)

    def rung2(self, cur: Incident, rr: RouterResult, rung1_text: str) -> Rung2Result:
        if self.L.rung2 is None or len(self.rung2s) >= self.cfg.max_rung2_calls:
            why = "Rung 2 not configured" if self.L.rung2 is None else "Rung 2 call budget exhausted"
            res = Rung2Result(Rung2Label.UNCLEAR, None, False, why, source="fallback")
        else:
            self.stage = "rung2"
            res = self.L.rung2(Rung2Input(cur.incident_id, rr.verdict, rr.replays, rr.evidence, rung1_text))
        self.big_calls += res.calls
        self.cost += res.cost
        self.rung2s.append(res)
        return res

    def repair(self, kind: RepairKind, cur: Incident, rr: RouterResult, evidence: tuple[Evidence, ...],
               rec: PassRecord) -> Optional[LoopResult]:
        """Repair until Rung 1 lets the patch through (return None: the loop re-runs the router on
        the patched code) or a cap ends the incident (return the terminal result).

        Every attempt is asked to repair ``cur``, the last accepted code (Rung 1 pass, residual or
        ambiguous). ``self.code`` only moves on acceptance, so a failed attempt (no_change,
        unrebuildable_patch, rung0_failed, rung1_fail) never becomes the next attempt's start."""
        if rr.branch not in (Branch.G, Branch.G_DEGRADED, Branch.C):
            return self.finish(LoopStatus.UNRESOLVED, "Stage 2 branch unknown: Rung 1 has no defined target scene")
        while True:
            if len(self.repairs) >= self.cfg.max_repairs:
                return self.finish(LoopStatus.UNRESOLVED, f"repair cap ({self.cfg.max_repairs}) reached")
            stop = self.budget_stop()
            if stop:
                return self.finish(LoopStatus.UNRESOLVED, f"{stop}; no further repair attempt")
            req = RepairRequest(kind, cur, rr.verdict, evidence, rr.branch, len(self.repairs) + 1,
                                tuple(self.feedback))
            self.stage, self.in_flight = "repairer", None
            out = self.L.repairer.repair(req)
            self.cost += out.cost
            self.big_calls += out.big_calls
            r = RepairRecord(req.attempt, kind.value, self.code.key(), out.code_ref.key(), "", diff=out.diff,
                             code_diff=out.code_diff, reviews=out.reviews,
                             pass_probabilities=out.pass_probabilities)
            self.repairs.append(r)
            self.in_flight = r
            if out.rebuild_error:
                r.result = "unrebuildable_patch"
                self.feedback.append(rebuild_feedback(req.attempt, out.rebuild_cause, out.rebuild_error))
                continue
            if out.code_ref.key() == self.code.key():
                r.result = "no_change"
                self.feedback.append(f"attempt {req.attempt}: the patch changed nothing. {out.notes}".strip())
                continue
            if self.L.rung0 is not None:
                self.stage = "rung0"
                r.rung0 = self.L.rung0(req, out)
                self.cost += r.rung0.reviewer_cost
                self.big_calls += r.rung0.reviewer_calls
                if not r.rung0.passed:
                    r.result = "rung0_failed"
                    self.feedback.append(f"attempt {req.attempt}: {r.rung0.feedback}")
                    continue
            self.stage = "rung1"
            r.rung1 = run_rung1(self.L.backend, cur, out.code_ref, rr.branch, self.cfg.rung1)
            r.result = f"rung1_{r.rung1.outcome.value}"
            log.info("rung1 attempt %d: %s", req.attempt, r.rung1.outcome.value)
            if r.rung1.outcome is Rung1Outcome.FAIL:
                self.feedback.append(f"attempt {req.attempt}: simulator validation failed.\n{r.rung1.compact()}")
                continue
            self.code, self.code_diff = out.code_ref, out.code_diff
            self.feedback.clear()
            if r.rung1.outcome is Rung1Outcome.AMBIGUOUS:
                self.pending_rung1 = r.rung1.compact()
            return None


__all__ = ["DataHandoff", "DataLadder", "IncidentLoop", "LoopConfig", "LoopResult",
           "LoopStatus", "PassRecord", "RepairRecord", "StubDataLadder", "rebuild_feedback"]
