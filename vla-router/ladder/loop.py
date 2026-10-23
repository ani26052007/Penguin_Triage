"""The incident loop (plan.md 3.2.11 and Section 4).

    IncidentLoop(backend, cfg, repairer=..., evidence_provider=..., data_ladder=...).run(incident)

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
  * ``EvidenceProvider``    ``Incident -> Evidence``. ``route()`` takes Stage 1 / C_behavior
                            evidence as an argument and Stage 1 is not wired into it, so the loop
                            asks a provider per pass (after a fix, the contract-violation record
                            disappears). The mock provider keyed on the CodeRef is a FIXTURE:
                            remove it when Stage 1 is wired into ``route()`` (build-order step 5).
  * ``DataLadder``          stub; the real pipeline is Section 5.
  * Rung 0 / Rung 2 hooks   optional callables; ``None`` skips Rung 0 and makes Rung 2 answer
                            "unclear".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Optional, Protocol, Sequence

from ladder.backend import RolloutBackend
from ladder.repair import RepairKind, RepairOutcome, RepairRequest, Repairer
from ladder.rung0 import Rung0Result
from ladder.rung1 import Rung1Config, Rung1Outcome, Rung1Result, run_rung1
from ladder.rung2 import Rung2Input, Rung2Label, Rung2Result
from router.prereq.rollout_cache import CachedBackend, RolloutCache
from router.router import RouterConfig, RouterResult, route
from router.schemas import CodeRef, Destination, Evidence, Incident, SceneSpec, Verdict
from router.stage2 import Branch, nominal_scene

log = logging.getLogger("ladder.loop")

EvidenceProvider = Callable[[Incident], Sequence[Evidence]]
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

    def __post_init__(self) -> None:
        if self.max_passes < 1 or self.max_repairs < 1 or self.max_rung2_calls < 0:
            raise ValueError("max_passes and max_repairs must be >= 1, max_rung2_calls >= 0")


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
    attempt: int
    kind: str
    code_before: str
    code_after: str
    result: str                                   # no_change | rung0_failed | rung1_fail | rung1_<outcome>
    rung0: Optional[Rung0Result] = None
    rung1: Optional[Rung1Result] = None


@dataclass(frozen=True)
class LoopResult:
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

    @property
    def destinations(self) -> tuple[str, ...]:
        return tuple(p.destination for p in self.passes)

    def to_dict(self) -> dict:
        return {"incident_id": self.incident_id, "status": self.status.value, "reason": self.reason,
                "passes": [p.__dict__ for p in self.passes],
                "repairs": [{"attempt": r.attempt, "kind": r.kind, "code_before": r.code_before,
                             "code_after": r.code_after, "result": r.result,
                             "rung0": r.rung0.to_dict() if r.rung0 else None,
                             "rung1": r.rung1.to_dict() if r.rung1 else None} for r in self.repairs],
                "rung2": [r.to_dict() for r in self.rung2], "final_code": self.final_code.to_dict(),
                "final_verdict": self.final.verdict.to_dict() if self.final else None,
                "handoff_factor": self.handoff.factor if self.handoff else None,
                "big_calls": self.big_calls, "cost": self.cost}


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #

class IncidentLoop:
    def __init__(self, backend: RolloutBackend, cfg: LoopConfig, *, repairer: Repairer,
                 evidence_provider: EvidenceProvider, data_ladder: DataLadder,
                 rung0: Optional[Rung0Hook] = None, rung2: Optional[Rung2Fn] = None):
        self.backend = backend if isinstance(backend, CachedBackend) else CachedBackend(backend, RolloutCache())
        self.cfg, self.repairer, self.evidence_provider = cfg, repairer, evidence_provider
        self.data_ladder, self.rung0, self.rung2 = data_ladder, rung0, rung2

    def run(self, incident: Incident) -> LoopResult:
        return _Run(self, incident).execute()


class _Run:
    """State of one incident."""

    def __init__(self, loop: IncidentLoop, incident: Incident):
        self.L, self.cfg, self.incident = loop, loop.cfg, incident
        self.code = incident.code_ref
        self.passes: list[PassRecord] = []
        self.repairs: list[RepairRecord] = []
        self.rung2s: list[Rung2Result] = []
        self.feedback: list[str] = []
        self.pending_rung1: Optional[str] = None      # Rung 1 text when it was ambiguous: Rung 2 reads the next pass
        self.big_calls, self.cost = 0, 0.0
        self.last: Optional[RouterResult] = None

    # -- helpers ----------------------------------------------------------- #
    def finish(self, status: LoopStatus, reason: str, handoff: Optional[DataHandoff] = None) -> LoopResult:
        log.info("incident %s: %s (%s) passes=%d repairs=%d", self.incident.incident_id, status.value, reason,
                 len(self.passes), len(self.repairs))
        return LoopResult(self.incident.incident_id, status, reason, tuple(self.passes), tuple(self.repairs),
                          tuple(self.rung2s), self.code, self.last, handoff, self.big_calls, self.cost)

    def execute(self) -> LoopResult:
        while True:
            if len(self.passes) >= self.cfg.max_passes:
                return self.finish(LoopStatus.UNRESOLVED, f"pass cap ({self.cfg.max_passes}) reached")
            cur = replace(self.incident, code_ref=self.code)
            evidence = tuple(self.L.evidence_provider(cur))
            rr = route(cur, self.L.backend, self.cfg.router, evidence)
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
                r2 = self.rung2(cur, rr, text)
                if r2.label is Rung2Label.DATA_GAP:
                    rec.action = "rung2:data_gap"
                    return self.handoff(cur, rr, via_rung2=True)
                if r2.label is Rung2Label.CODE:
                    rec.action = "rung2:code"
                    self.feedback.append(f"Rung 2 (big model) says a code problem remains: {r2.reason}")
                    stop = self.repair(RepairKind.CODE, cur, rr, evidence, rec)
                    if stop:
                        return stop
                    continue
                rec.action = "rung2:unclear"
                return self.finish(LoopStatus.UNRESOLVED, f"Rung 2 unclear: {r2.rejected_reason or r2.reason}")

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

    def rung2(self, cur: Incident, rr: RouterResult, rung1_text: str) -> Rung2Result:
        if self.L.rung2 is None or len(self.rung2s) >= self.cfg.max_rung2_calls:
            why = "Rung 2 not configured" if self.L.rung2 is None else "Rung 2 call budget exhausted"
            res = Rung2Result(Rung2Label.UNCLEAR, None, False, why, source="fallback")
        else:
            res = self.L.rung2(Rung2Input(cur.incident_id, rr.verdict, rr.replays, rr.evidence, rung1_text))
        self.big_calls += res.calls
        self.cost += res.cost
        self.rung2s.append(res)
        return res

    def repair(self, kind: RepairKind, cur: Incident, rr: RouterResult, evidence: tuple[Evidence, ...],
               rec: PassRecord) -> Optional[LoopResult]:
        """Repair until Rung 1 lets the patch through (return None: the loop re-runs the router on
        the patched code) or a cap ends the incident (return the terminal result)."""
        if rr.branch not in (Branch.G, Branch.G_DEGRADED, Branch.C):
            return self.finish(LoopStatus.UNRESOLVED, "Stage 2 branch unknown: Rung 1 has no defined target scene")
        while True:
            if len(self.repairs) >= self.cfg.max_repairs:
                return self.finish(LoopStatus.UNRESOLVED, f"repair cap ({self.cfg.max_repairs}) reached")
            req = RepairRequest(kind, cur, rr.verdict, evidence, rr.branch, len(self.repairs) + 1,
                                tuple(self.feedback))
            out = self.L.repairer.repair(req)
            self.cost += out.cost
            self.big_calls += out.big_calls
            r = RepairRecord(req.attempt, kind.value, self.code.key(), out.code_ref.key(), "")
            self.repairs.append(r)
            if out.code_ref.key() == self.code.key():
                r.result = "no_change"
                self.feedback.append(f"attempt {req.attempt}: the patch changed nothing. {out.notes}".strip())
                continue
            if self.L.rung0 is not None:
                r.rung0 = self.L.rung0(req, out)
                self.cost += r.rung0.reviewer_cost
                self.big_calls += r.rung0.reviewer_calls
                if not r.rung0.passed:
                    r.result = "rung0_failed"
                    self.feedback.append(f"attempt {req.attempt}: {r.rung0.feedback}")
                    continue
            r.rung1 = run_rung1(self.L.backend, cur, out.code_ref, rr.branch, self.cfg.rung1)
            r.result = f"rung1_{r.rung1.outcome.value}"
            log.info("rung1 attempt %d: %s", req.attempt, r.rung1.outcome.value)
            if r.rung1.outcome is Rung1Outcome.FAIL:
                self.feedback.append(f"attempt {req.attempt}: simulator validation failed.\n{r.rung1.compact()}")
                continue
            self.code = out.code_ref
            self.feedback.clear()
            if r.rung1.outcome is Rung1Outcome.AMBIGUOUS:
                self.pending_rung1 = r.rung1.compact()
            return None


__all__ = ["DataHandoff", "DataLadder", "EvidenceProvider", "IncidentLoop", "LoopConfig", "LoopResult",
           "LoopStatus", "PassRecord", "RepairRecord", "StubDataLadder"]
