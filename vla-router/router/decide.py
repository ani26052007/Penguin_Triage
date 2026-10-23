"""Stage 3: the decision table (plan.md 3.2.8) as a pure function.

    decide(evidence, replays, config) -> Verdict

No I/O, no randomness, no simulator, no models. The result depends only on the
*set* of evidence records and the *set* of replays, never on their order
(principle 2 in 3.2.1). Callers classify each replay's outcome with
``ladder.stats`` first; the signal and replay-id vocabulary is in
``router.vocab``.

How a row is decided
--------------------
Every row condition is evaluated to TRUE / FALSE / UNKNOWN. UNKNOWN means the
replays it needs are missing or INCONCLUSIVE (the interval straddles a
boundary: extend the seeds, 3.2.9). The verdict is the first TRUE row, except:

* if a higher-precedence row is UNKNOWN, the lower row cannot be trusted, so
  the verdict is row 10 (undetermined) and says which replays need more seeds;
* if the winning row sees conflicting evidence, the verdict is row 10.

Confidence (code verdicts): ``by_elimination=False`` when a located cause
exists (a contract violation, a constant delay, a consistent implied-target
offset, or G3 failing while G2 passes); ``True`` when the verdict rests on
Branch G elimination alone.

Flags: ``likely_second_cause`` when the winning replay only partly restores,
or when a code/infra verdict also sees a render/vision restore that is only
partial (a full restore is what a condition-specific code bug looks like, so
it is not flagged). ``multi_cause_flag`` when Stage 2 came back degraded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

from router import vocab as V
from router.schemas import Destination, Evidence, Outcome, ReplayResult, Verdict, VerdictLabel


# --------------------------------------------------------------------------- #
# public surface
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DecideConfig:
    """``min_strength``: evidence weaker than this is ignored. Evidence
    strength is already calibrated on no-fault runs (3.2.3); the cut-off is a
    step-6 calibration output, so it defaults to "use everything"."""
    min_strength: float = 0.0


LABEL_BY_ROW: Mapping[int, VerdictLabel] = {
    1: VerdictLabel.NO_SIGNIFICANT_FAILURE,
    2: VerdictLabel.SIM_ASSET,
    3: VerdictLabel.INFRA,
    4: VerdictLabel.CODE,
    5: VerdictLabel.RENDERING_MISMATCH,
    6: VerdictLabel.VISION_FACTOR,
    7: VerdictLabel.VISION_INTERACTING,
    8: VerdictLabel.ACTION_GAP,
    9: VerdictLabel.PERCEPTION_UNRESOLVED,
    10: VerdictLabel.UNDETERMINED,
}

DESTINATION_BY_ROW: Mapping[int, Destination] = {
    1: Destination.STOP,
    2: Destination.REPORT,
    3: Destination.REPORT,
    4: Destination.HAGENT,
    5: Destination.CONFIG_FIX,
    6: Destination.DATA_LADDER,
    7: Destination.DATA_LADDER,
    8: Destination.REPORT,
    9: Destination.REPORT,
    10: Destination.REPORT,
}


def inconclusive_replays(replays: Union[Mapping[str, ReplayResult], Iterable[ReplayResult]]) -> tuple[str, ...]:
    """Replay ids whose interval straddles a boundary (the router extends
    their seeds before calling ``decide`` again)."""
    return tuple(sorted(r.replay_id for r in _normalize(replays).values() if r.outcome is Outcome.INCONCLUSIVE))


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #

class _Tri(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


# replay states
_RESTORED, _PARTIAL, _NONE, _UNKNOWN, _MISSING = "restored", "partial", "none", "unknown", "missing"
_FAILS = (_PARTIAL, _NONE)          # a reference controller that is not restored has failed


class _Branch(Enum):
    C = "C"
    G = "G"
    G_DEGRADED = "G_degraded"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class _Match:
    tri: _Tri
    evidence_ids: tuple[str, ...] = ()
    replay_ids: tuple[str, ...] = ()
    factor: Optional[str] = None
    location: Optional[str] = None
    by_elimination: bool = False
    partial: bool = False          # the winning replay only partly restores
    conflict: str = ""             # non-empty: matched, but the evidence conflicts
    note: str = ""


_FALSE = _Match(_Tri.FALSE)


def _unknown(note: str) -> _Match:
    return _Match(_Tri.UNKNOWN, note=note)


def _normalize(replays: Union[Mapping[str, ReplayResult], Iterable[ReplayResult]]) -> dict[str, ReplayResult]:
    items = list(replays.values()) if isinstance(replays, Mapping) else list(replays)
    out: dict[str, ReplayResult] = {}
    for r in items:
        if r.replay_id in out:
            raise ValueError(f"duplicate replay id in table: {r.replay_id}")
        if r.outcome is None:
            raise ValueError(f"replay {r.replay_id} has no outcome; classify it with ladder.stats first")
        out[r.replay_id] = r
    if isinstance(replays, Mapping):
        for key, r in replays.items():
            if key != r.replay_id:
                raise ValueError(f"replay table key {key!r} does not match replay_id {r.replay_id!r}")
    return out


def _ids(evs: Iterable[Evidence]) -> tuple[str, ...]:
    return tuple(e.id for e in sorted(evs, key=lambda e: (-e.strength, e.id)))


class _Ctx:
    def __init__(self, evidence: Sequence[Evidence], replays: dict[str, ReplayResult], cfg: DecideConfig):
        seen: set[str] = set()
        for e in evidence:
            if e.id in seen:
                raise ValueError(f"duplicate evidence id: {e.id}")
            seen.add(e.id)
        self.all_evidence = tuple(evidence)
        self.evidence = [e for e in evidence if e.strength >= cfg.min_strength]
        self.replays = replays

    def ev(self, signal: str) -> list[Evidence]:
        return sorted((e for e in self.evidence if e.signal == signal), key=lambda e: (-e.strength, e.id))

    def phases(self) -> set[str]:
        p = V.SIG_FAILURE_PHASE_PREFIX
        return {e.signal[len(p):] for e in self.evidence if e.signal.startswith(p)}

    def state(self, rid: str) -> str:
        r = self.replays.get(rid)
        if r is None:
            return _MISSING
        return {Outcome.RESTORED: _RESTORED, Outcome.PARTIAL: _PARTIAL,
                Outcome.NO_EFFECT: _NONE, Outcome.INCONCLUSIVE: _UNKNOWN}[r.outcome]

    @property
    def branch(self) -> _Branch:
        return {_RESTORED: _Branch.C, _NONE: _Branch.G, _PARTIAL: _Branch.G_DEGRADED}.get(
            self.state(V.R_STAGE2_NOMINAL), _Branch.UNKNOWN)

    @property
    def factors(self) -> Optional[tuple[str, ...]]:
        r = self.replays.get(V.R_STAGE0)
        return None if r is None else r.scene.factor_names


def _strongest_location(evs: Sequence[Evidence]) -> Optional[str]:
    for e in sorted(evs, key=lambda e: (-e.strength, e.id)):
        if e.location:
            return e.location
    return None


# --------------------------------------------------------------------------- #
# rows
# --------------------------------------------------------------------------- #

def _row1(c: _Ctx) -> _Match:
    """Stage 0 shows no real drop. Variable staleness blocks it: a stale-server
    fault also passes in clean mode and belongs to row 3 (user-confirmed)."""
    s = c.state(V.R_STAGE0)
    if s in (_MISSING, _UNKNOWN):
        return _unknown(f"stage0 {'missing' if s == _MISSING else 'inconclusive'}")
    if s == _RESTORED and not c.ev(V.SIG_STALENESS_VARIABLE):
        return _Match(_Tri.TRUE, replay_ids=(V.R_STAGE0,))
    return _FALSE


def _row2(c: _Ctx) -> _Match:
    """Sim / asset / infeasible: broken render or physics, a failing reference
    controller (G2, C1) or an unqualified reference controller."""
    ev = c.ev(V.SIG_RENDER_PHYSICS_BROKEN) + c.ev(V.SIG_REFERENCE_UNQUALIFIED)
    failed, unknown = [], []
    for rid in (V.R_G2, V.R_C1):
        s = c.state(rid)
        if s in _FAILS:
            failed.append(rid)
        elif s == _UNKNOWN:
            unknown.append(rid)
    if ev or failed:
        return _Match(_Tri.TRUE, evidence_ids=_ids(ev), replay_ids=tuple(failed),
                      note="reference controller failed: " + ",".join(failed) if failed else "")
    if unknown:
        return _unknown("reference replay inconclusive: " + ",".join(unknown))
    need = {_Branch.G: (V.R_G2,), _Branch.G_DEGRADED: (V.R_G2,), _Branch.C: (V.R_C1,),
            _Branch.UNKNOWN: (V.R_G2, V.R_C1)}[c.branch]
    missing = [rid for rid in need if c.state(rid) == _MISSING]
    if missing:
        return _unknown("reference replay missing: " + ",".join(missing))
    return _FALSE


def _row3(c: _Ctx) -> _Match:
    """Infra: variable staleness, and clean mode restores (fully or partly)."""
    sv = c.ev(V.SIG_STALENESS_VARIABLE)
    if not sv:
        return _FALSE
    s = c.state(V.R_STAGE0)
    if s in (_MISSING, _UNKNOWN):
        return _unknown(f"stage0 {'missing' if s == _MISSING else 'inconclusive'}; needed for the clean-mode restore")
    if s == _NONE:
        return _FALSE
    return _Match(_Tri.TRUE, evidence_ids=_ids(sv), replay_ids=(V.R_STAGE0,), partial=(s == _PARTIAL))


def _row4(c: _Ctx) -> _Match:
    """Code / config. Located evidence first; Branch G by elimination last."""
    located = (c.ev(V.SIG_CONTRACT_VIOLATION) + c.ev(V.SIG_STALENESS_CONSTANT)
               + c.ev(V.SIG_IMPLIED_TARGET_OFFSET))
    g2, g3 = c.state(V.R_G2), c.state(V.R_G3)
    g3_fail = g2 == _RESTORED and g3 in _FAILS          # reference fine raw, broken through the wrapper
    g3_unknown = g3 == _UNKNOWN or (g3 in _FAILS and g2 == _UNKNOWN)
    proving = tuple(r for r in (V.R_G2, V.R_G3) if c.state(r) != _MISSING)

    if located or g3_fail:
        loc = _strongest_location(located) or ("action_side" if g3_fail else None)
        return _Match(_Tri.TRUE, evidence_ids=_ids(located), replay_ids=proving if g3_fail else (),
                      location=loc, note="located cause")
    if c.branch in (_Branch.G, _Branch.G_DEGRADED):
        both_ok = g2 == _RESTORED and g3 == _RESTORED
        return _Match(_Tri.TRUE, replay_ids=proving, by_elimination=True,
                      location="observation_side_or_policy_loading" if both_ok else None,
                      note="Branch G, rows 2-3 excluded, no located cause (elimination only)")
    if c.branch is _Branch.UNKNOWN:
        return _unknown("stage2_nominal missing or inconclusive; cannot tell Branch C from G")
    if g3_unknown:
        return _unknown("G3 inconclusive")
    return _FALSE


def _row5(c: _Ctx) -> _Match:
    """Rendering mismatch: eval render settings differ and C_render restores."""
    differ = c.ev(V.SIG_RENDER_SETTINGS_DIFFER)
    if not differ:
        return _FALSE
    s = c.state(V.R_C_RENDER)
    if s in (_MISSING, _UNKNOWN):
        return _unknown(f"C_render {'missing' if s == _MISSING else 'inconclusive'}")
    if s == _NONE:
        return _FALSE
    return _Match(_Tri.TRUE, evidence_ids=_ids(differ), replay_ids=(V.R_C_RENDER,),
                  location=_strongest_location(differ), partial=(s == _PARTIAL))


@dataclass(frozen=True)
class _Restores:
    kind: str                      # single | interacting | none | conflict | unknown
    factors: tuple[str, ...] = ()
    partial: bool = False
    replay_ids: tuple[str, ...] = ()
    note: str = ""


def _restores(c: _Ctx) -> _Restores:
    """Summarise the one-factor and all-factors restores (3.2.7)."""
    factors = c.factors
    if factors is None:
        return _Restores("unknown", note="stage0 missing; factor list unknown")
    rids = {f: V.restore_id(f) for f in factors}
    st = {f: c.state(r) for f, r in rids.items()}
    restored = [f for f in factors if st[f] == _RESTORED]
    partial = [f for f in factors if st[f] == _PARTIAL]
    unk = [rids[f] for f in factors if st[f] in (_UNKNOWN, _MISSING)]
    all_s = c.state(V.R_C_ALL) if len(factors) >= 2 else None
    shown = tuple(rids[f] for f in factors) + ((V.R_C_ALL,) if all_s is not None and all_s != _MISSING else ())

    if len(restored) >= 2:
        return _Restores("conflict", tuple(restored), replay_ids=tuple(rids[f] for f in restored),
                         note=f"factors {restored} each restore on their own; the factor cannot be named")
    if len(restored) == 1:
        if unk:
            return _Restores("unknown", note="other single-factor restores missing or inconclusive: " + ",".join(unk))
        f = restored[0]
        return _Restores("single", (f,), replay_ids=(rids[f],))
    if unk:
        return _Restores("unknown", note="single-factor restores missing or inconclusive: " + ",".join(unk))
    if all_s == _RESTORED:
        return _Restores("interacting", factors, replay_ids=shown)
    if all_s == _UNKNOWN or all_s == _MISSING:
        return _Restores("unknown", note=f"C_all {all_s}")
    if len(partial) == 1:
        f = partial[0]
        return _Restores("single", (f,), partial=True, replay_ids=(rids[f],))
    if len(partial) >= 2:
        return _Restores("conflict", tuple(partial), replay_ids=tuple(rids[f] for f in partial),
                         note=f"factors {partial} each restore only partly; the factor cannot be named")
    if all_s == _PARTIAL:
        return _Restores("interacting", factors, partial=True, replay_ids=shown)
    return _Restores("none", replay_ids=shown)


def _row6(c: _Ctx) -> _Match:
    r = _restores(c)
    if r.kind == "unknown":
        return _unknown(r.note)
    if r.kind == "single":
        return _Match(_Tri.TRUE, replay_ids=r.replay_ids, factor=r.factors[0], partial=r.partial)
    if r.kind == "conflict":
        return _Match(_Tri.TRUE, replay_ids=r.replay_ids, conflict=r.note)
    return _FALSE


def _row7(c: _Ctx) -> _Match:
    r = _restores(c)
    if r.kind == "unknown":
        return _unknown(r.note)
    if r.kind == "interacting":
        return _Match(_Tri.TRUE, replay_ids=r.replay_ids, factor=";".join(r.factors), partial=r.partial)
    return _FALSE


def _nothing_restores(c: _Ctx) -> Optional[_Match]:
    """None if nothing restores; otherwise the FALSE/UNKNOWN match for rows 8-9."""
    r = _restores(c)
    if r.kind == "unknown":
        return _unknown(r.note)
    if r.kind != "none":
        return _FALSE
    return None


def _row8(c: _Ctx) -> _Match:
    blocked = _nothing_restores(c)
    if blocked is not None:
        return blocked
    ok, wrong = c.ev(V.SIG_IMPLIED_TARGET_OK), c.ev(V.SIG_IMPLIED_TARGET_WRONG)
    if ok and wrong:
        return _Match(_Tri.TRUE, evidence_ids=_ids(ok + wrong),
                      conflict="implied target reported both right and wrong")
    if ok and c.phases() & set(V.PHASES_ACTION_GAP):
        phases = sorted(c.phases() & set(V.PHASES_ACTION_GAP))
        return _Match(_Tri.TRUE, evidence_ids=_ids(ok + [e for e in c.evidence if e.signal.startswith(V.SIG_FAILURE_PHASE_PREFIX)]),
                      note="failure at " + ",".join(phases))
    return _FALSE


def _row9(c: _Ctx) -> _Match:
    blocked = _nothing_restores(c)
    if blocked is not None:
        return blocked
    wrong = c.ev(V.SIG_IMPLIED_TARGET_WRONG)
    if wrong and not c.ev(V.SIG_IMPLIED_TARGET_OK):
        return _Match(_Tri.TRUE, evidence_ids=_ids(wrong))
    return _FALSE


_ROWS: tuple[Callable[[_Ctx], _Match], ...] = (_row1, _row2, _row3, _row4, _row5, _row6, _row7, _row8, _row9)


# --------------------------------------------------------------------------- #
# decide
# --------------------------------------------------------------------------- #

def decide(evidence: Sequence[Evidence],
           replays: Union[Mapping[str, ReplayResult], Iterable[ReplayResult]],
           config: Optional[DecideConfig] = None) -> Verdict:
    cfg = config or DecideConfig()
    table = _normalize(replays)
    c = _Ctx(tuple(evidence), table, cfg)
    matches = [row(c) for row in _ROWS]
    multi = c.branch is _Branch.G_DEGRADED

    unknown_rows: list[int] = []
    for i, m in enumerate(matches, start=1):
        if m.tri is _Tri.UNKNOWN:
            unknown_rows.append(i)
            continue
        if m.tri is not _Tri.TRUE:
            continue
        if unknown_rows or m.conflict:
            return _undetermined(c, table, matches, unknown_rows, m, i, multi)
        return _winner(i, m, matches, table, multi and i != 1)
    return _undetermined(c, table, matches, unknown_rows, None, None, multi)


def _winner(row: int, m: _Match, matches: Sequence[_Match], table: dict[str, ReplayResult],
            multi: bool) -> Verdict:
    second = m.partial
    if row in (3, 4):
        for j in range(row + 1, 8):
            mj = matches[j - 1]
            if mj.tri is _Tri.TRUE and (mj.partial or (row == 3 and j == 4 and not mj.by_elimination)):
                second = True
    incomplete = sorted({matches[j - 1].note for j in range(row + 1, 8)
                         if row in (3, 4) and matches[j - 1].tri is _Tri.UNKNOWN})
    conf = ""
    if row == 4:
        conf = "confidence=low (elimination only)" if m.by_elimination else "confidence=higher (located cause)"
    notes = "; ".join(x for x in (f"row {row}", conf, m.note,
                                  "second cause suspected" if second else "",
                                  ("second-cause check incomplete: " + " | ".join(incomplete)) if incomplete else "",
                                  "Stage 2 degraded: likely multi-cause" if multi else "") if x)
    return Verdict(
        label=LABEL_BY_ROW[row], destination=DESTINATION_BY_ROW[row], row=row,
        evidence_ids=m.evidence_ids, factor=m.factor, location=m.location,
        by_elimination=m.by_elimination, likely_second_cause=second, multi_cause_flag=multi,
        replays=tuple(table[r] for r in m.replay_ids if r in table), notes=notes)


def _undetermined(c: _Ctx, table: dict[str, ReplayResult], matches: Sequence[_Match],
                  unknown_rows: Sequence[int], won: Optional[_Match], won_row: Optional[int],
                  multi: bool) -> Verdict:
    reasons = [f"row {i} undecided: {matches[i - 1].note}" for i in unknown_rows]
    if won is not None and won.conflict:
        reasons.append(f"row {won_row} conflict: {won.conflict}")
    elif won is not None:
        reasons.append(f"row {won_row} would match but a higher-precedence row is undecided")
    if not reasons:
        reasons.append("no row matched")
    extend = inconclusive_replays(table)
    if extend:
        reasons.append("extend seeds for: " + ",".join(extend))
    return Verdict(
        label=VerdictLabel.UNDETERMINED, destination=DESTINATION_BY_ROW[10], row=10,
        evidence_ids=_ids(c.all_evidence), multi_cause_flag=multi,
        notes="row 10; " + "; ".join(reasons))
