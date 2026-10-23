"""Evidence-source registry and runner for Stage 1 (plan.md 3.2.3, 3.2.4).

Registers the five Stage 1 checks as evidence sources and runs them on an
incident. Its output is the ``extra_evidence`` that ``router.route`` takes.
This module does not touch ``route``, the replay stages or ``decide``: the
signals the checks emit are already in ``router.vocab``.

    result = run_stage1(incident, checks)
    route(incident, backend, config, extra_evidence=result.evidence)

A check that cannot read its inputs (``TraceUnavailable``) is reported in
``skipped``, not silently dropped: an empty evidence list means "checked,
nothing found" only for the sources not listed there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from router.prereq.trace import TraceUnavailable
from router.schemas import Evidence, Incident
from router.stage1.base import Check, log

EVIDENCE_SOURCES = ("contract_cfg", "contract_obs", "contract_action", "staleness", "sanity")


@dataclass(frozen=True)
class Stage1Result:
    evidence: tuple[Evidence, ...]
    skipped: tuple[tuple[str, str], ...] = ()     # (source, reason)

    @property
    def sources_run(self) -> tuple[str, ...]:
        skipped = {s for s, _ in self.skipped}
        return tuple(s for s in EVIDENCE_SOURCES if s not in skipped)


def run_stage1(incident: Incident, checks: Sequence[Check]) -> Stage1Result:
    sources = [c.source for c in checks]
    unknown = sorted(set(sources) - set(EVIDENCE_SOURCES))
    if unknown:
        raise ValueError(f"checks with unregistered evidence sources: {unknown}")
    if len(set(sources)) != len(sources):
        raise ValueError(f"two checks share an evidence source: {sources}")

    evidence: list[Evidence] = []
    skipped: list[tuple[str, str]] = []
    for check in checks:
        try:
            found = check.run(incident)
        except TraceUnavailable as exc:
            log.warning("stage 1: %s skipped: %s", check.source, exc)
            skipped.append((check.source, str(exc)))
            continue
        wrong = [e.id for e in found if e.source != check.source]
        if wrong:
            raise ValueError(f"check {check.source!r} emitted evidence from another source: {wrong}")
        evidence += found

    ids = [e.id for e in evidence]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate evidence ids from Stage 1: {dup}")
    return Stage1Result(tuple(evidence), tuple(skipped))
