"""Evidence-source registry and runner for Stage 1 (plan.md 3.2.3, 3.2.4).

Registers the five Stage 1 checks as evidence sources and runs them on an
incident. ``router.route`` calls it on every incident, right after Stage 0
(``RouterInputs.stage1_checks``); the signals the checks emit are in
``router.vocab``.

    result = run_stage1(incident, checks)

Nothing reads as "checked, nothing found" unless it was checked:

* ``sources_run``  the sources whose check ran (fully or partly);
* ``skipped``      (source, reason) for a check that could not read its inputs
                   (``TraceUnavailable``) and for a registered source with no
                   check configured;
* ``partial``      (source, PartialSkip) for a check that ran but could not look
                   at part of its inputs (``Check.run_partial``); the skip names the
                   signals that part would have emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from router.prereq.trace import TraceUnavailable
from router.schemas import Evidence, Incident
from router.vocab import PartialSkip
from router.stage1.base import Check, log

EVIDENCE_SOURCES = ("contract_cfg", "contract_obs", "contract_action", "staleness", "sanity")


NOT_CONFIGURED = "no check configured for this source"


@dataclass(frozen=True)
class Stage1Result:
    evidence: tuple[Evidence, ...]
    skipped: tuple[tuple[str, str], ...] = ()     # (source, reason): did not run
    partial: tuple[tuple[str, PartialSkip], ...] = ()   # ran, but part of it was not checked
    sources_run: tuple[str, ...] = ()             # ran (fully or partly), in registry order
    notes: tuple[str, ...] = ()                   # informational, from the checks that ran


def run_stage1(incident: Incident, checks: Sequence[Check]) -> Stage1Result:
    sources = [c.source for c in checks]
    unknown = sorted(set(sources) - set(EVIDENCE_SOURCES))
    if unknown:
        raise ValueError(f"checks with unregistered evidence sources: {unknown}")
    if len(set(sources)) != len(sources):
        raise ValueError(f"two checks share an evidence source: {sources}")

    evidence: list[Evidence] = []
    skipped: list[tuple[str, str]] = []
    partial: list[tuple[str, PartialSkip]] = []
    ran: set[str] = set()
    notes: list[str] = []
    for check in checks:
        try:
            found, blind = check.run_partial(incident)
        except TraceUnavailable as exc:
            log.warning("stage 1: %s skipped: %s", check.source, exc)
            skipped.append((check.source, str(exc)))
            continue
        wrong = [e.id for e in found if e.source != check.source]
        if wrong:
            raise ValueError(f"check {check.source!r} emitted evidence from another source: {wrong}")
        ran.add(check.source)
        evidence += found
        partial += [(check.source, skip) for skip in blind]
        notes += check.notes(incident)
    skipped += [(s, NOT_CONFIGURED) for s in EVIDENCE_SOURCES if s not in sources]

    ids = [e.id for e in evidence]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate evidence ids from Stage 1: {dup}")
    return Stage1Result(tuple(evidence), tuple(skipped), tuple(partial),
                        tuple(s for s in EVIDENCE_SOURCES if s in ran), tuple(notes))
