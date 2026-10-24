"""Common pieces of the Stage 1 checks (plan.md 3.2.4, 3.2.14).

Every check is a module with the same interface:

    check.run(incident) -> list[Evidence]

A check emits a record only when it fires; an empty list means "checked,
nothing found" (``router.vocab``). A check that cannot read its inputs raises
``TraceUnavailable`` instead of returning an empty list.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Sequence

from router.schemas import Evidence, Incident

log = logging.getLogger("router.stage1")


class Check(ABC):
    """One Stage 1 check. ``source`` is the ``Evidence.source`` it emits and
    must be one of ``router.stage1.registry.EVIDENCE_SOURCES``."""

    source: ClassVar[str]

    @abstractmethod
    def run(self, incident: Incident) -> list[Evidence]:
        ...

    def run_partial(self, incident: Incident) -> tuple[list[Evidence], list[str]]:
        """``run`` plus the parts the check could not look at although it ran:
        (evidence, reasons). ``run_stage1`` records each reason as a partial
        skip, so a half-blind check never reads as "checked, nothing found".
        Checks that can be partly blind override this."""
        return self.run(incident), []


def saturating(ratio: float) -> float:
    """Map a ratio ``deviation / threshold`` to [0, 1): 0 at 0, 0.5 at 1, ->1.

    The one place a deviation becomes a strength, so strengths from different
    checks are comparable: 0.5 means "just past the firing threshold".
    """
    r = max(0.0, float(ratio))
    if r > 1e100:
        return 1.0
    return r * r / (1.0 + r * r)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Pearson correlation, or None when either series is constant or the
    series are too short or differ in length."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mx, my = math.fsum(xs) / n, math.fsum(ys) / n
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    syy = math.fsum((y - my) ** 2 for y in ys)
    if sxx <= 1e-18 or syy <= 1e-18:
        return None
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))


def nrmse(cmd: Sequence[float], actual: Sequence[float], min_range: float = 1e-9) -> Optional[float]:
    """RMS(cmd - actual) divided by the range of ``cmd``; None if ``cmd`` is
    (nearly) constant or the series differ in length."""
    n = len(cmd)
    if n == 0 or n != len(actual):
        return None
    span = max(cmd) - min(cmd)
    if span < min_range:
        return None
    rms = math.sqrt(math.fsum((c - a) ** 2 for c, a in zip(cmd, actual)) / n)
    return rms / span
