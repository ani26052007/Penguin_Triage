"""Shared replay runner (plan.md 3.2.4, 3.2.9).

Every router replay goes through one ``ReplayRunner`` per incident, so all
replays use:

* the same seed list (``seed_set``): the incident's failing seeds first, padded
  with fresh seeds. A replay at n seeds uses the first n, so every pair of
  replays shares a common prefix and comparisons stay paired across conditions;
* the same cache (``CachedBackend``): extending a replay from 10 to 20 seeds
  only simulates the 10 new seeds, and identical (scene, code, actor, mode)
  configurations under different replay ids are simulated once;
* the same anchors and outcome rule (``ladder.stats.replay_outcome``).

Extension rule: run ``n_initial`` seeds; if the outcome is INCONCLUSIVE, run
up to the cap (``n_max_stage0`` for Stage 0, ``n_max`` otherwise) and classify
again. Whatever comes back at the cap is kept,
including INCONCLUSIVE, which ``decide`` turns into row 10 with "extend seeds".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from ladder.stats import replay_outcome
from router.prereq.rollout_cache import CachedBackend
from router.schemas import ActorRef, CodeRef, Mode, Outcome, ReplayResult, SceneSpec


@dataclass(frozen=True)
class ReplayConfig:
    """Anchors and seed counts. ``failure_rate`` and ``nominal_band`` are
    per-policy calibration outputs (3.2.9, build-order step 6); there are no
    defaults on purpose.

    Seed counts: every replay starts at ``n_initial`` (10). Stage 0 extends to
    ``n_max_stage0`` (20, plan 3.2.4); every other replay extends to ``n_max``
    (30, the top of plan 3.2.9's "20-30"). See DECISIONS.md for the measured
    effect of 20 vs 30."""
    failure_rate: float
    nominal_band: tuple[float, float]
    n_initial: int = 10
    n_max: int = 30
    n_max_stage0: int = 20
    confidence: float = 0.95

    def __post_init__(self) -> None:
        lo, hi = self.nominal_band
        if not (0.0 <= self.failure_rate < lo <= hi <= 1.0):
            raise ValueError(f"need 0 <= failure_rate < band_lo <= band_hi <= 1, got "
                             f"{self.failure_rate}, {self.nominal_band}")
        if not (0 < self.n_initial <= self.n_max_stage0 <= self.n_max):
            raise ValueError("need 0 < n_initial <= n_max_stage0 <= n_max")

    def cap(self, replay_id: str) -> int:
        from router.vocab import R_STAGE0
        return self.n_max_stage0 if replay_id == R_STAGE0 else self.n_max


def seed_set(failing_seeds: Iterable[int], n: int) -> tuple[int, ...]:
    """The incident's failing seeds (deduplicated, in order), padded with the
    smallest unused seeds above them until there are ``n``."""
    out = list(dict.fromkeys(int(s) for s in failing_seeds))[:n]
    used = set(out)
    nxt = max(out, default=-1) + 1
    while len(out) < n:
        if nxt not in used:
            out.append(nxt)
            used.add(nxt)
        nxt += 1
    return tuple(out)


class ReplayRunner:
    """Runs and classifies replays for one incident; keeps them in run order."""

    def __init__(self, backend: CachedBackend, seeds: Sequence[int], cfg: ReplayConfig) -> None:
        if len(seeds) < cfg.n_max:
            raise ValueError(f"need at least n_max={cfg.n_max} seeds, got {len(seeds)}")
        if len(set(seeds)) != len(seeds):
            raise ValueError("duplicate seeds in the seed set")
        self.backend = backend
        self.seeds = tuple(int(s) for s in seeds)
        self.cfg = cfg
        self.results: list[ReplayResult] = []
        self.extended: list[str] = []

    def run(self, replay_id: str, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef,
            mode: Mode = Mode.CLEAN) -> ReplayResult:
        if any(r.replay_id == replay_id for r in self.results):
            raise ValueError(f"replay {replay_id} already run for this incident")
        n, cap = self.cfg.n_initial, self.cfg.cap(replay_id)
        while True:
            eps = self.backend.run(scene, code_ref, actor, self.seeds[:n], mode)
            outcome, prop = replay_outcome(sum(e.success for e in eps), n, self.cfg.failure_rate,
                                           self.cfg.nominal_band, self.cfg.confidence)
            if outcome is not Outcome.INCONCLUSIVE or n >= cap:
                break
            n = cap
            self.extended.append(replay_id)
        result = ReplayResult(replay_id, scene, code_ref, actor, mode, tuple(eps), outcome, prop.lo, prop.hi)
        self.results.append(result)
        return result

    def get(self, replay_id: str) -> Optional[ReplayResult]:
        return next((r for r in self.results if r.replay_id == replay_id), None)

    @property
    def episodes_used(self) -> int:
        return sum(r.n for r in self.results)
