"""Router orchestrator (plan.md 3.2.3): run the stages, collect the evidence
and the replay table, call ``decide``.

    route(incident, backend, config, extra_evidence=()) -> RouterResult

Order:
  Stage 0   reproduce in clean mode. Stop if ``decide`` on Stage 0 alone says
            row 1 (no significant failure), or if Stage 0 is still
            inconclusive at its 20-seed cap (the verdict is then row 10
            whatever else runs).
  Stage 1   NOT built in this step. Its records (contract, staleness, render
            and physics sanity) and the C_behavior records (implied target,
            failure phase) arrive as ``extra_evidence``. Their signals gate
            C_render and feed ``decide`` unchanged.
  Stage 2   branch split on the nominal scene; ``stage2.branches_to_run``.
  Branch G  G1 (when observed in incident mode), G2, G3.
  Branch C  C1, every one-factor restore, C_all when needed, C_render when needed.
  Stage 3   ``decide(evidence, replays)`` over everything that ran.

No early stop across hypotheses: once past Stage 0, every replay of each
branch that runs is executed before ``decide`` is called once. All replays use
one seed set and one cache. Not here (step 7): the re-run loop and pass cap
(3.2.11). No model is called anywhere in this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ladder.backend import RolloutBackend
from router.branches.branch_c import run_branch_c
from router.branches.branch_g import run_branch_g, run_g1
from router.decide import DecideConfig, decide
from router.prereq.rollout_cache import CachedBackend, RolloutCache
from router.replay import ReplayConfig, ReplayRunner, seed_set
from router.schemas import Evidence, Incident, Outcome, ReplayResult, Verdict
from router.stage0 import failure_only_in_incident_mode, run_stage0
from router.stage2 import Branch, branch_of, branches_to_run, run_stage2

log = logging.getLogger("router")


@dataclass(frozen=True)
class RouterConfig:
    """``reference_checkpoint``: version id of the qualified reference
    controller (3.2.2). ``render_config_key``: the eval-config key holding the
    render settings (TODO(verify) per policy)."""
    replay: ReplayConfig
    reference_checkpoint: str
    render_config_key: str = "render"
    decide: DecideConfig = field(default_factory=DecideConfig)


@dataclass(frozen=True)
class RouterResult:
    incident_id: str
    verdict: Verdict
    replays: tuple[ReplayResult, ...]          # every replay that ran, in run order
    evidence: tuple[Evidence, ...]
    seeds: tuple[int, ...]
    stages: tuple[str, ...]                    # e.g. ("stage0", "stage2", "branch_C")
    branch: Optional[Branch]                   # None when Stage 0 stopped the router
    episodes_simulated: int                    # new episodes run on the backend (cache misses)
    episodes_used: int                         # episodes in the replay table, cache hits included
    cache_hits: int
    extended: tuple[str, ...] = ()             # replays extended from n_initial to n_max
    notes: tuple[str, ...] = ()

    def episodes_by_replay(self) -> dict[str, int]:
        return {r.replay_id: r.n for r in self.replays}

    def replay(self, replay_id: str) -> Optional[ReplayResult]:
        return next((r for r in self.replays if r.replay_id == replay_id), None)


def route(incident: Incident, backend: RolloutBackend, config: RouterConfig,
          extra_evidence: Sequence[Evidence] = ()) -> RouterResult:
    cached = backend if isinstance(backend, CachedBackend) else CachedBackend(backend, RolloutCache())
    misses0, hits0 = cached.misses, cached.hits
    evidence = tuple(extra_evidence)
    seeds = seed_set(incident.failing_seeds, config.replay.n_max)
    runner = ReplayRunner(cached, seeds, config.replay)
    stages, notes = ["stage0"], []
    branch: Optional[Branch] = None

    stage0 = run_stage0(runner, incident)
    early = decide(evidence, [stage0], config.decide)
    if early.row == 1 or stage0.outcome is Outcome.INCONCLUSIVE:
        # Row 1, or Stage 0 still inconclusive at its cap: row 1 is then
        # undecided and blocks every lower row, so any further replay is wasted.
        verdict = early
    else:
        incident_only = failure_only_in_incident_mode(stage0, incident)
        if incident_only:
            notes.append("failure seen only in incident mode; clean-mode Stage 0 passes")
        stages.append("stage2")
        branch = branch_of(run_stage2(runner, incident))
        run_g, run_c = branches_to_run(branch)
        if run_g:
            stages.append("branch_G")
            run_branch_g(runner, incident, config.reference_checkpoint)
        elif incident_only:
            stages.append("G1")
            run_g1(runner, incident)
        if run_c:
            stages.append("branch_C")
            _, c_notes = run_branch_c(runner, incident, config.reference_checkpoint, evidence,
                                      config.render_config_key)
            notes += c_notes
        verdict = decide(evidence, runner.results, config.decide)

    result = RouterResult(
        incident_id=incident.incident_id, verdict=verdict, replays=tuple(runner.results),
        evidence=evidence, seeds=seeds, stages=tuple(stages), branch=branch,
        episodes_simulated=cached.misses - misses0, episodes_used=runner.episodes_used,
        cache_hits=cached.hits - hits0, extended=tuple(runner.extended), notes=tuple(notes))
    log.info("incident %s: verdict=%s row=%d branch=%s replays=%d episodes_simulated=%d "
             "episodes_used=%d cache_hits=%d extended=%s",
             incident.incident_id, verdict.label.value, verdict.row,
             branch.value if branch else "-", len(result.replays), result.episodes_simulated,
             result.episodes_used, result.cache_hits, ",".join(result.extended) or "-")
    return result
