"""Router orchestrator (plan.md 3.2.3): run the stages, collect the evidence
and the replay table, call ``decide``.

    route(incident, backend, config, inputs, extra_evidence=()) -> RouterResult

Order:
  Stage 0     reproduce in clean mode.
  Stage 1     checks from logs alone (``inputs.stage1_checks``), on every
              incident. They run before the Stage 0 stop decision because
              ``staleness_variable`` is what keeps a clean-mode pass of a
              stale-server incident from closing as row 1.
              Stop unless row 1 is FALSE (``decide.row1_state``): TRUE is
              "no significant failure"; UNKNOWN (Stage 0 inconclusive at its
              20-seed cap, or staleness_variable not checked) makes every
              lower row row 10 whatever else runs.
  Stage 2     branch split on the nominal scene; ``stage2.branches_to_run``.
  Branch G    G1 (when observed in incident mode), G2, G3. G3 may be
              unavailable on a backend without a reference controller that
              runs through the wrapper (``ladder.backend.ReplayUnavailable``):
              recorded in ``RouterResult.unavailable`` and passed to ``decide``
              as ``unavailable`` (router change 4).
  Branch C    C_behavior (logs only, ``inputs.behavior_source``), C1, every
              one-factor restore, C_all when needed, C_render when needed.
  Stage 3     ``decide(evidence, replays)`` over everything that ran.

Evidence that ``decide`` sees: Stage 1 + C_behavior + ``extra_evidence``. The
last is externally supplied and recorded separately
(``RouterResult.external_evidence``). A check that could not run is recorded
in ``skipped`` / ``partial`` with its reason and named in the verdict notes;
it never reads as "checked, nothing found": its signals go to ``decide`` as
``unchecked`` (``vocab.SIGNALS_BY_SOURCE``), and a row that could only be
ruled out by them becomes UNKNOWN (``vocab.ROW_SIGNALS``).

No early stop across hypotheses: once past Stage 0, every replay of each
branch that runs is executed before ``decide`` is called once. All replays use
one seed set and one cache. No model is called anywhere in this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence

from ladder.backend import RolloutBackend
from router.behavior import SOURCE as BEHAVIOR_SOURCE
from router.behavior import BehaviorConfig, run_behavior_analysis
from router.branches.branch_c import run_branch_c
from router.branches.branch_g import run_branch_g, run_g1
from router import vocab as V
from router.decide import DecideConfig, decide, row1_state
from router.prereq.behavior_trace import BehaviorSource
from router.prereq.rollout_cache import CachedBackend, RolloutCache
from router.prereq.trace import TraceUnavailable
from router.replay import ReplayConfig, ReplayRunner, seed_set
from router.schemas import Evidence, Incident, Mode, ReplayResult, Verdict
from router.stage0 import failure_only_in_incident_mode, run_stage0
from router.stage1.base import Check
from router.stage1.nominal import NominalStats
from router.stage1.registry import run_stage1
from router.stage2 import Branch, branch_of, branches_to_run, run_stage2

log = logging.getLogger("router")


@dataclass(frozen=True)
class RouterConfig:
    """``reference_checkpoint``: version id of the qualified reference
    controller (3.2.2). ``render_config_key``: the eval-config key holding the
    render settings (TODO(verify) per policy). ``behavior``: C_behavior
    thresholds and trace mode."""
    replay: ReplayConfig
    reference_checkpoint: str
    render_config_key: str = "render"
    decide: DecideConfig = field(default_factory=DecideConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)


@dataclass(frozen=True)
class RouterInputs:
    """Where the log-only evidence comes from: the Stage 1 checks (each holds
    its trace source and nominal stats) and the C_behavior source with its
    nominal stats.

    Kept apart from ``RouterConfig``: the config is plain values (thresholds,
    sizes) shared through ``LoopConfig``; these are handles to log stores plus
    calibration data for one scene condition (DECISIONS.md, Fix F1). Anything
    left out is recorded as skipped, never as clean."""
    stage1_checks: tuple[Check, ...] = ()
    behavior_source: Optional[BehaviorSource] = None
    behavior_nominal: Optional[NominalStats] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage1_checks", tuple(self.stage1_checks))
        if (self.behavior_source is None) != (self.behavior_nominal is None):
            raise ValueError("behavior_source and behavior_nominal must be given together")


@dataclass(frozen=True)
class RouterResult:
    incident_id: str
    verdict: Verdict
    replays: tuple[ReplayResult, ...]          # every replay that ran, in run order
    evidence: tuple[Evidence, ...]             # everything decide saw: stage1 + behavior + external
    seeds: tuple[int, ...]
    stages: tuple[str, ...]                    # e.g. ("stage0", "stage1", "stage2", "C_behavior", "branch_C")
    branch: Optional[Branch]                   # None when Stage 0 stopped the router
    episodes_simulated: int                    # new episodes run on the backend (cache misses)
    episodes_used: int                         # episodes in the replay table, cache hits included
    cache_hits: int
    extended: tuple[str, ...] = ()             # replays extended from n_initial to n_max
    notes: tuple[str, ...] = ()
    stage1_evidence: tuple[Evidence, ...] = ()
    behavior_evidence: tuple[Evidence, ...] = ()
    external_evidence: tuple[Evidence, ...] = ()   # extra_evidence: supplied by the caller, not by a check
    sources_run: tuple[str, ...] = ()          # Stage 1 sources and "behavior" that ran
    skipped: tuple[tuple[str, str], ...] = ()  # (source, reason): did not run
    partial: tuple[tuple[str, V.PartialSkip], ...] = ()  # ran, part of it not checked
    unchecked: tuple[str, ...] = ()            # signals passed to decide as not checked
    unavailable: tuple[tuple[str, str], ...] = ()  # (replay id, reason): the backend could not run it

    def episodes_by_replay(self) -> dict[str, int]:
        return {r.replay_id: r.n for r in self.replays}

    def replay(self, replay_id: str) -> Optional[ReplayResult]:
        return next((r for r in self.replays if r.replay_id == replay_id), None)


def _with_skip_notes(verdict: Verdict, skipped: Sequence[tuple[str, str]],
                     partial: Sequence[tuple[str, V.PartialSkip]],
                     unavailable: Optional[Mapping[str, str]] = None) -> Verdict:
    """Name every check that did not (fully) run, and every replay the backend
    could not run, in the verdict notes. Only the notes change: row, label,
    destination and flags are decide's."""
    extra = []
    if unavailable:
        extra.append("replays unavailable: " + " | ".join(f"{r} ({why})" for r, why in unavailable.items()))
    if skipped:
        extra.append("checks skipped: " + " | ".join(f"{s} ({why})" for s, why in skipped))
    if partial:
        extra.append("checks partly skipped: " + " | ".join(f"{s} ({skip.reason})" for s, skip in partial))
    if not extra:
        return verdict
    return replace(verdict, notes="; ".join([verdict.notes] + extra) if verdict.notes else "; ".join(extra))


def unchecked_signals(incident: Incident, skipped: Sequence[tuple[str, str]],
                      partial: Sequence[tuple[str, V.PartialSkip]]) -> frozenset[str]:
    """Signals that a skipped check would have emitted (all of its source's) and
    the signals each partial skip names. ``staleness_variable`` only applies to
    an incident observed in incident mode."""
    out = set()
    for source, _ in skipped:
        out |= V.SIGNALS_BY_SOURCE.get(source, frozenset())
    for _, skip in partial:
        out |= skip.signals
    if incident.observed_mode is not Mode.INCIDENT:
        out.discard(V.SIG_STALENESS_VARIABLE)
    return frozenset(out)


def _run_c_behavior(incident: Incident, inputs: RouterInputs, cfg: BehaviorConfig
                    ) -> tuple[tuple[Evidence, ...], list[str], Optional[str], tuple[V.PartialSkip, ...]]:
    """(evidence, notes, skip reason or None, parts that were not judged)."""
    if inputs.behavior_source is None:
        return (), [], "no behavior source configured", ()
    try:
        a = run_behavior_analysis(incident, inputs.behavior_source, inputs.behavior_nominal, cfg)
    except TraceUnavailable as exc:
        log.warning("C_behavior skipped: %s", exc)
        return (), [], str(exc), ()
    return a.evidence, [f"C_behavior: {n}" for n in a.notes], None, a.partial


def route(incident: Incident, backend: RolloutBackend, config: RouterConfig,
          inputs: Optional[RouterInputs] = None, *, extra_evidence: Sequence[Evidence] = ()) -> RouterResult:
    if inputs is not None and not isinstance(inputs, RouterInputs):
        raise TypeError(f"inputs must be RouterInputs, got {type(inputs).__name__} "
                        "(pass hand-made evidence as extra_evidence=...)")
    inputs = inputs or RouterInputs()
    cached = backend if isinstance(backend, CachedBackend) else CachedBackend(backend, RolloutCache())
    misses0, hits0 = cached.misses, cached.hits
    external = tuple(extra_evidence)
    seeds = seed_set(incident.failing_seeds, config.replay.n_max)
    runner = ReplayRunner(cached, seeds, config.replay)
    stages, notes = ["stage0", "stage1"], []
    branch: Optional[Branch] = None

    stage0 = run_stage0(runner, incident)
    s1 = run_stage1(incident, inputs.stage1_checks)
    skipped, partial, sources_run = list(s1.skipped), list(s1.partial), list(s1.sources_run)
    behavior_ev: tuple[Evidence, ...] = ()
    evidence = s1.evidence + external
    notes += s1.notes

    unchecked = unchecked_signals(incident, skipped, partial)
    if row1_state(evidence, [stage0], config.decide, unchecked) != "false":
        # Row 1 TRUE wins; row 1 UNKNOWN blocks every lower row, so any
        # further replay is wasted.
        verdict = decide(evidence, [stage0], config.decide, unchecked)
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
            # C_behavior belongs to Branch C (plan 3.2.7): rows 8-9 need "nothing restores".
            stages.append("C_behavior")
            behavior_ev, b_notes, b_skip, b_partial = _run_c_behavior(incident, inputs, config.behavior)
            notes += b_notes
            if b_skip is None:
                sources_run.append(BEHAVIOR_SOURCE)
                partial += [(BEHAVIOR_SOURCE, skip) for skip in b_partial]
            else:
                skipped.append((BEHAVIOR_SOURCE, b_skip))
            unchecked = unchecked_signals(incident, skipped, partial)
            evidence += behavior_ev
            stages.append("branch_C")
            _, c_notes = run_branch_c(runner, incident, config.reference_checkpoint, evidence,
                                      config.render_config_key)
            notes += c_notes
        verdict = decide(evidence, runner.results, config.decide, unchecked, runner.unavailable)

    verdict = _with_skip_notes(verdict, skipped, partial, runner.unavailable)
    result = RouterResult(
        incident_id=incident.incident_id, verdict=verdict, replays=tuple(runner.results),
        evidence=evidence, seeds=seeds, stages=tuple(stages), branch=branch,
        episodes_simulated=cached.misses - misses0, episodes_used=runner.episodes_used,
        cache_hits=cached.hits - hits0, extended=tuple(runner.extended), notes=tuple(notes),
        stage1_evidence=s1.evidence, behavior_evidence=behavior_ev, external_evidence=external,
        sources_run=tuple(sources_run), skipped=tuple(skipped), partial=tuple(partial),
        unchecked=tuple(sorted(unchecked)), unavailable=tuple(runner.unavailable.items()))
    log.info("incident %s: verdict=%s row=%d branch=%s replays=%d episodes_simulated=%d "
             "episodes_used=%d cache_hits=%d extended=%s skipped=%s",
             incident.incident_id, verdict.label.value, verdict.row,
             branch.value if branch else "-", len(result.replays), result.episodes_simulated,
             result.episodes_used, result.cache_hits, ",".join(result.extended) or "-",
             ",".join(s for s, _ in skipped + partial) or "-")
    return result
