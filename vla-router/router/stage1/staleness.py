"""Stage 1 check 4: staleness probe (plan.md 3.2.4, 6.4).

"From the logged (observation step, applied step) pairs, compute the age of
each applied action. A constant extra delay that is also present in clean mode
is code latency. A variable delay that grows with load and disappears in clean
mode is server staleness."

The age of an action is ``StepPair.extra_delay``: applied step minus
observation step minus the action's offset in its chunk, so the sawtooth of
action chunking is not counted as delay.

What fires
----------
* ``staleness_constant``  the clean-mode delay is above nominal
  (``staleness/clean_mean_delay``). Anything present in clean mode is code (or
  config), whether or not it jitters; the signal feeds row 4 as a located cause
  (``controller.latency``). The detail gives the recovered extra delay in steps.
* ``staleness_variable``  the incident-mode delay is above the clean-mode delay
  (``staleness/mode_gap``): a delay that disappears in clean mode is the
  server's. It feeds row 3 (infra) together with the clean-mode Stage 0 replay,
  and blocks row 1 (``decide``). Only evaluated when the incident was observed
  in incident mode; a clean-mode incident has no server to be stale. If the
  incident-mode traces are then missing, the probe still reports the clean
  half and returns a partial skip (``run_partial``), never a silent clean.

Both can fire for one incident (a code delay plus a loaded server); ``decide``
then sends it to row 4 because clean mode does not restore.

The name ``variable`` is the vocabulary's (``router.vocab``). The test applied
is presence in clean mode, which is what separates the two causes in the plan;
the delay's spread is reported in the detail but does not gate the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, median, pstdev
from typing import Optional, Sequence

from router import vocab as V
from router.prereq.trace import EpisodeTrace, TraceSource, TraceUnavailable
from router.schemas import Evidence, Incident, Mode
from router.stage1.base import Check, log
from router.stage1.nominal import NominalStats

M_CLEAN_DELAY = "staleness/clean_mean_delay"
M_MODE_GAP = "staleness/mode_gap"
REQUIRED_METRICS = (M_CLEAN_DELAY, M_MODE_GAP)


def extra_delays(trace: EpisodeTrace) -> list[int]:
    return [p.extra_delay for p in trace.step_pairs]


def episode_mean_delays(traces: Sequence[EpisodeTrace]) -> list[float]:
    """Mean extra delay of each episode (episodes without step pairs left out)."""
    return [fmean(d) for d in (extra_delays(t) for t in traces) if d]


@dataclass(frozen=True)
class DelaySummary:
    n_episodes: int
    n_actions: int
    mean: float        # median over episodes of the episode mean
    std: float         # spread of all individual delays
    max: int


def delay_summary(traces: Sequence[EpisodeTrace]) -> DelaySummary:
    means = episode_mean_delays(traces)
    if not means:
        raise ValueError("no step pairs in the traces")
    every = [d for t in traces for d in extra_delays(t)]
    return DelaySummary(len(means), len(every), median(means), pstdev(every), max(every))


def mode_gap(clean: Sequence[EpisodeTrace], incident: Sequence[EpisodeTrace]) -> float:
    """Incident-mode minus clean-mode delay, in steps. One value per incident;
    calibrate ``staleness/mode_gap`` on the gaps of no-fault incidents."""
    return median(episode_mean_delays(incident)) - median(episode_mean_delays(clean))


def estimate_extra_delay(clean: Sequence[EpisodeTrace], nominal: NominalStats) -> float:
    """Clean-mode delay above the nominal mean, in steps: the "known delay" the
    probe recovers."""
    return median(episode_mean_delays(clean)) - nominal.metric(M_CLEAN_DELAY).mean


class StalenessProbe(Check):
    source = "staleness"

    def __init__(self, nominal: NominalStats, traces: TraceSource, *, fire_z: Optional[float] = None) -> None:
        nominal.require(REQUIRED_METRICS)
        self.nominal, self.traces, self.fire_z = nominal, traces, fire_z

    def run(self, incident: Incident) -> list[Evidence]:
        return self.run_partial(incident)[0]

    def run_partial(self, incident: Incident) -> tuple[list[Evidence], list[str]]:
        """The clean half is required (``TraceUnavailable``). The incident half
        is needed only when the incident was observed in incident mode; if its
        traces are missing, ``staleness_variable`` was not checked, and that is
        returned as a partial skip."""
        clean = self.traces.load(incident, Mode.CLEAN)           # raises TraceUnavailable: required
        if not episode_mean_delays(clean):
            raise TraceUnavailable("clean-mode traces have no (observation step, applied step) pairs")
        out: list[Evidence] = []
        cs = delay_summary(clean)

        d = self.nominal.deviation(M_CLEAN_DELAY, cs.mean, "high", self.fire_z)
        if d.fires:
            shape = "constant" if cs.std < 0.25 else f"jittery (std {cs.std:.2f})"
            out.append(Evidence(
                "s1.staleness.constant", self.source, V.SIG_STALENESS_CONSTANT, d.strength, "controller.latency",
                f"actions are applied {estimate_extra_delay(clean, self.nominal):.2f} steps later than nominal in "
                f"clean mode (mean delay {cs.mean:.2f}, nominal {self.nominal.metric(M_CLEAN_DELAY).mean:.2f}; "
                f"{shape}): code latency"))

        if incident.observed_mode is Mode.INCIDENT:
            try:
                inc = self.traces.load(incident, Mode.INCIDENT)
            except TraceUnavailable as exc:
                why = f"incident-mode traces unavailable ({exc}); server staleness ({V.SIG_STALENESS_VARIABLE}) not checked"
                log.warning("staleness: %s", why)
                return out, [why]
            if not episode_mean_delays(inc):
                why = (f"incident-mode traces have no (observation step, applied step) pairs; "
                       f"server staleness ({V.SIG_STALENESS_VARIABLE}) not checked")
                log.warning("staleness: %s", why)
                return out, [why]
            gap = mode_gap(clean, inc)
            d = self.nominal.deviation(M_MODE_GAP, gap, "high", self.fire_z)
            if d.fires:
                s = delay_summary(inc)
                out.append(Evidence(
                    "s1.staleness.variable", self.source, V.SIG_STALENESS_VARIABLE, d.strength, "policy_server",
                    f"actions are {gap:.2f} steps staler in incident mode than in clean mode (incident mean "
                    f"{s.mean:.2f}, std {s.std:.2f}, max {s.max}; clean mean {cs.mean:.2f}): server staleness"))
        return out, []
