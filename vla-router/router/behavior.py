"""C_behavior: implied target and failure phase from the logs (plan.md 3.2.7).

"No new rollouts, only the logs."

*Implied target.* Where the end effector was when the gripper closed, against
where the target object really was, over the failing episodes:

* small scatter around the right object      -> ``implied_target_ok``
* consistent offset in one direction         -> ``implied_target_offset``
  (a frame or extrinsic bug, even in Branch C: row 4 of the decision table)
* large scatter, or the wrong object         -> ``implied_target_wrong``

Exactly one of the three is emitted, or none when the data cannot say, so this
module never hands ``decide`` an ok + wrong pair (a row 10 conflict) by itself.

*Failure phase.* The first subtask (reach, grasp, lift, place, and the optional
post_place when the episodes log it) that failing episodes did not complete, by
majority: ``failure_phase:<phase>``.

Group metrics (one value per group of failing episodes, metres):

  behavior/offset_bias            |mean of (ee - target)|
  behavior/offset_scatter         RMS distance of (ee - target) from that mean
  behavior/wrong_object_fraction  share of episodes whose end effector was
                                  closer to a distractor than to the target

Strength is ``NominalStats.deviation`` like every Stage 1 check: z against the
no-fault distribution, squashed, 0.5 at the firing threshold. Two exceptions,
both stated in the evidence:

* ``implied_target_ok`` has no deviation to measure. Its strength is
  ``1 - max(strength of the three metrics)``, so it is above 0.5 and grows as
  the metrics sit deeper inside the nominal band.
* ``failure_phase`` is categorical. Its strength is the share of failing
  episodes in the modal phase, not a deviation from nominal.

The nominal samples are group-level values, so build them from groups of
no-fault episodes of about the size of the incident's failing set: a bias
computed from few episodes is noisier than one computed from many.

TODO(verify): thresholds (``fire_z``, ``min_consistency``, ``min_phase_share``)
are placeholders until calibrated on no-fault runs (plan.md 3.2.9, build step 6).
Nothing here calls a model.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from statistics import fmean
from typing import Mapping, Optional, Sequence

from router import vocab as V
from router.prereq.behavior_trace import BehaviorSource, BehaviorTrace
from router.schemas import Evidence, Incident, Mode
from router.stage1.nominal import NominalStats

SOURCE = "behavior"

M_BIAS = "behavior/offset_bias"
M_SCATTER = "behavior/offset_scatter"
M_WRONG = "behavior/wrong_object_fraction"
REQUIRED_METRICS = (M_BIAS, M_SCATTER, M_WRONG)

EV_TARGET = "behavior.implied_target"
EV_PHASE = "behavior.failure_phase"


@dataclass(frozen=True)
class BehaviorConfig:
    """``min_episodes``:     fewest failing episodes (with a grasp record, for the
                             implied target) the analysis will judge on.
    ``min_consistency``:  bias / RMS offset needed to call a shift "consistent"
                          (1 = every episode off by the same vector).
    ``min_phase_share``:  share of failing episodes the modal phase needs.
    ``fire_z``:           overrides ``NominalStats.fire_z`` when set.
    ``mode``:             which logged episodes to read. CLEAN: Stage 0 always
                          produces clean-mode episodes of the failing seeds on
                          the code under test (also on a re-run after a fix),
                          and stale server actions cannot smear the grasp
                          position into a false "perception failed"
                          (DECISIONS.md, Fix F1)."""
    mode: Mode = Mode.CLEAN
    failed_only: bool = True
    min_episodes: int = 3
    min_consistency: float = 0.9
    min_phase_share: float = 0.6
    fire_z: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", Mode(self.mode))
        if self.min_episodes < 1:
            raise ValueError("min_episodes must be >= 1")
        if not (0.0 < self.min_consistency <= 1.0) or not (0.0 < self.min_phase_share <= 1.0):
            raise ValueError("min_consistency and min_phase_share must be in (0, 1]")


@dataclass(frozen=True)
class BehaviorAnalysis:
    evidence: tuple[Evidence, ...]
    notes: tuple[str, ...] = ()
    n_episodes: int = 0                  # episodes analysed (failing ones when failed_only)
    n_with_grasp: int = 0
    bias: Optional[float] = None
    scatter: Optional[float] = None
    wrong_fraction: Optional[float] = None
    mean_offset: tuple[float, ...] = ()
    phase_counts: Mapping[str, int] = field(default_factory=dict)
    partial: tuple[V.PartialSkip, ...] = ()   # a "not judged" part, with the signals it leaves unchecked


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _mean_vec(vs: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(fmean(c) for c in zip(*vs))


def behavior_metrics(traces: Sequence[BehaviorTrace]) -> dict[str, float]:
    """Group metrics over the episodes in ``traces`` that logged a grasp.
    Empty when none did. The caller selects the episodes (failing ones for an
    incident; a no-fault group when calibrating ``NominalStats``)."""
    grasps = [t.grasp for t in traces if t.grasp is not None]
    if not grasps:
        return {}
    dims = {len(g.ee_pos) for g in grasps}
    if len(dims) != 1:
        raise ValueError(f"grasp records mix position dimensions {sorted(dims)}")
    offsets = [tuple(e - t for e, t in zip(g.ee_pos, g.target_pos)) for g in grasps]
    mean = _mean_vec(offsets)
    bias = math.hypot(*mean)
    scatter = math.sqrt(fmean(sum((o - m) ** 2 for o, m in zip(off, mean)) for off in offsets))
    wrong = fmean(
        float(bool(g.distractor_pos) and min(math.dist(g.ee_pos, p) for p in g.distractor_pos)
              < math.dist(g.ee_pos, g.target_pos))
        for g in grasps)
    return {M_BIAS: bias, M_SCATTER: scatter, M_WRONG: wrong}


def _fmt_vec(v: Sequence[float]) -> str:
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

def analyze_behavior(traces: Sequence[BehaviorTrace], nominal: NominalStats,
                     config: BehaviorConfig = BehaviorConfig()) -> BehaviorAnalysis:
    nominal.require(REQUIRED_METRICS)
    pool = [t for t in traces if not t.success] if config.failed_only else list(traces)
    notes: list[str] = []
    partial: list[V.PartialSkip] = []
    evidence: list[Evidence] = []

    def not_judged(note: str, signals: frozenset[str]) -> None:
        notes.append(note)
        partial.append(V.PartialSkip(note, signals))

    if not pool:
        why = "no failing episodes in the logs; nothing to analyse"
        return BehaviorAnalysis((), (why,), 0, 0, partial=(V.PartialSkip(why, V.SIGNALS_BY_SOURCE[SOURCE]),))

    # ---- implied target ---------------------------------------------------- #
    with_grasp = [t for t in pool if t.grasp is not None]
    target_signals = frozenset({V.SIG_IMPLIED_TARGET_OFFSET, V.SIG_IMPLIED_TARGET_OK, V.SIG_IMPLIED_TARGET_WRONG})
    phase_signals = frozenset(V.failure_phase(p) for p in V.PHASES)
    metrics: dict[str, float] = {}
    mean_offset: tuple[float, ...] = ()
    if len(with_grasp) < config.min_episodes:
        not_judged(f"implied target not judged: {len(with_grasp)} episode(s) with a grasp record, "
                   f"need {config.min_episodes} ({len(pool) - len(with_grasp)} never closed the gripper)",
                   target_signals)
    else:
        metrics = behavior_metrics(with_grasp)
        mean_offset = _mean_vec([[e - t for e, t in zip(t.grasp.ee_pos, t.grasp.target_pos)] for t in with_grasp])
        dev = {m: nominal.deviation(m, metrics[m], "high", config.fire_z) for m in REQUIRED_METRICS}
        bias, scatter = metrics[M_BIAS], metrics[M_SCATTER]
        rms = math.hypot(bias, scatter)
        consistency = bias / rms if rms > 0 else 0.0
        wrong_fires = dev[M_WRONG].fires
        offset = dev[M_BIAS].fires and consistency >= config.min_consistency and not wrong_fires
        wrong = wrong_fires or (dev[M_SCATTER].fires and not offset)
        nom = lambda m: nominal.metric(m).mean  # noqa: E731
        stats = (f"over {len(with_grasp)} failing episodes: bias {bias:.3f} m (nominal {nom(M_BIAS):.3f}), "
                 f"scatter {scatter:.3f} m (nominal {nom(M_SCATTER):.3f}), "
                 f"wrong-object {metrics[M_WRONG]:.0%} (nominal {nom(M_WRONG):.0%})")
        if offset:
            evidence.append(Evidence(
                EV_TARGET, SOURCE, V.SIG_IMPLIED_TARGET_OFFSET, dev[M_BIAS].strength, "grasp.ee_offset",
                f"end effector closes {_fmt_vec(mean_offset)} m from the true object (ee - target), "
                f"{consistency:.0%} consistent across episodes {stats}"))
        elif wrong:
            fired = [dev[m].strength for m in REQUIRED_METRICS if dev[m].fires and m != M_BIAS]
            why = "closer to another object than to the target in some episodes" if wrong_fires \
                else "scattered around the true object"
            evidence.append(Evidence(
                EV_TARGET, SOURCE, V.SIG_IMPLIED_TARGET_WRONG, max(fired), "grasp.target",
                f"end effector closes {why} {stats}"))
        elif not any(d.fires for d in dev.values()):
            evidence.append(Evidence(
                EV_TARGET, SOURCE, V.SIG_IMPLIED_TARGET_OK, 1.0 - max(d.strength for d in dev.values()),
                "grasp.target", f"end effector closes on the right object, small scatter {stats} "
                                f"(strength = 1 - largest metric strength)"))
        else:
            not_judged(f"implied target not judged: bias fires but the offset is only {consistency:.0%} "
                       f"consistent and scatter is within nominal {stats}", target_signals)

    # ---- failure phase ----------------------------------------------------- #
    counts = Counter(p for p in (t.failure_phase() for t in pool) if p)
    if sum(counts.values()) < config.min_episodes:
        not_judged(f"failure phase not judged: {sum(counts.values())} failing episode(s) with a subtask "
                   f"log, need {config.min_episodes}", phase_signals)
    else:
        total = sum(counts.values())
        (top, top_n), *rest = counts.most_common()
        share = top_n / total
        if rest and rest[0][1] == top_n:
            not_judged(f"failure phase not judged: tie between phases {dict(counts)}", phase_signals)
        elif share < config.min_phase_share:
            not_judged(f"failure phase not judged: modal phase {top!r} holds {share:.0%} of "
                       f"{total} failures, need {config.min_phase_share:.0%}", phase_signals)
        else:
            evidence.append(Evidence(
                EV_PHASE, SOURCE, V.failure_phase(top), share, f"subtask.{top}",
                f"{top_n} of {total} failing episodes did not complete '{top}' "
                f"(phases: {', '.join(f'{p}={n}' for p, n in counts.most_common())}); "
                f"strength = share of failing episodes, not a deviation from nominal"))

    return BehaviorAnalysis(
        tuple(evidence), tuple(notes), len(pool), len(with_grasp), metrics.get(M_BIAS), metrics.get(M_SCATTER),
        metrics.get(M_WRONG), mean_offset, dict(counts), tuple(partial))


def run_behavior_analysis(incident: Incident, source: BehaviorSource, nominal: NominalStats,
                          config: BehaviorConfig = BehaviorConfig()) -> BehaviorAnalysis:
    """C_behavior for one incident, with its notes ("not judged" reasons).
    ``router.route`` calls this on Branch C; it is not a Stage 1 source.
    Raises ``TraceUnavailable`` when the logs are missing."""
    return analyze_behavior(source.load(incident, config.mode), nominal, config)


def run_behavior(incident: Incident, source: BehaviorSource, nominal: NominalStats,
                 config: BehaviorConfig = BehaviorConfig()) -> list[Evidence]:
    """The evidence of ``run_behavior_analysis``."""
    return list(run_behavior_analysis(incident, source, nominal, config).evidence)
