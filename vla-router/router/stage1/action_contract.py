"""Stage 1 check 3: action contract (plan.md 3.2.4).

"Check the policy's raw outputs against the training action space (dimensions,
ranges, gripper sign convention). Compare commanded vs actual gripper width and
commanded vs tracked joint positions. Systematic tracking error points at
gains. A gripper that never opens points at a sign or threshold bug."

Sub-checks (all ``contract_violation``, located on the action side):

* dimensions      raw output length vs the training action space (exact)
                  ``action.dims``
* range           ``action/range_excess``: mean over steps of how far the raw
                  output leaves the training [min, max], in units of the range
                  ``action.range``
* gripper sign    ``action/gripper_sign_corr``: correlation of the policy's
                  gripper output with the commanded gripper width. Inverted
                  sign gives the opposite correlation; a threshold bug that
                  freezes the command gives 0. ``action.gripper_sign``
* gripper tracking ``action/gripper_tracking_err``: commanded vs actual width
                  ``action.gripper_tracking``
* joint tracking   ``action/joint_tracking_err``: commanded vs tracked joint
                  positions, normalised by the commanded range. High error
                  "points at gains" (it also rises with code latency, so this
                  can fire next to ``staleness_constant``; both are code)
                  ``action.joint_tracking``

Why tracking is measured commanded-vs-actual: that comparison is mechanical.
A policy that fails because it never decides to close the gripper still has an
actual width that follows its command, so a vision failure does not look like
a gripper bug.

Training action space: from ``TrainingSpec.action_layout`` (dimensions, gripper
index) and ``TrainingSpec.norm_stats[<action key>]["min" / "max"]`` (range).
A missing piece skips the sub-checks that need it, with a warning.
TODO(verify): the real key names (``norm_stats`` layout, the gripper's layout
name) for the chosen policy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, median
from typing import Optional, Sequence

from router import vocab as V
from router.prereq.trace import EpisodeTrace, TraceSource
from router.schemas import Evidence, Incident, Mode, TrainingSpec
from router.stage1.base import Check, log, nrmse, pearson
from router.stage1.nominal import NominalStats

M_RANGE = "action/range_excess"
M_GRIPPER_SIGN = "action/gripper_sign_corr"
M_GRIPPER_TRACK = "action/gripper_tracking_err"
M_JOINT_TRACK = "action/joint_tracking_err"
REQUIRED_METRICS = (M_RANGE, M_GRIPPER_SIGN, M_GRIPPER_TRACK, M_JOINT_TRACK)


@dataclass(frozen=True)
class ActionSpace:
    dims: Optional[int]
    train_min: Optional[tuple[float, ...]]
    train_max: Optional[tuple[float, ...]]
    gripper_index: Optional[int]


def action_space(spec: TrainingSpec, stats_key: str = "action", gripper_name: str = "gripper") -> ActionSpace:
    layout = spec.action_layout or {}
    dims = max((int(r[1]) for r in layout.values()), default=None)
    stats = (spec.norm_stats or {}).get(stats_key) or {}
    tmin = tuple(float(x) for x in stats["min"]) if "min" in stats else None
    tmax = tuple(float(x) for x in stats["max"]) if "max" in stats else None
    if dims is None and tmin is not None:
        dims = len(tmin)
    gi = None
    if gripper_name in layout:
        start, end = layout[gripper_name]
        gi = int(start) if int(end) - int(start) == 1 else None
    return ActionSpace(dims, tmin, tmax, gi)


# --------------------------------------------------------------------------- #
# metrics (per episode)
# --------------------------------------------------------------------------- #

def _range_excess(tr: EpisodeTrace, sp: ActionSpace) -> Optional[float]:
    if not tr.raw_actions or sp.train_min is None or sp.train_max is None:
        return None
    if any(len(a) != len(sp.train_min) or len(a) != len(sp.train_max) for a in tr.raw_actions):
        return None
    widths = [hi - lo for lo, hi in zip(sp.train_min, sp.train_max)]
    per_step = []
    for a in tr.raw_actions:
        per_step.append(max((max(0.0, lo - x, x - hi) / w for x, lo, hi, w in zip(a, sp.train_min, sp.train_max, widths)
                             if w > 0), default=0.0))
    return fmean(per_step)


def _gripper_sign(tr: EpisodeTrace, sp: ActionSpace) -> Optional[float]:
    gi = sp.gripper_index
    if gi is None or not tr.raw_actions or len(tr.raw_actions) != len(tr.gripper_cmd_width):
        return None
    if any(len(a) <= gi for a in tr.raw_actions):
        return None
    raw = [a[gi] for a in tr.raw_actions]
    if max(raw) - min(raw) < 1e-9:
        return None                                  # the policy never moved the gripper: nothing to compare
    if max(tr.gripper_cmd_width) - min(tr.gripper_cmd_width) < 1e-9:
        return 0.0                                   # policy varies, command is frozen: no response at all
    return pearson(raw, tr.gripper_cmd_width)


def _joint_tracking(tr: EpisodeTrace) -> Optional[float]:
    if not tr.joint_cmd or len(tr.joint_cmd) != len(tr.joint_pos):
        return None
    errs = []
    for j in range(len(tr.joint_cmd[0])):
        e = nrmse([row[j] for row in tr.joint_cmd], [row[j] for row in tr.joint_pos])
        if e is not None:
            errs.append(e)
    return fmean(errs) if errs else None


def action_metrics(traces: Sequence[EpisodeTrace], sp: ActionSpace) -> dict[str, list[float]]:
    """Per-episode metric values, one list per metric (an episode that cannot
    be scored for a metric is left out of that metric's list)."""
    out: dict[str, list[float]] = defaultdict(list)
    for tr in traces:
        for name, v in ((M_RANGE, _range_excess(tr, sp)), (M_GRIPPER_SIGN, _gripper_sign(tr, sp)),
                        (M_GRIPPER_TRACK, nrmse(tr.gripper_cmd_width, tr.gripper_width)),
                        (M_JOINT_TRACK, _joint_tracking(tr))):
            if v is not None:
                out[name].append(v)
    return dict(out)


# --------------------------------------------------------------------------- #
# the check
# --------------------------------------------------------------------------- #

class ActionContractCheck(Check):
    """``mode``: which logged episodes to read. CLEAN by default: a stale
    server delays actions, which inflates the tracking error and would read as
    a gains bug (DECISIONS.md, Fix F1)."""

    source = "contract_action"

    def __init__(self, nominal: NominalStats, traces: TraceSource, *, mode: Mode = Mode.CLEAN,
                 stats_key: str = "action", gripper_name: str = "gripper", fire_z: Optional[float] = None) -> None:
        nominal.require(REQUIRED_METRICS)
        self.nominal, self.traces, self.mode, self.fire_z = nominal, traces, Mode(mode), fire_z
        self.stats_key, self.gripper_name = stats_key, gripper_name

    def run(self, incident: Incident) -> list[Evidence]:
        return self.run_partial(incident)[0]

    def run_partial(self, incident: Incident) -> tuple[list[Evidence], list[V.PartialSkip]]:
        """A sub-check the training spec cannot support is a partial skip on
        ``contract_violation``: the check ran, but not all of it."""
        traces = self.traces.load(incident, self.mode)
        sp = action_space(incident.training_spec, self.stats_key, self.gripper_name)
        out: list[Evidence] = []
        skips: list[V.PartialSkip] = []

        def not_checked(what: str, why: str) -> None:
            msg = f"action contract: {why}; {what} not checked"
            log.warning("%s", msg)
            skips.append(V.PartialSkip(msg, {V.SIG_CONTRACT_VIOLATION}))

        bad = [t.seed for t in traces if sp.dims is not None and any(len(a) != sp.dims for a in t.raw_actions)]
        if bad:
            seen = sorted({len(a) for t in traces for a in t.raw_actions if len(a) != sp.dims})
            out.append(Evidence("s1.action.dims", self.source, V.SIG_CONTRACT_VIOLATION, 1.0, "action.dims",
                                f"policy raw output has {seen} dimensions in {len(bad)} of {len(traces)} episodes; "
                                f"the training action space has {sp.dims}"))
        elif sp.dims is None:
            not_checked("dimensions", "no action layout in the training spec")

        vals = action_metrics(traces, sp)
        if sp.train_min is None or sp.train_max is None:
            not_checked("range", "no action min/max in the training spec")
        if sp.gripper_index is None:
            not_checked("gripper sign", f"no single-index {self.gripper_name!r} layout in the training spec")

        def fire(metric: str, side: str, signal_id: str, location: str, text) -> None:
            xs = vals.get(metric)
            if not xs:
                return
            d = self.nominal.deviation(metric, median(xs), side, self.fire_z)
            if d.fires:
                nom = self.nominal.metric(metric).mean
                out.append(Evidence(f"s1.action.{signal_id}", self.source, V.SIG_CONTRACT_VIOLATION, d.strength,
                                    location, f"{text(d.value, nom)}"))

        fire(M_RANGE, "high", "range", "action.range",
             lambda v, n: f"policy raw output leaves the training action range by {v:.3g} of the range on average "
                          f"(nominal {n:.3g})")
        fire(M_GRIPPER_SIGN, "both", "gripper_sign", "action.gripper_sign",
             lambda v, n: f"correlation of the policy's gripper output with the commanded gripper width is {v:+.2f} "
                          f"(nominal {n:+.2f}): sign convention inverted or command not following the output")
        fire(M_GRIPPER_TRACK, "high", "gripper_tracking", "action.gripper_tracking",
             lambda v, n: f"actual gripper width misses the command by {v:.3g} of the command range (nominal {n:.3g})")
        fire(M_JOINT_TRACK, "high", "joint_tracking", "action.joint_tracking",
             lambda v, n: f"tracked joint positions miss the command by {v:.3g} of the command range (nominal {n:.3g}): "
                          f"systematic tracking error, points at controller gains")
        return out, skips
