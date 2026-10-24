"""Stage 1 check 2: observation contract (plan.md 3.2.4).

Compares the logged policy-input tensors with what the policy was trained on:

* value range (normalisation)         ``obs/<cam>/value_min`` and ``value_max``
* channel order (RGB vs BGR)          ``obs/<cam>/opp_dist``
* orientation (flip, rotation)        ``obs/<cam>/row_corr`` and ``col_corr``
* resolution / aspect ratio           exact ``(height, width)`` vs ``image_size``
* the stronger open-loop version      ``obs/open_loop_ratio`` (optional probe)

All of them emit ``contract_violation``, located at the observation side
(``obs.normalization``, ``obs.channel_order``, ``obs.orientation``,
``obs.image_size``, ``obs.preprocessing``).

Reading the metrics
-------------------
Unit = one episode. Each metric is the median over that episode's frames, and
the check scores the median over episodes against the nominal per-episode
distribution (``NominalStats``). A fault has to show in most episodes, which
keeps a single odd episode from raising the alarm.

* Channel order uses the *opponent vector* ``means - mean(means)``, which does
  not move when the whole image gets brighter or darker, and needs no division
  (a zero-mean normalised tensor would break a chromaticity ratio). When it
  fires, the best channel permutation names the swap (``BGR``) if one explains
  it clearly. A near-grey scene has a near-zero opponent vector, so a swap there
  is invisible; that is a limit of the check, not a clean bill of health.
* Orientation correlates the frame's row / column intensity profile with the
  nominal profile and tries the four flips of it. It needs a scene with some
  vertical / horizontal intensity structure.
* Value range, channel and orientation references are the *nominal* ones for
  the incident's scene condition (see ``nominal.py``).

Open-loop probe
---------------
"Push a few training-set frames through the current preprocessing and measure
the policy's open-loop action error against the dataset actions. If it is much
worse than with the reference preprocessing, the observation path is broken."
That needs the policy and the training data, so it sits behind ``OpenLoopProbe``.
TODO(verify): the real probe (``PolicyOpenLoopProbe``).
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from statistics import fmean, median
from typing import Mapping, Optional, Sequence

from router import vocab as V
from router.prereq.trace import EpisodeTrace, FrameStats, TraceSource
from router.schemas import Evidence, Incident, Mode
from router.stage1.base import Check, log, pearson
from router.stage1.nominal import NominalStats

M_OPEN_LOOP_RATIO = "obs/open_loop_ratio"


def metric(cam: str, name: str) -> str:
    return f"obs/{cam}/{name}"


def required_metrics(cameras: Sequence[str]) -> list[str]:
    return [metric(c, n) for c in cameras for n in ("value_min", "value_max", "opp_dist", "row_corr", "col_corr")]


def required_vectors(cameras: Sequence[str]) -> list[str]:
    return [metric(c, n) for c in cameras for n in ("opp", "row_profile", "col_profile")]


# --------------------------------------------------------------------------- #
# open-loop probe (interface)
# --------------------------------------------------------------------------- #

class PreprocessingPath(str, Enum):
    CURRENT = "current"      # the preprocessing in the code under test
    REFERENCE = "reference"  # the training pipeline's own preprocessing


class OpenLoopProbe(ABC):
    @abstractmethod
    def action_error(self, incident: Incident, path: PreprocessingPath) -> float:
        """Mean normalised action error of the policy, run open loop on a few
        training-set frames through ``path``, against the dataset actions."""


class PolicyOpenLoopProbe(OpenLoopProbe):
    """Placeholder for the real probe. Deliberately unimplemented.

      TODO(verify): where the training-set frames and their actions live and
                    how to read a few of them (LeRobot parquet + mp4 for GR00T).
      TODO(verify): how to run the policy forward on them without the simulator.
      TODO(verify): how to build the reference preprocessing from the policy's
                    own training transform, and the current one from the wrapper.
      TODO(verify): the action error (per-dimension normalisation, horizon).
    """

    def action_error(self, incident, path):  # pragma: no cover
        raise NotImplementedError("PolicyOpenLoopProbe is not implemented; see TODO(verify) notes")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def opponent(channel_means: Sequence[float]) -> tuple[float, ...]:
    m = fmean(channel_means)
    return tuple(x - m for x in channel_means)


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b)))


def _frame_metrics(f: FrameStats, cam: str, vectors: Mapping[str, Sequence[float]]) -> dict[str, float]:
    out = {metric(cam, "value_min"): f.min_value, metric(cam, "value_max"): f.max_value}
    ref = vectors.get(metric(cam, "opp"))
    if ref is not None and len(ref) == len(f.channel_means):
        out[metric(cam, "opp_dist")] = _dist(opponent(f.channel_means), ref)
    for name, series in (("row", f.row_profile), ("col", f.col_profile)):
        ref = vectors.get(metric(cam, f"{name}_profile"))
        if ref is not None and len(ref) == len(series):
            c = pearson(series, ref)
            if c is not None:
                out[metric(cam, f"{name}_corr")] = c
    return out


def observation_metrics(traces: Sequence[EpisodeTrace], cameras: Sequence[str],
                        vectors: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Per-episode metric values (median over the episode's frames), one list
    per metric. Used by the check and to calibrate ``NominalStats`` on no-fault
    traces."""
    out: dict[str, list[float]] = defaultdict(list)
    for tr in traces:
        per_frame: dict[str, list[float]] = defaultdict(list)
        for fs in tr.frames:
            if fs.camera in cameras:
                for k, v in _frame_metrics(fs.policy_input, fs.camera, vectors).items():
                    per_frame[k].append(v)
        for k, vs in per_frame.items():
            out[k].append(median(vs))
    return dict(out)


def _median_vector(vs: Sequence[Sequence[float]]) -> tuple[float, ...]:
    return tuple(median(col) for col in zip(*vs))


def _order_name(perm: Sequence[int], order: Optional[str]) -> str:
    letters = list(order) if order and len(order) == len(perm) else [str(i) for i in range(len(perm))]
    s: list[str] = [""] * len(perm)
    for i, p in enumerate(perm):
        s[p] = letters[i]
    return "".join(s)


def _best_permutation(opp: Sequence[float], ref: Sequence[float]) -> tuple[tuple[int, ...], float]:
    best, best_d = tuple(range(len(opp))), _dist(opp, ref)
    for p in itertools.permutations(range(len(opp))):
        d = _dist([opp[i] for i in p], ref)
        if d < best_d - 1e-12:
            best, best_d = p, d
    return best, best_d


_ORIENTATIONS = {(False, False): "none", (True, False): "vertical flip",
                 (False, True): "horizontal flip", (True, True): "rotation by 180 degrees"}


def _orientation(rows: Sequence[float], cols: Sequence[float], ref_rows: Sequence[float],
                 ref_cols: Sequence[float]) -> str:
    """Name the flip that best maps the frame's profiles onto the nominal ones,
    or "none" when no flip explains the mismatch clearly."""
    scores = {}
    for rr, cr in _ORIENTATIONS:
        r = pearson(list(reversed(rows)) if rr else rows, ref_rows)
        c = pearson(list(reversed(cols)) if cr else cols, ref_cols)
        if r is not None and c is not None:
            scores[(rr, cr)] = r + c
    if (False, False) not in scores:
        return "none"
    best = max(scores, key=lambda k: (scores[k], k == (False, False)))
    return _ORIENTATIONS[best] if scores[best] - scores[(False, False)] > 0.5 else "none"


# --------------------------------------------------------------------------- #
# the check
# --------------------------------------------------------------------------- #

class ObservationContractCheck(Check):
    """``mode``: which logged episodes to read. Clean mode by default, so server
    effects cannot contaminate a code contract (3.2.2)."""

    source = "contract_obs"

    def __init__(self, nominal: NominalStats, traces: TraceSource, cameras: Sequence[str], *,
                 mode: Mode = Mode.CLEAN, probe: Optional[OpenLoopProbe] = None,
                 fire_z: Optional[float] = None) -> None:
        self.cameras = tuple(cameras)
        if not self.cameras:
            raise ValueError("need at least one camera")
        metrics = required_metrics(self.cameras) + ([M_OPEN_LOOP_RATIO] if probe else [])
        nominal.require(metrics, required_vectors(self.cameras))
        self.nominal, self.traces, self.mode, self.probe, self.fire_z = nominal, traces, Mode(mode), probe, fire_z
        self._vectors = {n: nominal.vector(n) for n in required_vectors(self.cameras)}

    def run(self, incident: Incident) -> list[Evidence]:
        traces = self.traces.load(incident, self.mode)
        spec = incident.training_spec
        vals = observation_metrics(traces, self.cameras, self._vectors)
        out = self._resolution(spec, traces)
        for cam in self.cameras:
            out += self._range(cam, vals) + self._channels(cam, spec, traces, vals) \
                + self._orientation(cam, traces, vals)
        if self.probe is not None:
            out += self._open_loop(incident)
        return out

    # -- sub-checks ----------------------------------------------------------
    def _dev(self, name: str, vals: Mapping[str, Sequence[float]], side: str):
        xs = vals.get(name)
        if not xs:
            log.warning("obs contract: no logged values for %s; not checked", name)
            return None
        return self.nominal.deviation(name, median(xs), side, self.fire_z)

    def _resolution(self, spec, traces: Sequence[EpisodeTrace]) -> list[Evidence]:
        out = []
        frames = [fs for t in traces for fs in t.frames]
        if spec.image_size is not None:
            bad = [fs for fs in frames if tuple(fs.policy_input.shape[:2]) != tuple(spec.image_size)]
            if bad:
                seen = sorted({tuple(fs.policy_input.shape[:2]) for fs in bad})
                out.append(Evidence(
                    "s1.obs.image_size", self.source, V.SIG_CONTRACT_VIOLATION, 1.0, "obs.image_size",
                    f"policy input is {seen} (height, width) in {len(bad)} of {len(frames)} logged frames; "
                    f"the training spec says {tuple(spec.image_size)}"))
        if spec.channel_order is not None:
            bad = [fs for fs in frames if fs.policy_input.shape[2] != len(spec.channel_order)]
            if bad:
                out.append(Evidence(
                    "s1.obs.channels", self.source, V.SIG_CONTRACT_VIOLATION, 1.0, "obs.image_size",
                    f"policy input has {bad[0].policy_input.shape[2]} channels in {len(bad)} of {len(frames)} "
                    f"logged frames; the training spec says {spec.channel_order!r}"))
        return out

    def _range(self, cam: str, vals) -> list[Evidence]:
        devs = [d for d in (self._dev(metric(cam, "value_min"), vals, "both"),
                            self._dev(metric(cam, "value_max"), vals, "both")) if d]
        hit = [d for d in devs if d.fires]
        if not hit:
            return []
        lo, hi = median(vals[metric(cam, "value_min")]), median(vals[metric(cam, "value_max")])
        n_lo, n_hi = self.nominal.metric(metric(cam, "value_min")).mean, self.nominal.metric(metric(cam, "value_max")).mean
        return [Evidence(f"s1.obs.value_range.{cam}", self.source, V.SIG_CONTRACT_VIOLATION,
                         max(d.strength for d in hit), "obs.normalization",
                         f"camera {cam}: policy-input value range [{lo:.4g}, {hi:.4g}] vs nominal "
                         f"[{n_lo:.4g}, {n_hi:.4g}] (normalisation)")]

    def _channels(self, cam: str, spec, traces, vals) -> list[Evidence]:
        d = self._dev(metric(cam, "opp_dist"), vals, "high")
        if d is None or not d.fires:
            return []
        ref = self._vectors[metric(cam, "opp")]
        frames = [fs.policy_input for t in traces for fs in t.frames if fs.camera == cam]
        opp = _median_vector([opponent(f.channel_means) for f in frames if len(f.channel_means) == len(ref)])
        loc, why = "obs.channel_means", "channel means differ from the nominal colour balance"
        if len(opp) <= 4:
            perm, dist = _best_permutation(opp, ref)
            if perm != tuple(range(len(perm))) and dist < 0.5 * _dist(opp, ref):
                name = _order_name(perm, spec.channel_order)
                loc, why = "obs.channel_order", (f"channels look permuted: order appears to be {name}"
                                                  + (f" instead of {spec.channel_order}" if spec.channel_order else ""))
        return [Evidence(f"s1.obs.channel_order.{cam}", self.source, V.SIG_CONTRACT_VIOLATION, d.strength, loc,
                         f"camera {cam}: {why} (opponent-vector distance {d.value:.3g}, nominal "
                         f"{self.nominal.metric(d.metric).mean:.3g})")]

    def _orientation(self, cam: str, traces, vals) -> list[Evidence]:
        devs = [d for d in (self._dev(metric(cam, "row_corr"), vals, "low"),
                            self._dev(metric(cam, "col_corr"), vals, "low")) if d]
        hit = [d for d in devs if d.fires]
        if not hit:
            return []
        frames = [fs.policy_input for t in traces for fs in t.frames if fs.camera == cam]
        rp, cp = self._vectors[metric(cam, "row_profile")], self._vectors[metric(cam, "col_profile")]
        rows = _median_vector([f.row_profile for f in frames if len(f.row_profile) == len(rp)])
        cols = _median_vector([f.col_profile for f in frames if len(f.col_profile) == len(cp)])
        what = _orientation(rows, cols, rp, cp)
        why = (f"image looks like a {what} of the training orientation" if what != "none"
               else "intensity profile does not match the training orientation and no flip explains it")
        return [Evidence(f"s1.obs.orientation.{cam}", self.source, V.SIG_CONTRACT_VIOLATION,
                         max(d.strength for d in hit), "obs.orientation",
                         f"camera {cam}: {why} (row/col profile correlation "
                         f"{median(vals[metric(cam, 'row_corr')]):.2f}/{median(vals[metric(cam, 'col_corr')]):.2f})")]

    def _open_loop(self, incident: Incident) -> list[Evidence]:
        current = self.probe.action_error(incident, PreprocessingPath.CURRENT)
        reference = self.probe.action_error(incident, PreprocessingPath.REFERENCE)
        if not reference > 0:
            raise ValueError(f"reference open-loop action error must be positive, got {reference}")
        d = self.nominal.deviation(M_OPEN_LOOP_RATIO, current / reference, "high", self.fire_z)
        if not d.fires:
            return []
        return [Evidence("s1.obs.open_loop", self.source, V.SIG_CONTRACT_VIOLATION, d.strength, "obs.preprocessing",
                         f"open-loop action error through the current preprocessing is {d.value:.2f}x the error "
                         f"through the reference preprocessing (nominal "
                         f"{self.nominal.metric(M_OPEN_LOOP_RATIO).mean:.2f}x)")]
