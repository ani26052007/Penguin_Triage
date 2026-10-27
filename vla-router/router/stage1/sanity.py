"""Stage 1 check 5: render and physics sanity (plan.md 3.2.4).

"Black or saturated frames, missing textures, NaN states, interpenetration, an
object resting pose that differs from the logged initial state."

All of them emit ``render_physics_broken``, which is row 2 of the decision
table (sim / asset), the highest-precedence row after "no failure". A false
alarm here sends a healthy incident to the report, so the check is deliberately
conservative:

* it reads the **render** stats (the simulator's camera output before
  preprocessing), never the policy-input stats. A normalisation bug saturates
  the policy input, and must stay a code bug (row 4), not a sim fault;
* unit = one episode, scored by the median over episodes, so a fault must show
  in most logged episodes;
* the thresholds are the nominal distribution of the same scene condition. A
  declared glare factor legitimately raises the saturated fraction.

Metrics (per episode, one-sided: only "worse than nominal" counts):

  sanity/black_fraction            share of render frames with no signal at all
  sanity/saturated_fraction        mean share of saturated pixels
  sanity/missing_texture_fraction  mean share of missing-texture pixels
  sanity/nan_fraction              share of logged steps with NaN / inf state
  sanity/max_penetration           deepest contact penetration (m)
  sanity/rest_shift                distance between the object's logged initial
                                   position and its resting position (m)

A ``SimHealth`` field the backend does not measure is None (router change 5):
it is left out of its metric, never read as healthy, and the check says
"not measured" in the detail of the evidence it emits and in its notes. A
measured value is scored exactly as before.

TODO(verify): how the real backend measures penetration depth, the NaN
fraction, the missing-texture colour and the resting pose, and the pixel scale
that ``black_eps`` assumes (0 to 1).
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import fmean, median
from typing import Optional, Sequence

from router import vocab as V
from router.prereq.trace import EpisodeTrace, TraceSource
from router.schemas import Evidence, Incident, Mode
from router.stage1.base import Check
from router.stage1.nominal import NominalStats

M_BLACK = "sanity/black_fraction"
M_SATURATED = "sanity/saturated_fraction"
M_MISSING_TEXTURE = "sanity/missing_texture_fraction"
M_NAN = "sanity/nan_fraction"
M_PENETRATION = "sanity/max_penetration"
M_REST_SHIFT = "sanity/rest_shift"
REQUIRED_METRICS = (M_BLACK, M_SATURATED, M_MISSING_TEXTURE, M_NAN, M_PENETRATION, M_REST_SHIFT)

_WHAT = {   # metric -> (evidence id, location, description)
    M_BLACK: ("black_frames", "render.black_frames", "of the render frames are black (no signal)"),
    M_SATURATED: ("saturated_frames", "render.saturation", "of the render pixels are saturated"),
    M_MISSING_TEXTURE: ("missing_texture", "render.missing_texture", "of the render pixels show the missing-texture colour"),
    M_NAN: ("nan_state", "sim.nan_state", "of the logged steps have NaN or infinite simulator state"),
    M_PENETRATION: ("interpenetration", "sim.interpenetration", "m is the deepest contact penetration"),
    M_REST_SHIFT: ("rest_pose", "sim.rest_pose", "m is how far the object's resting pose is from its logged initial position"),
}


# SimHealth field -> metric, for the fields a backend may leave unmeasured (None).
_OPTIONAL_SIM = {"max_penetration": M_PENETRATION}


def not_measured(traces: Sequence[EpisodeTrace]) -> dict[str, int]:
    """Metric -> number of episodes whose ``SimHealth`` field for it is None (not measured)."""
    out: dict[str, int] = {}
    for tr in traces:
        if tr.sim is None:
            continue
        for attr, metric in _OPTIONAL_SIM.items():
            if getattr(tr.sim, attr) is None:
                out[metric] = out.get(metric, 0) + 1
    return out


def sanity_metrics(traces: Sequence[EpisodeTrace], black_eps: float = 1e-3) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for tr in traces:
        renders = [fs.render for fs in tr.frames if fs.render is not None]
        if renders:
            out[M_BLACK].append(sum(r.max_value <= black_eps for r in renders) / len(renders))
            out[M_SATURATED].append(fmean(r.saturated_fraction for r in renders))
            out[M_MISSING_TEXTURE].append(fmean(r.missing_texture_fraction for r in renders))
        if tr.sim is not None:
            out[M_NAN].append(tr.sim.nan_fraction)
            if tr.sim.max_penetration is not None:
                out[M_PENETRATION].append(tr.sim.max_penetration)
            a, b = tr.sim.object_initial_pos, tr.sim.object_rest_pos
            if a and len(a) == len(b):
                out[M_REST_SHIFT].append(math.dist(a, b))
    return dict(out)


class SanityCheck(Check):
    """``mode``: which logged episodes to read. CLEAN by default, the mode
    Stage 0 always runs in on the code under test; render and physics health
    do not depend on the inference mode (DECISIONS.md, Fix F1)."""

    source = "sanity"

    def __init__(self, nominal: NominalStats, traces: TraceSource, *, mode: Mode = Mode.CLEAN,
                 black_eps: float = 1e-3, fire_z: Optional[float] = None) -> None:
        nominal.require(REQUIRED_METRICS)
        self.nominal, self.traces, self.mode = nominal, traces, Mode(mode)
        self.black_eps, self.fire_z = black_eps, fire_z
        self._unmeasured: dict[str, str] = {}      # incident id -> "not measured" text of its last run

    def run(self, incident: Incident) -> list[Evidence]:
        traces = self.traces.load(incident, self.mode)
        vals = sanity_metrics(traces, self.black_eps)
        missing = not_measured(traces)
        unmeasured = "; ".join(f"{m} not measured ({k} of {len(traces)} episodes)" for m, k in sorted(missing.items()))
        self._unmeasured[incident.incident_id] = unmeasured
        out = []
        for name in REQUIRED_METRICS:
            xs = vals.get(name)
            if not xs:
                continue
            d = self.nominal.deviation(name, median(xs), "high", self.fire_z)
            if d.fires:
                eid, loc, what = _WHAT[name]
                shown = f"{d.value:.1%}" if name in (M_BLACK, M_SATURATED, M_MISSING_TEXTURE, M_NAN) else f"{d.value:.3g}"
                detail = (f"{shown} {what} (median over {len(xs)} episodes; nominal "
                          f"{self.nominal.metric(name).mean:.3g})")
                out.append(Evidence(f"s1.sanity.{eid}", self.source, V.SIG_RENDER_PHYSICS_BROKEN, d.strength, loc,
                                    f"{detail}; {unmeasured}" if unmeasured else detail))
        return out

    def notes(self, incident: Incident) -> list[str]:
        """The metrics the last run could not score because the backend does not measure them."""
        unmeasured = self._unmeasured.get(incident.incident_id)
        return [f"sanity: {unmeasured}"] if unmeasured else []
