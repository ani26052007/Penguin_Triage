"""Nominal distribution of the Stage 1 metrics (plan.md 3.2.3 "strength").

"Strength: how far outside the nominal distribution it is, calibrated on
no-fault runs (0 to 1)." This module is that distribution, passed to the checks
as data.

What it holds
-------------
* ``metrics``: for each metric name, the mean / std / count of the per-episode
  value over no-fault runs, plus a ``min_std`` floor. The floor is what keeps a
  metric that is exactly constant in no-fault runs (a NaN fraction of 0, a
  fixed base latency) from turning a rounding error into an infinite z-score.
  Floors are per-metric calibration inputs, like the thresholds in 3.2.9.
* ``vectors``: reference vectors the metrics are computed against (the nominal
  row profile of a camera, its opponent-channel vector).
* ``fire_z``: how many scale units outside the nominal a metric must be before
  a check emits evidence. Calibrated on no-fault incidents (build-order step 6)
  so that the false-flag rate stays under the chosen target; 4.0 is a
  placeholder, not a calibrated value.

Calibrate on the same scene condition the incident ran in. A declared factor
(dark lighting, glare) legitimately moves brightness and range metrics; a
nominal distribution built on the nominal scene would call that a code bug.

Strength: ``z / fire_z`` squashed by ``base.saturating``: 0.5 exactly at the
firing threshold, approaching 1 as the deviation grows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import Mapping, Optional, Sequence

from router.stage1.base import saturating

SIDES = ("both", "high", "low")


class MissingNominal(ValueError):
    """A check needs a metric or reference vector the nominal stats lack."""


@dataclass(frozen=True)
class MetricNominal:
    mean: float
    std: float
    n: int
    min_std: float = 1e-6

    def __post_init__(self) -> None:
        if not (math.isfinite(self.mean) and math.isfinite(self.std)) or self.std < 0:
            raise ValueError(f"bad nominal mean/std: {self.mean}, {self.std}")
        if self.min_std <= 0:
            raise ValueError("min_std must be positive")

    @property
    def scale(self) -> float:
        return max(self.std, self.min_std)


@dataclass(frozen=True)
class Deviation:
    """How far one value is outside the nominal distribution.

    z is signed toward the flagged side (``high``: above the mean, ``low``:
    below, ``both``: absolute), so a negative z means "on the harmless side".
    """
    metric: str
    value: float
    z: float
    fires: bool
    strength: float


class NominalStats:
    def __init__(self, metrics: Mapping[str, MetricNominal],
                 vectors: Optional[Mapping[str, Sequence[float]]] = None,
                 fire_z: float = 4.0) -> None:
        if not fire_z > 0:
            raise ValueError("fire_z must be positive")
        self.metrics = dict(metrics)
        self.vectors = {k: tuple(float(x) for x in v) for k, v in (vectors or {}).items()}
        self.fire_z = float(fire_z)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_samples(cls, samples: Mapping[str, Sequence[float]], *,
                     vectors: Optional[Mapping[str, Sequence[float]]] = None, fire_z: float = 4.0,
                     min_std: Optional[Mapping[str, float]] = None,
                     default_min_std: float = 1e-6) -> "NominalStats":
        floors = dict(min_std or {})
        metrics = {}
        for name, xs in samples.items():
            xs = [float(x) for x in xs]
            if len(xs) < 2:
                raise ValueError(f"metric {name!r}: need at least 2 no-fault samples, got {len(xs)}")
            if not all(math.isfinite(x) for x in xs):
                raise ValueError(f"metric {name!r}: non-finite sample")
            metrics[name] = MetricNominal(fmean(xs), stdev(xs), len(xs), floors.get(name, default_min_std))
        return cls(metrics, vectors, fire_z)

    # -- queries -------------------------------------------------------------
    def metric(self, name: str) -> MetricNominal:
        try:
            return self.metrics[name]
        except KeyError:
            raise MissingNominal(f"no nominal distribution for metric {name!r}") from None

    def vector(self, name: str) -> tuple[float, ...]:
        try:
            return self.vectors[name]
        except KeyError:
            raise MissingNominal(f"no nominal reference vector {name!r}") from None

    def require(self, metrics: Sequence[str] = (), vectors: Sequence[str] = ()) -> None:
        """Raise ``MissingNominal`` listing everything that is missing, so a
        check fails when it is built, not halfway through an incident."""
        missing = [f"metric {m!r}" for m in metrics if m not in self.metrics]
        missing += [f"vector {v!r}" for v in vectors if v not in self.vectors]
        if missing:
            raise MissingNominal("nominal stats are missing: " + ", ".join(missing))

    def deviation(self, metric: str, value: float, side: str = "both",
                  fire_z: Optional[float] = None) -> Deviation:
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {side!r}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"metric {metric!r}: non-finite value {value}")
        m = self.metric(metric)
        raw = (value - m.mean) / m.scale
        z = {"high": raw, "low": -raw, "both": abs(raw)}[side]
        threshold = self.fire_z if fire_z is None else float(fire_z)
        if not threshold > 0:
            raise ValueError("fire_z must be positive")
        return Deviation(metric, value, z, z >= threshold, saturating(max(z, 0.0) / threshold))

    # -- json ----------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"fire_z": self.fire_z,
                "metrics": {k: {"mean": m.mean, "std": m.std, "n": m.n, "min_std": m.min_std}
                            for k, m in sorted(self.metrics.items())},
                "vectors": {k: list(v) for k, v in sorted(self.vectors.items())}}

    @classmethod
    def from_dict(cls, d: Mapping) -> "NominalStats":
        metrics = {k: MetricNominal(float(m["mean"]), float(m["std"]), int(m["n"]), float(m.get("min_std", 1e-6)))
                   for k, m in d.get("metrics", {}).items()}
        return cls(metrics, d.get("vectors", {}), float(d.get("fire_z", 4.0)))
