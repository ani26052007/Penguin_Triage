"""SceneSpec factors -> Arena variation overrides (plan 3.2.2; PLAN_UPDATE (a), 6.1).

A scene is "nominal plus named factor changes". Each factor name maps to exactly one Arena
variation (``<asset>.<variation>``) through a YAML factor table, for example::

    factors:
      lighting.intensity: {asset: light, variation: intensity, kind: build_time, sampler: uniform}
      lighting.hdr:       {asset: light, variation: hdr_image, kind: build_time, sampler: choice,
                           values_field: hdr_names}

A factor value is held fixed (``docs/PILOT_AND_ORDER.md``, ``A:docs/pages/concepts/variations/
variations.rst:110-131``):

* ``uniform``   ``enabled=true`` with ``sampler_cfg.low == sampler_cfg.high == [v1, ...]``; the
                value string is ``dims`` comma-separated numbers;
* ``choice``    ``enabled=true`` with a one-element ``<values_field>`` list (HDR names);
* ``bernoulli`` ``enabled=true`` with ``sampler_cfg.probability = v``.

Restoring a factor is disabling its variation: variations start disabled
(``A:isaaclab_arena/variations/variation_base.py:38``), so a factor that is not in the scene has no
override at all, and restore(i) removes exactly factor i's overrides.

The mapping is total and injective: an unknown factor name raises ``UnknownFactor`` (never a
silent nominal), and two factor names may not point at the same variation.

Field names come from the code, not the catalogue docstring: the sampler field is ``sampler_cfg``
(``variation_base.py:41``), although ``variations_printing.py:47`` shows ``sampler``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from router.schemas import SceneSpec

KINDS = ("build_time", "run_time")
SAMPLERS = ("uniform", "choice", "bernoulli")


class UnknownFactor(KeyError):
    """A scene-spec factor that the factor table does not know."""


class FactorTableError(ValueError):
    """A factor table that cannot give a total, injective mapping."""


def hydra_override(path: str, value: Any) -> str:
    """One override string, serialised as Arena's ``overrides_from_dict`` does
    (``A:isaaclab_arena/variations/variations_hydra.py:225-239``)."""
    return f"{path}={json.dumps(value, separators=(',', ':'))}"


@dataclass(frozen=True)
class FactorSpec:
    name: str
    asset: str
    variation: str
    kind: str
    sampler: str
    dims: int = 1
    values_field: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise FactorTableError(f"factor {self.name!r}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.sampler not in SAMPLERS:
            raise FactorTableError(f"factor {self.name!r}: sampler must be one of {SAMPLERS}, got {self.sampler!r}")
        if self.sampler == "choice" and not self.values_field:
            raise FactorTableError(f"factor {self.name!r}: a choice factor needs values_field (e.g. hdr_names)")
        if self.dims < 1:
            raise FactorTableError(f"factor {self.name!r}: dims must be >= 1")
        for part in (self.asset, self.variation):
            if not part or "." in part:
                raise FactorTableError(f"factor {self.name!r}: asset / variation must be non-empty, without '.'")

    @property
    def path(self) -> str:
        return f"{self.asset}.{self.variation}"

    def assignments(self, value: str) -> dict[str, Any]:
        """``{dotted override path: value}`` holding this factor at ``value``."""
        p = self.path
        out: dict[str, Any] = {f"{p}.enabled": True}
        if self.sampler == "uniform":
            try:
                nums = [float(x) for x in str(value).split(",")]
            except ValueError:
                raise ValueError(f"factor {self.name!r}: value {value!r} is not {self.dims} number(s)") from None
            if len(nums) != self.dims:
                raise ValueError(f"factor {self.name!r}: value {value!r} has {len(nums)} number(s), needs {self.dims}")
            out[f"{p}.sampler_cfg.low"] = nums
            out[f"{p}.sampler_cfg.high"] = list(nums)
        elif self.sampler == "choice":
            if not str(value):
                raise ValueError(f"factor {self.name!r}: empty choice value")
            out[f"{p}.{self.values_field}"] = [str(value)]
        else:
            prob = float(value)
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"factor {self.name!r}: probability {value!r} outside [0, 1]")
            out[f"{p}.sampler_cfg.probability"] = prob
        return out

    def to_dict(self) -> dict:
        d = {"asset": self.asset, "variation": self.variation, "kind": self.kind, "sampler": self.sampler}
        if self.dims != 1:
            d["dims"] = self.dims
        if self.values_field:
            d["values_field"] = self.values_field
        return d


class FactorTable:
    def __init__(self, specs: Mapping[str, FactorSpec]) -> None:
        by_path: dict[str, str] = {}
        for name, spec in specs.items():
            if spec.name != name:
                raise FactorTableError(f"factor table key {name!r} holds spec named {spec.name!r}")
            if spec.path in by_path:
                raise FactorTableError(f"factors {by_path[spec.path]!r} and {name!r} both map to {spec.path!r} "
                                       f"(the mapping must be injective)")
            by_path[spec.path] = name
        self.specs = dict(specs)

    @classmethod
    def from_dict(cls, d: Mapping) -> "FactorTable":
        if set(d) != {"factors"} or not isinstance(d["factors"], Mapping):
            raise FactorTableError("a factor table is a mapping with one key, 'factors'")
        allowed = {"asset", "variation", "kind", "sampler", "dims", "values_field"}
        specs = {}
        for name, row in d["factors"].items():
            unknown = set(row) - allowed
            if unknown:
                raise FactorTableError(f"factor {name!r}: unknown keys {sorted(unknown)}")
            specs[str(name)] = FactorSpec(name=str(name), **row)
        return cls(specs)

    @classmethod
    def load(cls, path: str | Path) -> "FactorTable":
        from vla_isaac.config import read_yaml
        return cls.from_dict(read_yaml(path))

    def to_dict(self) -> dict:
        return {"factors": {n: s.to_dict() for n, s in sorted(self.specs.items())}}

    def spec(self, name: str) -> FactorSpec:
        if name not in self.specs:
            raise UnknownFactor(f"factor {name!r} is not in the factor table; known: {sorted(self.specs)}")
        return self.specs[name]

    def assignments(self, scene: SceneSpec) -> dict[str, Any]:
        """Every override for ``scene``, factors in name order. Unknown factors raise."""
        out: dict[str, Any] = {}
        for f in scene.factors:
            out.update(self.spec(f.name).assignments(f.value))
        return out

    def overrides(self, scene: SceneSpec) -> list[str]:
        """Hydra override strings for the policy-runner command line."""
        return [hydra_override(p, v) for p, v in self.assignments(scene).items()]

    def build_time_factors(self, scene: SceneSpec) -> tuple[str, ...]:
        return tuple(f.name for f in scene.factors if self.spec(f.name).kind == "build_time")
