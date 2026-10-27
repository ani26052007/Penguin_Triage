"""Failure-map grid over any ``RolloutBackend`` (plan 6.1, ``docs/PLAN_UPDATE.md`` 6.1).

A grid YAML names factor levels. Every combination is a cell, a ``SceneSpec`` "nominal plus named
factor changes"; a ``null`` level means the factor is absent (its variation stays disabled). The
nominal cell (no factor) is always part of the grid, because the candidate rule and the router
compare against it::

    name: gr1_fridge_lighting
    nominal: gr1_fridge_ranch
    factors:
      lighting.intensity: [null, "100.0", "3000.0"]
      lighting.hdr: [null, home_office_robolab]
    episodes: 20
    mode: clean
    policy_checkpoint: nvidia/.../ranch_bottle_into_fridge
    code_base: pw                       # WorkspaceManager base label (CLI only)
    reference: true                     # qualify the RAW reference on every cell
    confidence: 0.95
    candidate: {nominal_min: 0.8, shifted_max: 0.4}
    hosts: [l4-a, l4-b]                 # shards; [] = one host

Per cell, the runner makes **one** ``backend.run`` call for the policy, with seeds ``0 .. episodes-1``.
On ``IsaacBackend`` these are episodes 0..n-1 of build seed 0 (``vla_isaac.seeds``), so a cell is one
environment build, which is what build-time factors (lighting) need. With ``reference: true`` it
also runs the RAW reference (``ActorRef(REFERENCE, <checkpoint>, RAW)``, Mimic on ``IsaacBackend``)
on ``qualification_seeds(rule)`` and qualifies it with ``vla_isaac.reference.mimic.qualify``: the
same rule as Phase 4 (Wilson lower bound vs the configured threshold, a single pass/fail for a
deterministic reference). A cell whose reference is unqualified is flagged ``reference_unqualified``
in the table: lighting must not change the reference, so such a cell changed something besides
appearance (plan 6.1 step 7).

Resume: each finished cell is written to ``<out>/cells/<cell id>.json`` atomically, with a
fingerprint of everything that determines its result (scene, code, policy, seeds, mode, reference,
backend id). A rerun skips cells whose file has the same fingerprint and refuses one whose
fingerprint differs (a stale cell is never mixed into a new table).

Sharding: with ``hosts``, cell i (in grid order) belongs to ``hosts[i % len(hosts)]``. Each host runs
``run(host=...)`` into its own output directory; ``collect`` merges the cell files of every directory
and ``GridTable`` builds the same table one host would have built.

Intervals come from ``ladder.stats`` (``tables.proportion``); nothing here computes one.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ladder.backend import RolloutBackend
from router.schemas import ActionPipeline, ActorKind, ActorRef, CodeRef, Factor, Mode, SceneSpec, short_hash

from vla_isaac.config import ConfigError, read_yaml
from vla_isaac.experiments.tables import csv_text, flat, proportion, write_atomic, write_json
from vla_isaac.reference.mimic import QualificationRule, qualification_seeds, qualify

CELLS_DIR = "cells"
TABLE_JSON = "grid_table.json"
TABLE_CSV = "grid_table.csv"
FLAG_REF_UNQUALIFIED = "reference_unqualified"
FLAG_CANDIDATE = "candidate"
FLAG_NOMINAL = "nominal"


class GridError(RuntimeError):
    """A grid that cannot be run or merged as asked (stale cell, missing cell, conflicting shards)."""


# --------------------------------------------------------------------------- #
# the grid YAML
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CandidateRule:
    """A cell is a candidate failure (plan 6.1 steps 3-4) when the nominal cell's policy rate is at least
    ``nominal_min``, the cell's is at most ``shifted_max``, and its reference is not unqualified."""
    nominal_min: float = 0.8
    shifted_max: float = 0.4


@dataclass(frozen=True)
class GridSpec:
    name: str
    nominal: str
    factors: Mapping[str, tuple[Optional[str], ...]]
    episodes: int
    policy_checkpoint: str
    mode: Mode = Mode.CLEAN
    code_base: Optional[str] = None
    reference: bool = True
    confidence: float = 0.95
    candidate: CandidateRule = field(default_factory=CandidateRule)
    hosts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.factors:
            raise ConfigError(f"grid {self.name!r}: no factors")
        for name, levels in self.factors.items():
            if not levels or len(set(levels)) != len(levels):
                raise ConfigError(f"grid {self.name!r}: factor {name!r} needs distinct levels, got {list(levels)}")
        if self.episodes < 1:
            raise ConfigError(f"grid {self.name!r}: episodes must be >= 1")
        if not 0.0 < self.confidence < 1.0:
            raise ConfigError(f"grid {self.name!r}: confidence must be in (0, 1)")
        if len(set(self.hosts)) != len(self.hosts):
            raise ConfigError(f"grid {self.name!r}: duplicate hosts {list(self.hosts)}")

    @classmethod
    def from_dict(cls, d: Mapping) -> "GridSpec":
        allowed = {"name", "nominal", "factors", "episodes", "mode", "policy_checkpoint", "code_base", "reference",
                   "confidence", "candidate", "hosts"}
        required = {"name", "nominal", "factors", "episodes", "policy_checkpoint"}
        unknown, missing = set(d) - allowed, required - set(d)
        if unknown or missing:
            raise ConfigError(f"grid: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
        if not isinstance(d["factors"], Mapping):
            raise ConfigError("grid: factors must map a factor name to a list of levels")
        factors = {str(k): tuple(None if v is None else str(v) for v in levels) for k, levels in d["factors"].items()}
        cand = d.get("candidate") or {}
        if set(cand) - {"nominal_min", "shifted_max"}:
            raise ConfigError(f"grid: candidate has unknown keys {sorted(set(cand) - {'nominal_min', 'shifted_max'})}")
        return cls(name=str(d["name"]), nominal=str(d["nominal"]), factors=factors, episodes=int(d["episodes"]),
                   policy_checkpoint=str(d["policy_checkpoint"]), mode=Mode(d.get("mode", "clean")),
                   code_base=d.get("code_base"), reference=bool(d.get("reference", True)),
                   confidence=float(d.get("confidence", 0.95)),
                   candidate=CandidateRule(**{k: float(v) for k, v in cand.items()}),
                   hosts=tuple(str(h) for h in d.get("hosts") or ()))

    @classmethod
    def load(cls, path: str | Path) -> "GridSpec":
        return cls.from_dict(read_yaml(path))

    @property
    def policy(self) -> ActorRef:
        return ActorRef(ActorKind.POLICY, self.policy_checkpoint)

    def seeds(self) -> list[int]:
        return list(range(self.episodes))

    def cells(self) -> list["Cell"]:
        """Every combination of levels in YAML order, the nominal cell first."""
        names = list(self.factors)
        scenes = [SceneSpec(self.nominal)]
        for combo in itertools.product(*(self.factors[n] for n in names)):
            scene = SceneSpec(self.nominal, tuple(Factor(n, v) for n, v in zip(names, combo) if v is not None))
            if scene.key() not in {s.key() for s in scenes}:
                scenes.append(scene)
        return [Cell(i, s) for i, s in enumerate(scenes)]

    def host_of(self, index: int) -> Optional[str]:
        return self.hosts[index % len(self.hosts)] if self.hosts else None

    def check_factors(self, known: Sequence[str]) -> None:
        unknown = sorted(set(self.factors) - set(known))
        if unknown:
            raise ConfigError(f"grid {self.name!r}: factors {unknown} are not in the factor table; known {sorted(known)}")


@dataclass(frozen=True)
class Cell:
    index: int
    scene: SceneSpec

    @property
    def cell_id(self) -> str:
        return short_hash(self.scene.key(), 12)

    @property
    def nominal(self) -> bool:
        return not self.scene.factors


# --------------------------------------------------------------------------- #
# running cells
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ReferenceSpec:
    """The RAW reference controller and the Phase 4 qualification rule (task config ``reference``)."""
    actor: ActorRef
    rule: QualificationRule

    def __post_init__(self) -> None:
        if self.actor.kind is not ActorKind.REFERENCE or self.actor.pipeline is not ActionPipeline.RAW:
            raise ConfigError(f"the grid qualifies the RAW reference only, not {self.actor.key()}")

    def to_dict(self) -> dict:
        r = self.rule
        return {"actor": self.actor.key(), "threshold": r.threshold, "confidence": r.confidence, "trials": r.trials}


@dataclass(frozen=True)
class CellResult:
    cell_id: str
    index: int
    scene: SceneSpec
    fingerprint: Mapping[str, Any]
    policy: Mapping[str, Any]                  # tables.proportion
    outcomes: Mapping[int, bool]               # policy seed -> success
    reference: Optional[Mapping[str, Any]]     # Qualification.to_dict(), or None without a reference
    host: Optional[str] = None

    @property
    def reference_unqualified(self) -> bool:
        return self.reference is not None and not self.reference["qualified"]

    def to_dict(self) -> dict:
        return {"cell_id": self.cell_id, "index": self.index, "scene": self.scene.to_dict(),
                "fingerprint": dict(self.fingerprint), "policy": dict(self.policy),
                "outcomes": {str(s): ok for s, ok in sorted(self.outcomes.items())},
                "reference": None if self.reference is None else dict(self.reference), "host": self.host}

    @classmethod
    def from_dict(cls, d: Mapping) -> "CellResult":
        return cls(d["cell_id"], int(d["index"]), SceneSpec.from_dict(d["scene"]), d["fingerprint"], d["policy"],
                   {int(s): bool(ok) for s, ok in d["outcomes"].items()}, d["reference"], d.get("host"))


class GridRunner:
    """Runs the cells of ``spec`` on ``backend`` and writes one file per finished cell under ``out_dir``."""

    def __init__(self, backend: RolloutBackend, spec: GridSpec, code_ref: CodeRef, out_dir: str | Path, *,
                 reference: Optional[ReferenceSpec] = None) -> None:
        if spec.reference and reference is None:
            raise ConfigError(f"grid {spec.name!r} qualifies the reference on every cell but no reference was given")
        self.backend, self.spec, self.code_ref = backend, spec, code_ref
        self.out_dir = Path(out_dir)
        self.reference = reference if spec.reference else None

    def fingerprint(self, cell: Cell) -> dict:
        return {"grid": self.spec.name, "scene": cell.scene.key(), "code": self.code_ref.key(),
                "policy": self.spec.policy.key(), "seeds": self.spec.seeds(), "mode": self.spec.mode.value,
                "confidence": self.spec.confidence, "backend": getattr(self.backend, "backend_id", ""),
                "reference": None if self.reference is None else self.reference.to_dict()}

    def cell_path(self, cell: Cell) -> Path:
        return self.out_dir / CELLS_DIR / f"{cell.cell_id}.json"

    def my_cells(self, host: Optional[str]) -> list[Cell]:
        cells = self.spec.cells()
        if host is None:
            return cells
        if host not in self.spec.hosts:
            raise GridError(f"host {host!r} is not one of the grid's hosts {list(self.spec.hosts)}")
        return [c for c in cells if self.spec.host_of(c.index) == host]

    def run(self, host: Optional[str] = None) -> list[CellResult]:
        """This host's cells (all cells without ``host``), resuming from finished cell files."""
        out = []
        for cell in self.my_cells(host):
            done = self._load(cell)
            out.append(done if done is not None else self._run_cell(cell, host))
        return out

    def _load(self, cell: Cell) -> Optional[CellResult]:
        path = self.cell_path(cell)
        if not path.is_file():
            return None
        done = CellResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if done.fingerprint != self.fingerprint(cell):
            raise GridError(f"cell {cell.scene.key()} at {path} was run with another configuration; move it away "
                            f"or use another output directory (stored {done.fingerprint}, now {self.fingerprint(cell)})")
        return done

    def _run_cell(self, cell: Cell, host: Optional[str]) -> CellResult:
        seeds = self.spec.seeds()
        eps = self.backend.run(cell.scene, self.code_ref, self.spec.policy, seeds, self.spec.mode)
        if [e.seed for e in eps] != seeds:
            raise GridError(f"cell {cell.scene.key()}: the backend returned seeds {[e.seed for e in eps]}, not {seeds}")
        k = sum(bool(e.success) for e in eps)
        ref = None
        if self.reference is not None:
            r = self.backend.run(cell.scene, self.code_ref, self.reference.actor,
                                 qualification_seeds(self.reference.rule), Mode.CLEAN)
            ref = qualify(r, self.reference.rule, cell.scene.key()).to_dict()
        res = CellResult(cell.cell_id, cell.index, cell.scene, self.fingerprint(cell),
                         proportion(k, len(eps), self.spec.confidence), {e.seed: bool(e.success) for e in eps}, ref, host)
        write_json(self.cell_path(cell), res.to_dict())
        return res


def collect(spec: GridSpec, dirs: Sequence[str | Path]) -> list[CellResult]:
    """Every cell of ``spec`` from the cell files under ``dirs`` (one per host), in grid order. A missing
    cell, a cell that is not in the grid, or two different results for one cell is a ``GridError``."""
    cells = {c.cell_id: c for c in spec.cells()}
    found: dict[str, CellResult] = {}
    for d in dirs:
        for path in sorted((Path(d) / CELLS_DIR).glob("*.json")):
            res = CellResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if res.cell_id not in cells or cells[res.cell_id].scene.key() != res.scene.key():
                raise GridError(f"{path} is not a cell of grid {spec.name!r}")
            if res.fingerprint.get("grid") != spec.name:
                raise GridError(f"{path} belongs to grid {res.fingerprint.get('grid')!r}, not {spec.name!r}")
            prev = found.get(res.cell_id)
            if prev is not None and _result_part(prev) != _result_part(res):
                raise GridError(f"cell {res.scene.key()} has two different results ({path})")
            found.setdefault(res.cell_id, res)
    missing = [c.scene.key() for c in spec.cells() if c.cell_id not in found]
    if missing:
        raise GridError(f"grid {spec.name!r}: {len(missing)} cell(s) not run yet: {missing}")
    return [found[c.cell_id] for c in spec.cells()]


def _result_part(r: CellResult) -> dict:
    d = r.to_dict()
    d.pop("host")
    return d


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #

PROP_KEYS = ("k", "n", "rate", "lower", "upper")


class GridTable:
    """The failure map with the reference qualification table next to it (plan 6.1 steps 3, 4, 7)."""

    def __init__(self, spec: GridSpec, results: Sequence[CellResult]) -> None:
        self.spec = spec
        self.results = list(results)
        nominal = [r for r in self.results if not r.scene.factors]
        if len(nominal) != 1:
            raise GridError(f"grid {spec.name!r}: the table needs exactly one nominal cell")
        self.nominal = nominal[0]

    def flags(self, r: CellResult) -> list[str]:
        out = []
        if not r.scene.factors:
            out.append(FLAG_NOMINAL)
        if r.reference_unqualified:
            out.append(FLAG_REF_UNQUALIFIED)
        c = self.spec.candidate
        if (r.scene.factors and not r.reference_unqualified and self.nominal.policy["rate"] >= c.nominal_min
                and r.policy["rate"] <= c.shifted_max):
            out.append(FLAG_CANDIDATE)
        return out

    def rows(self) -> list[dict]:
        rows = []
        for r in self.results:
            rows.append({"cell_id": r.cell_id, "scene": r.scene.key(),
                         "factors": {n: r.scene.factor(n) for n in self.spec.factors},
                         "policy": dict(r.policy), "reference": None if r.reference is None else dict(r.reference),
                         FLAG_REF_UNQUALIFIED: r.reference_unqualified, "flags": self.flags(r)})
        return rows

    def to_dict(self) -> dict:
        return {"grid": self.spec.name, "mode": self.spec.mode.value, "episodes": self.spec.episodes,
                "confidence": self.spec.confidence,
                "candidate": {"nominal_min": self.spec.candidate.nominal_min,
                              "shifted_max": self.spec.candidate.shifted_max},
                "cells": self.rows()}

    def csv_columns(self) -> list[str]:
        return (["cell_id", "scene"] + [f"factor:{n}" for n in self.spec.factors]
                + [f"policy_{k}" for k in PROP_KEYS] + [f"reference_{k}" for k in PROP_KEYS]
                + ["reference_deterministic", FLAG_REF_UNQUALIFIED, "flags"])

    def csv(self) -> str:
        flat_rows = []
        for row in self.rows():
            ref = row["reference"]
            flat_rows.append({"cell_id": row["cell_id"], "scene": row["scene"],
                              **{f"factor:{n}": v for n, v in row["factors"].items()},
                              **flat("policy", row["policy"], PROP_KEYS), **flat("reference", ref, PROP_KEYS),
                              "reference_deterministic": None if ref is None else ref["deterministic"],
                              FLAG_REF_UNQUALIFIED: row[FLAG_REF_UNQUALIFIED], "flags": row["flags"]})
        return csv_text(self.csv_columns(), flat_rows)

    def write(self, out_dir: str | Path) -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        write_json(out_dir / TABLE_JSON, self.to_dict())
        write_atomic(out_dir / TABLE_CSV, self.csv())
        return out_dir / TABLE_JSON, out_dir / TABLE_CSV


# --------------------------------------------------------------------------- #
# CLI (GPU box)
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vla_isaac.experiments.grid", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run this host's cells (resumes); writes the table when the grid is complete")
    r.add_argument("--grid", required=True)
    r.add_argument("--backend", required=True, help="backend config YAML of this host")
    r.add_argument("--out", required=True)
    r.add_argument("--host", default=None, help="this host's name in the grid's hosts (omit: every cell)")
    m = sub.add_parser("merge", help="merge the cell files of several hosts into one table")
    m.add_argument("--grid", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--from", dest="sources", nargs="+", required=True)
    a = ap.parse_args(argv)
    spec = GridSpec.load(a.grid)
    if a.cmd == "merge":
        print(*GridTable(spec, collect(spec, a.sources)).write(a.out), sep="\n")
        return 0
    from vla_isaac.backend.isaac_backend import IsaacBackend

    backend = IsaacBackend.from_config(a.backend)
    spec.check_factors(list(backend.factors.specs))
    if not spec.code_base:
        raise ConfigError(f"grid {spec.name!r}: the CLI needs code_base (a workspace base label)")
    ref = None
    if spec.reference:
        if backend.task.reference is None:
            raise ConfigError(f"task config {backend.task.source} has no reference section")
        tr = backend.task.reference
        ref = ReferenceSpec(ActorRef(ActorKind.REFERENCE, tr.actor_checkpoint, ActionPipeline.RAW), tr.qualification)
    runner = GridRunner(backend, spec, backend.workspaces.base_ref(spec.code_base), a.out, reference=ref)
    runner.run(a.host)
    try:
        print(*GridTable(spec, collect(spec, [a.out])).write(a.out), sep="\n")
    except GridError as exc:
        print(f"cells of this host done; table not written yet: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
