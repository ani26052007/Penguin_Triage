"""Replay agreement (ISSUES #15; plan 3.2.2, 6.1 step 8): run the same seeds twice in clean mode and
count how often the outcome matches.

Per cell the report gives:

* the agreement rate with its Wilson interval (``ladder.stats``; the rate is ``ladder.stats.agreement_rate``);
* the both-success, both-failure and discordant counts (first fail -> second success, and the reverse);
* the overall success rate over both runs, and each run's rate.

A policy that always fails agrees with itself perfectly and says nothing about whether successes
reproduce. So a cell where every agreement is a failure (or every one a success) carries a note, and
``informative`` is False: the agreement number is never shown without that.

Rules:

* **Clean mode only**, and the results must be paired (``logs["paired"]`` not False): an incident-mode
  run (IMPL_PLAN D10) or a Mimic reference (trials, not seeds) is refused.
* **No cache.** A ``CachedBackend`` returns the first run's episodes for the second and hides exactly the
  non-determinism being measured (ISSUES #15); it is refused. ``IsaacBackend`` only writes extra episodes
  to its cache, it never reads them.
* The backend must be asked twice; the second call is a real re-run (``MockBackend``: a repeat, which
  ``replay_noise`` may flip).
* **Both runs' raw files are kept.** A backend whose ``run`` takes ``run_tag`` (``IsaacBackend``) gets
  ``agreement_1`` and ``agreement_2``, so each run writes its Arena log, experiment YAML and episode JSONL
  into its own directory; neither run wipes the other's. Each cell lists both runs' directories
  (``logs["paths"]["run_root"]``) in ``run_roots``.
"""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ladder.backend import RolloutBackend
from ladder.stats import agreement_rate
from router.prereq.rollout_cache import CachedBackend
from router.schemas import ActorRef, CodeRef, EpisodeResult, Mode, SceneSpec

from vla_isaac.experiments.tables import csv_text, flat, proportion, write_atomic, write_json

AGREEMENT_JSON = "replay_agreement.json"
AGREEMENT_CSV = "replay_agreement.csv"
RUN_TAGS = ("agreement_1", "agreement_2")


class AgreementError(RuntimeError):
    """The two runs cannot be compared seed by seed."""


@dataclass(frozen=True)
class CellAgreement:
    scene: str
    seeds: tuple[int, ...]
    agreement: Mapping                    # tables.proportion over seeds: k = seeds whose outcome matched
    both_success: int
    both_failure: int
    first_fail_second_success: int
    first_success_second_fail: int
    success: Mapping                      # tables.proportion over both runs (2n episodes)
    first_rate: float
    second_rate: float
    run_roots: tuple[tuple[str, ...], tuple[str, ...]] = field(default=((), ()))  # per run: its output dirs

    @property
    def discordant(self) -> int:
        return self.first_fail_second_success + self.first_success_second_fail

    @property
    def informative(self) -> bool:
        """False when every agreeing seed agreed on the same outcome (or none agreed on either)."""
        return self.both_success > 0 and self.both_failure > 0

    @property
    def note(self) -> Optional[str]:
        if self.informative:
            return None
        rate = self.success["rate"]
        if self.both_success == 0 and self.both_failure == 0:
            return "no seed agreed on either outcome"
        only = "failures" if self.both_success == 0 else "successes"
        other = "successes" if only == "failures" else "failures"
        return (f"agreement comes only from {only} (overall success rate {rate:.2f}): it says nothing about "
                f"whether {other} reproduce")

    def to_dict(self) -> dict:
        return {"scene": self.scene, "seeds": list(self.seeds), "agreement": dict(self.agreement),
                "both_success": self.both_success, "both_failure": self.both_failure,
                "first_fail_second_success": self.first_fail_second_success,
                "first_success_second_fail": self.first_success_second_fail, "discordant": self.discordant,
                "success": dict(self.success), "first_rate": self.first_rate, "second_rate": self.second_rate,
                "informative": self.informative, "note": self.note,
                "run_roots": {"first": list(self.run_roots[0]), "second": list(self.run_roots[1])}}


def compare_runs(scene: SceneSpec, first: Sequence[EpisodeResult], second: Sequence[EpisodeResult],
                 confidence: float = 0.95) -> CellAgreement:
    seeds = [e.seed for e in first]
    if [e.seed for e in second] != seeds or len(set(seeds)) != len(seeds) or not seeds:
        raise AgreementError(f"{scene.key()}: the two runs must cover the same distinct seeds in the same order")
    for e in list(first) + list(second):
        if e.logs.get("paired") is False:
            raise AgreementError(f"{scene.key()}: seed {e.seed} is not paired (incident mode or a reference's "
                                 f"trials); agreement needs clean-mode seeds")
    a = {e.seed: bool(e.success) for e in first}
    b = {e.seed: bool(e.success) for e in second}
    n = len(seeds)
    both_s = sum(a[s] and b[s] for s in seeds)
    both_f = sum(not a[s] and not b[s] for s in seeds)
    fs = sum(not a[s] and b[s] for s in seeds)
    sf = sum(a[s] and not b[s] for s in seeds)
    agree = both_s + both_f
    assert agree / n == agreement_rate(a, b)
    k1, k2 = sum(a.values()), sum(b.values())
    roots = (_run_roots(first), _run_roots(second))
    if roots[0] and set(roots[0]) & set(roots[1]):
        raise AgreementError(f"{scene.key()}: both runs wrote into {sorted(set(roots[0]) & set(roots[1]))}; "
                             f"the second run would have replaced the first run's raw files")
    return CellAgreement(scene.key(), tuple(seeds), proportion(agree, n, confidence), both_s, both_f, fs, sf,
                         proportion(k1 + k2, 2 * n, confidence), k1 / n, k2 / n, roots)


def _run_roots(results: Sequence[EpisodeResult]) -> tuple[str, ...]:
    roots = {(e.logs.get("paths") or {}).get("run_root") for e in results}
    return tuple(sorted(r for r in roots if r))


def _tagged_run(backend: RolloutBackend, tag: str):
    """``backend.run`` with ``run_tag=tag`` when the backend takes one (separate output directories)."""
    try:
        takes_tag = "run_tag" in inspect.signature(backend.run).parameters
    except (TypeError, ValueError):
        takes_tag = False
    if not takes_tag:
        return backend.run
    return lambda *args: backend.run(*args, run_tag=tag)


@dataclass(frozen=True)
class AgreementReport:
    cells: tuple[CellAgreement, ...]
    pooled: Mapping                       # tables.proportion over every seed of every cell
    confidence: float

    @property
    def informative(self) -> bool:
        return all(c.informative for c in self.cells)

    def to_dict(self) -> dict:
        return {"confidence": self.confidence, "pooled_agreement": dict(self.pooled),
                "informative": self.informative, "cells": [c.to_dict() for c in self.cells]}

    def csv(self) -> str:
        keys = ("k", "n", "rate", "lower", "upper")
        cols = (["scene"] + [f"agreement_{k}" for k in keys]
                + ["both_success", "both_failure", "first_fail_second_success", "first_success_second_fail",
                   "success_rate", "first_rate", "second_rate", "informative", "note"])
        rows = [{"scene": c.scene, **flat("agreement", c.agreement, keys), "both_success": c.both_success,
                 "both_failure": c.both_failure, "first_fail_second_success": c.first_fail_second_success,
                 "first_success_second_fail": c.first_success_second_fail, "success_rate": c.success["rate"],
                 "first_rate": c.first_rate, "second_rate": c.second_rate, "informative": c.informative,
                 "note": c.note} for c in self.cells]
        return csv_text(cols, rows)

    def write(self, out_dir: str | Path) -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        write_json(out_dir / AGREEMENT_JSON, self.to_dict())
        write_atomic(out_dir / AGREEMENT_CSV, self.csv())
        return out_dir / AGREEMENT_JSON, out_dir / AGREEMENT_CSV


def measure_agreement(backend: RolloutBackend, scenes: Sequence[SceneSpec], code_ref: CodeRef, policy: ActorRef,
                      seeds: Sequence[int], confidence: float = 0.95) -> AgreementReport:
    """Each scene's seeds twice in clean mode on ``backend`` (never through a cache)."""
    if isinstance(backend, CachedBackend):
        raise AgreementError("replay agreement through a CachedBackend would read the first run back as the second; "
                             "pass the uncached backend (ISSUES #15)")
    seeds = [int(s) for s in seeds]
    cells = []
    for scene in scenes:
        first = _tagged_run(backend, RUN_TAGS[0])(scene, code_ref, policy, seeds, Mode.CLEAN)
        second = _tagged_run(backend, RUN_TAGS[1])(scene, code_ref, policy, seeds, Mode.CLEAN)
        cells.append(compare_runs(scene, first, second, confidence))
    agree = sum(c.agreement["k"] for c in cells)
    total = sum(c.agreement["n"] for c in cells)
    return AgreementReport(tuple(cells), proportion(agree, total, confidence), confidence)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Agreement on the cells of a grid YAML, on the GPU box."""
    from router.schemas import ActorKind

    from vla_isaac.backend.isaac_backend import IsaacBackend
    from vla_isaac.experiments.grid import GridSpec

    ap = argparse.ArgumentParser(prog="python -m vla_isaac.experiments.replay_agreement")
    ap.add_argument("--grid", required=True, help="grid YAML: its cells, policy, code base and episode count")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nominal-only", action="store_true", help="only the nominal cell")
    a = ap.parse_args(argv)
    spec = GridSpec.load(a.grid)
    backend = IsaacBackend.from_config(a.backend)
    cells = spec.cells()[:1] if a.nominal_only else spec.cells()
    report = measure_agreement(backend, [c.scene for c in cells], backend.workspaces.base_ref(spec.code_base),
                               ActorRef(ActorKind.POLICY, spec.policy_checkpoint), spec.seeds(), spec.confidence)
    print(*report.write(a.out), sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
