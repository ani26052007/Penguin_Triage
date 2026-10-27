"""Clean vs incident nominal success (plan 3.1, ``docs/PLAN_UPDATE.md`` 6.1 step 9).

Runs the nominal scene in clean mode (one env, serial) and in incident mode (N envs, shared server),
and reports both rates with Wilson intervals and whether they differ. Everything statistical is
``ladder.stats``: ``wilson_interval`` (through ``tables.proportion``) and ``fisher_exact``.

**Unpaired.** Incident-mode episodes cannot be matched to clean-mode seeds: with N > 1 envs the start
order is not in Arena's record (IMPL_PLAN D10), and ``IsaacBackend`` marks those results
``paired: False``. So the comparison is always the unpaired two-sided Fisher exact test, whatever the
replay agreement, and the report says ``pairing: "unpaired"``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ladder.backend import RolloutBackend
from ladder.stats import fisher_exact
from router.schemas import ActorRef, CodeRef, Mode, SceneSpec

from vla_isaac.experiments.tables import proportion, write_json

MODE_GAP_JSON = "mode_gap.json"
PAIRING = "unpaired"
PAIRING_REASON = ("incident-mode episodes are not paired with clean-mode seeds: their start order is not "
                  "recorded with N > 1 envs (IMPL_PLAN D10)")


@dataclass(frozen=True)
class ModeGap:
    scene: str
    clean: Mapping                       # tables.proportion
    incident: Mapping
    p_value: float
    alpha: float
    incident_paired_in_logs: Optional[bool]   # what the backend said; the comparison is unpaired regardless

    @property
    def gap(self) -> float:
        """Clean minus incident success rate."""
        return self.clean["rate"] - self.incident["rate"]

    @property
    def differ(self) -> bool:
        return self.p_value < self.alpha

    def to_dict(self) -> dict:
        return {"scene": self.scene, "clean": dict(self.clean), "incident": dict(self.incident), "gap": self.gap,
                "test": "fisher_exact (ladder.stats), two-sided", "p_value": self.p_value, "alpha": self.alpha,
                "differ": self.differ, "pairing": PAIRING, "pairing_reason": PAIRING_REASON,
                "incident_paired_in_logs": self.incident_paired_in_logs}

    def write(self, out_dir: str | Path) -> Path:
        path = Path(out_dir) / MODE_GAP_JSON
        write_json(path, self.to_dict())
        return path


def measure_mode_gap(backend: RolloutBackend, scene: SceneSpec, code_ref: CodeRef, policy: ActorRef,
                     seeds: Sequence[int], *, incident_seeds: Optional[Sequence[int]] = None,
                     confidence: float = 0.95, alpha: float = 0.05) -> ModeGap:
    """``scene`` (the nominal one) in both modes. ``incident_seeds`` defaults to ``seeds``."""
    seeds = [int(s) for s in seeds]
    inc_seeds = seeds if incident_seeds is None else [int(s) for s in incident_seeds]
    if not seeds or not inc_seeds:
        raise ValueError("need seeds in both modes")
    clean = backend.run(scene, code_ref, policy, seeds, Mode.CLEAN)
    incident = backend.run(scene, code_ref, policy, inc_seeds, Mode.INCIDENT)
    kc, ki = sum(bool(e.success) for e in clean), sum(bool(e.success) for e in incident)
    flags = {e.logs.get("paired") for e in incident}
    said = flags.pop() if len(flags) == 1 else None
    return ModeGap(scene.key(), proportion(kc, len(clean), confidence), proportion(ki, len(incident), confidence),
                   fisher_exact(kc, len(clean), ki, len(incident)), alpha, said)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """The mode gap on a grid YAML's nominal cell, on the GPU box."""
    from router.schemas import ActorKind

    from vla_isaac.backend.isaac_backend import IsaacBackend
    from vla_isaac.experiments.grid import GridSpec

    ap = argparse.ArgumentParser(prog="python -m vla_isaac.experiments.mode_gap")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)
    spec = GridSpec.load(a.grid)
    backend = IsaacBackend.from_config(a.backend)
    gap = measure_mode_gap(backend, spec.cells()[0].scene, backend.workspaces.base_ref(spec.code_base),
                           ActorRef(ActorKind.POLICY, spec.policy_checkpoint), spec.seeds(),
                           confidence=spec.confidence, alpha=a.alpha)
    print(gap.write(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
