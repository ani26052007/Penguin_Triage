"""Router seeds <-> Arena (build seed, episode index).

Arena seeds a *build* (``--seed`` / ``environment_builder.seed``), not an episode. Each variation
draws from its own CPU generator seeded by ``sha256(f"{seed}:{variation_key}")``, so the k-th
episode to start in a build gets the same run-time values for any ``num_envs``
(``A:isaaclab_arena/variations/variation_base.py:124-131``). A router seed is therefore a pair:

    seed = build_seed * stride + episode_index          (0 <= episode_index < stride)

``stride`` is ``BackendConfig.seed_stride``. The router pads an incident's failing seeds with the
smallest unused seeds (``R:router/replay.py:60-63``), so a replay of 10-30 seeds is one build.

Running episode k of a build means running episodes 0..k of that build (the draws are taken in
start order), so a plan asks each build for ``max(index) + 1`` episodes. The episodes nobody asked
for are not thrown away: the backend puts them into the rollout cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class EpisodeSeed:
    build_seed: int
    episode_index: int


def split_seed(seed: int, stride: int) -> EpisodeSeed:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"router seeds must be non-negative ints, got {seed!r}")
    return EpisodeSeed(seed // stride, seed % stride)


def join_seed(build_seed: int, episode_index: int, stride: int) -> int:
    if not (0 <= episode_index < stride) or build_seed < 0:
        raise ValueError(f"episode index must be in [0, {stride}) and build seed >= 0, "
                         f"got ({build_seed}, {episode_index})")
    return build_seed * stride + episode_index


@dataclass(frozen=True)
class SeedPlan:
    """The caller's seeds grouped by build seed. ``builds`` is ordered by build seed; each value is
    the sorted episode indices requested from that build."""
    seeds: tuple[int, ...]
    stride: int
    builds: Mapping[int, tuple[int, ...]]

    def episodes_needed(self, build_seed: int) -> int:
        return max(self.builds[build_seed]) + 1

    def produced_seeds(self, build_seed: int) -> tuple[int, ...]:
        """Every router seed a run of ``build_seed`` produces (requested or not)."""
        return tuple(join_seed(build_seed, k, self.stride) for k in range(self.episodes_needed(build_seed)))

    def in_caller_order(self, by_seed: Mapping[int, object]) -> list:
        """``by_seed`` values in the order of ``seeds``. A missing seed is a ``KeyError``."""
        missing = [s for s in self.seeds if s not in by_seed]
        if missing:
            raise KeyError(f"no result for seeds {missing}")
        return [by_seed[s] for s in self.seeds]


def plan_seeds(seeds: Sequence[int], stride: int) -> SeedPlan:
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("no seeds to run")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {sorted(s for s in set(seeds) if seeds.count(s) > 1)}")
    groups: dict[int, set[int]] = {}
    for s in seeds:
        es = split_seed(s, stride)
        groups.setdefault(es.build_seed, set()).add(es.episode_index)
    return SeedPlan(seeds, stride, {b: tuple(sorted(groups[b])) for b in sorted(groups)})
