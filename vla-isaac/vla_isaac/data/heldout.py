"""The held-out split (plan 6.3): train on some shifted conditions, evaluate on the others.

The split is over **conditions** (factor values), never over episodes, so no condition and no episode
is on both sides. It is a pure function of the condition set and ``heldout.seed``: the shifted
condition keys are sorted, shuffled with ``random.Random(seed)``, and the first
``round(fraction * n)`` (at least one, and at least one left for training) are held out.

The **nominal** condition (no factors) is never held out and never trained on through this split: its
episodes form a separate **regression** set, evaluated after fine-tuning to check that nominal success
did not drop (plan 6.3). The mixer's nominal source is a separate input (``vla_isaac.data.mixer``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from vla_isaac.data.config import Condition, HeldoutConfig, condition_key


class SplitError(ValueError):
    """The conditions or episodes cannot be split as asked."""


@dataclass(frozen=True)
class EpisodeRef:
    """One LeRobot episode: ``dataset`` (a source id) and its ``episode_index``, recorded under ``condition``."""
    dataset: str
    episode_index: int
    condition: Mapping[str, str]

    @property
    def id(self) -> str:
        return f"{self.dataset}:{self.episode_index}"

    @property
    def condition_key(self) -> str:
        return condition_key(self.condition)


@dataclass(frozen=True)
class ConditionSplit:
    seed: int
    fraction: float
    train: tuple[str, ...]                # shifted condition keys, sorted
    heldout: tuple[str, ...]
    regression: tuple[str, ...]           # ("nominal",) when the nominal condition was given, else ()

    def to_dict(self) -> dict:
        return {"seed": self.seed, "fraction": self.fraction, "train": list(self.train), "heldout": list(self.heldout),
                "regression": list(self.regression), "keyed_by": "condition"}


def split_conditions(conditions: Sequence[Condition], cfg: HeldoutConfig) -> ConditionSplit:
    keys = [condition_key(c) for c in conditions]
    if len(set(keys)) != len(keys):
        raise SplitError(f"duplicate conditions {sorted(k for k in set(keys) if keys.count(k) > 1)}")
    shifted = sorted(k for k in keys if k != "nominal")
    if len(shifted) < 2:
        raise SplitError(f"a held-out split needs at least two shifted conditions, got {shifted}")
    order = list(shifted)
    random.Random(cfg.seed).shuffle(order)
    n_held = min(max(1, round(cfg.fraction * len(order))), len(order) - 1)
    return ConditionSplit(cfg.seed, cfg.fraction, tuple(sorted(order[n_held:])), tuple(sorted(order[:n_held])),
                          ("nominal",) if "nominal" in keys else ())


@dataclass(frozen=True)
class EpisodeSplit:
    conditions: ConditionSplit
    train: tuple[EpisodeRef, ...]
    heldout: tuple[EpisodeRef, ...]
    regression: tuple[EpisodeRef, ...]

    def to_dict(self) -> dict:
        def ids(eps):
            return [e.id for e in eps]
        return {**self.conditions.to_dict(), "train_episodes": ids(self.train), "heldout_episodes": ids(self.heldout),
                "regression_episodes": ids(self.regression)}


def split_episodes(episodes: Sequence[EpisodeRef], cfg: HeldoutConfig) -> EpisodeSplit:
    """Every episode goes where its condition goes. An episode id under two conditions is refused."""
    seen: dict[str, str] = {}
    for e in episodes:
        if seen.setdefault(e.id, e.condition_key) != e.condition_key:
            raise SplitError(f"episode {e.id} is recorded under two conditions ({seen[e.id]!r} and {e.condition_key!r})")
    if len(seen) != len(episodes):
        raise SplitError("duplicate episode ids")
    by_cond: dict[str, list[EpisodeRef]] = {}
    for e in episodes:
        by_cond.setdefault(e.condition_key, []).append(e)
    split = split_conditions([next(iter(v)).condition for v in by_cond.values()], cfg)

    def pick(keys):
        return tuple(sorted((e for k in keys for e in by_cond[k]), key=lambda e: (e.dataset, e.episode_index)))
    return EpisodeSplit(split, pick(split.train), pick(split.heldout), pick(split.regression))
