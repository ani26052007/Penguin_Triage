"""GR00T ``statistics.json`` -> ``TrainingSpec.norm_stats``.

GR00T stores ``{embodiment_tag: {modality: {joint_group: {stat: [per-dimension values]}}}}``
(``G:gr00t/data/state_action/state_action_processor.py:93-104``). The router's action contract reads
``norm_stats[<modality>][<stat>]`` as one flat vector over the modality's layout
(``R:router/stage1/action_contract.py:70-72``). ``flatten_statistics`` concatenates the groups in
the modality config's key order, for the stats every group has.

``training_norm_stats`` is the one function both sides of the config contract call: the training-spec
extraction (``TrainingSpec.norm_stats``) and vla-isaac's eval-config loader (the derived
``server.effective_norm_stats``). It reads the group order from the checkpoint's
``processor_config.json`` (the checkpoint's modality config wins, facts §8), falling back to the task
config's ``norm_stats_groups`` only when the checkpoint has no processor config, and flattens the
checkpoint's ``statistics.json`` or the given override file. Same checkpoint, same function, so the two
values cannot differ in structure; an override file can only differ in its numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

STATISTICS_FILE = "statistics.json"
PROCESSOR_CONFIG_FILE = "processor_config.json"


class NormStatsError(ValueError):
    """Statistics that do not have the expected embodiment, modality or groups."""


def flatten_statistics(statistics: Mapping[str, Any], embodiment: str,
                       groups: Mapping[str, Sequence[str]]) -> dict[str, dict[str, list[float]]]:
    if embodiment not in statistics:
        raise NormStatsError(f"no statistics for embodiment {embodiment!r}; have {sorted(statistics)}")
    per_modality = statistics[embodiment]
    out: dict[str, dict[str, list[float]]] = {}
    for modality, names in groups.items():
        stats = per_modality.get(modality)
        if not isinstance(stats, Mapping):
            raise NormStatsError(f"{embodiment}: no {modality!r} statistics")
        missing = [g for g in names if g not in stats]
        if missing:
            raise NormStatsError(f"{embodiment}.{modality}: no statistics for groups {missing}")
        common = set.intersection(*(set(stats[g]) for g in names))
        flat: dict[str, list[float]] = {}
        for stat in sorted(common):
            values: list[float] = []
            for g in names:
                v = stats[g][stat]
                # VERIFY(V-stats-format): every stat is a flat per-dimension list in statistics.json
                if not isinstance(v, list) or any(isinstance(x, list) for x in v):
                    raise NormStatsError(f"{embodiment}.{modality}.{g}.{stat} is not a flat list")
                values.extend(float(x) for x in v)
            flat[stat] = values
        out[modality] = flat
    return out


def load_flat_statistics(path: str | Path, embodiment: str,
                         groups: Mapping[str, Sequence[str]]) -> dict[str, dict[str, list[float]]]:
    with open(path, encoding="utf-8") as f:
        return flatten_statistics(json.load(f), embodiment, groups)


@dataclass(frozen=True)
class TrainingNormStats:
    stats: dict[str, dict[str, list[float]]]
    groups: dict[str, tuple[str, ...]]          # modality -> group order used
    groups_source: str                          # "checkpoint:processor_config.json" or "task config"
    statistics: Path                            # the file the numbers came from


def checkpoint_groups(checkpoint_dir: str | Path, embodiment: str,
                      modalities: Sequence[str]) -> Optional[dict[str, tuple[str, ...]]]:
    """Group order per modality from the checkpoint's processor config
    (``processor_kwargs.modality_configs[<embodiment>][<modality>].modality_keys``); None if the
    checkpoint has no ``processor_config.json``."""
    path = Path(checkpoint_dir) / PROCESSOR_CONFIG_FILE
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        configs = json.load(f).get("processor_kwargs", {}).get("modality_configs", {})
    if embodiment not in configs:
        raise NormStatsError(f"{path}: no modality config for embodiment {embodiment!r}; have {sorted(configs)}")
    out = {}
    for m in modalities:
        keys = configs[embodiment].get(m, {}).get("modality_keys")
        if not keys:
            raise NormStatsError(f"{path}: no {m!r} modality keys for {embodiment!r}")
        out[m] = tuple(str(k) for k in keys)
    return out


def training_norm_stats(checkpoint_dir: str | Path, embodiment: str, fallback_groups: Mapping[str, Sequence[str]],
                        *, statistics: Optional[str | Path] = None) -> TrainingNormStats:
    """The normalisation statistics as ``TrainingSpec.norm_stats`` holds them, for ``checkpoint_dir``:
    the checkpoint's ``statistics.json`` (or ``statistics``, an override file of the same format),
    flattened in the checkpoint's group order (``checkpoint_groups``, else ``fallback_groups``)."""
    groups = checkpoint_groups(checkpoint_dir, embodiment, tuple(fallback_groups))
    source = f"checkpoint:{PROCESSOR_CONFIG_FILE}"
    if groups is None:
        groups, source = {m: tuple(g) for m, g in fallback_groups.items()}, "task config"
    path = Path(statistics) if statistics is not None else Path(checkpoint_dir) / STATISTICS_FILE
    if not path.is_file():
        raise NormStatsError(f"no normalisation statistics at {path}")
    return TrainingNormStats(load_flat_statistics(path, embodiment, groups), groups, source, path)
