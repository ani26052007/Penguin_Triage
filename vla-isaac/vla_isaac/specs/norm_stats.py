"""GR00T ``statistics.json`` -> ``TrainingSpec.norm_stats``.

GR00T stores ``{embodiment_tag: {modality: {joint_group: {stat: [per-dimension values]}}}}``
(``G:gr00t/data/state_action/state_action_processor.py:93-104``). The router's action contract reads
``norm_stats[<modality>][<stat>]`` as one flat vector over the modality's layout
(``R:router/stage1/action_contract.py:70-72``). ``flatten_statistics`` concatenates the groups in
the modality config's key order, for the stats every group has. The training-spec extraction
and vla-isaac's eval-config loader both use it, so the two sides of the config contract have the
same structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


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
