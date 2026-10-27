"""The training mix (plan 5.5): nominal LeRobot episodes plus shifted ones at a configured ratio.

``select_mix(nominal, shifted, cfg)`` is pure and deterministic for ``mix.seed``. ``mix.ratio`` (default
0.5) is the fraction of **shifted** episodes in the mix. Every shifted episode is used when the nominal
pool is large enough; otherwise every nominal episode is used and the shifted ones are sampled down.
Counts are exact (``floor(x + 1/2)`` on rationals, never float rounding). An empty source is refused:
a "mix" with one side missing would silently be a plain fine-tune on the other.

``write_mixed_dataset`` writes one GR00T LeRobot v2 directory, because ``launch_finetune.py`` takes a
single ``--dataset-path`` (``G:gr00t/experiment/launch_finetune.py:40-52``). GR00T finds an episode's
parquet and videos through ``meta/episodes.jsonl`` ``episode_index`` and the ``data_path`` /
``video_path`` patterns of ``meta/info.json`` (``G:gr00t/data/dataset/lerobot_episode_loader.py:150-182,
283-302,465``), so the files are copied under new indices and the parquet is not rewritten. Its
annotation task indices go through ``meta/tasks.jsonl``, so the sources must have the same tasks map
(and the same modality, features and patterns); otherwise the mix is refused. ``meta/stats.json`` is
**not** copied: GR00T's dataset factory computes it when it is missing and keeps a stale one
(``G:gr00t/data/dataset/factory.py:42-47``, ``G:gr00t/data/stats.py:98-108``). ``meta/vla_mix.json``
maps every new index to its source episode.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from fractions import Fraction
from math import floor
from pathlib import Path
from typing import Any, Mapping, Sequence

from vla_isaac.data.config import Condition, MixConfig
from vla_isaac.data.heldout import EpisodeRef

MIX_FILE = "vla_mix.json"
NOT_COPIED = ("stats.json", "relative_stats.json")


class MixError(ValueError):
    """The sources cannot be mixed as asked."""


def _half_up(x: Fraction) -> int:
    return floor(x + Fraction(1, 2))


@dataclass(frozen=True)
class MixSelection:
    ratio: float
    seed: int
    nominal: tuple[EpisodeRef, ...]
    shifted: tuple[EpisodeRef, ...]
    available: Mapping[str, int]

    @property
    def episodes(self) -> tuple[EpisodeRef, ...]:
        return self.nominal + self.shifted

    def to_dict(self) -> dict:
        n_s, n = len(self.shifted), len(self.episodes)
        return {"ratio": self.ratio, "seed": self.seed, "nominal_count": len(self.nominal), "shifted_count": n_s,
                "achieved_ratio": n_s / n, "available": dict(self.available),
                "nominal": [e.id for e in self.nominal], "shifted": [e.id for e in self.shifted]}


def mix_counts(n_nominal: int, n_shifted: int, ratio: float) -> tuple[int, int]:
    """``(nominal, shifted)`` counts for ``ratio`` = shifted / total."""
    if n_nominal < 1 or n_shifted < 1:
        raise MixError(f"both sources must hold episodes (nominal {n_nominal}, shifted {n_shifted})")
    r = Fraction(ratio).limit_denominator(10_000)
    need_nominal = _half_up(n_shifted * (1 - r) / r)
    if need_nominal <= n_nominal:
        counts = need_nominal, n_shifted
    else:
        counts = n_nominal, min(n_shifted, _half_up(n_nominal * r / (1 - r)))
    if 0 in counts:
        raise MixError(f"ratio {ratio} with {n_nominal} nominal and {n_shifted} shifted episodes leaves one side empty")
    return counts


def select_mix(nominal: Sequence[EpisodeRef], shifted: Sequence[EpisodeRef], cfg: MixConfig) -> MixSelection:
    for name, eps in (("nominal", nominal), ("shifted", shifted)):
        ids = [e.id for e in eps]
        if len(set(ids)) != len(ids):
            raise MixError(f"duplicate {name} episode ids")
    if {e.id for e in nominal} & {e.id for e in shifted}:
        raise MixError("an episode is in both sources")
    k_n, k_s = mix_counts(len(nominal), len(shifted), cfg.ratio)
    rng = random.Random(cfg.seed)

    def sample(eps, k):
        pool = sorted(eps, key=lambda e: (e.dataset, e.episode_index))
        return tuple(sorted(rng.sample(pool, k), key=lambda e: (e.dataset, e.episode_index)))
    shifted_pick = sample(shifted, k_s)
    return MixSelection(cfg.ratio, cfg.seed, sample(nominal, k_n), shifted_pick,
                        {"nominal": len(nominal), "shifted": len(shifted)})


# --------------------------------------------------------------------------- #
# LeRobot directories
# --------------------------------------------------------------------------- #

def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class LeRobotSource:
    """A GR00T LeRobot v2 directory (``meta/info.json``, ``episodes.jsonl``, ``tasks.jsonl``,
    ``modality.json``) recorded under one condition."""
    id: str
    path: Path
    condition: Condition

    def meta(self, name: str) -> Path:
        return Path(self.path) / "meta" / name

    def info(self) -> dict:
        return _json(self.meta("info.json"))

    def episode_lines(self) -> list[dict]:
        return _jsonl(self.meta("episodes.jsonl"))

    def episodes(self) -> list[EpisodeRef]:
        return [EpisodeRef(self.id, int(e["episode_index"]), dict(self.condition)) for e in self.episode_lines()]

    def tasks(self) -> dict[int, str]:
        return {int(t["task_index"]): str(t["task"]) for t in _jsonl(self.meta("tasks.jsonl"))}

    def episode_files(self, index: int, info: Mapping[str, Any]) -> list[tuple[str, Path]]:
        """``(pattern kind, relative path)`` of the parquet and every video of episode ``index``."""
        chunk = index // int(info["chunks_size"])
        out = [("data", Path(info["data_path"].format(episode_chunk=chunk, episode_index=index)))]
        for key in video_keys(info):
            out.append((key, Path(info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=index))))
        return out


def video_keys(info: Mapping[str, Any]) -> list[str]:
    return sorted(k for k, f in (info.get("features") or {}).items() if isinstance(f, Mapping) and f.get("dtype") == "video")


SAME = ("features", "data_path", "video_path", "chunks_size", "fps")


def _check_compatible(sources: Sequence[LeRobotSource]) -> None:
    first = sources[0]
    info0, tasks0, mod0 = first.info(), first.tasks(), _json(first.meta("modality.json"))
    for s in sources[1:]:
        info = s.info()
        for k in SAME:
            if info.get(k) != info0.get(k):
                raise MixError(f"{s.id} and {first.id} differ in meta/info.json {k!r}")
        if _json(s.meta("modality.json")) != mod0:
            raise MixError(f"{s.id} and {first.id} differ in meta/modality.json")
        if s.tasks() != tasks0:
            raise MixError(f"{s.id} and {first.id} have different meta/tasks.jsonl ({s.tasks()} vs {tasks0}); the parquet "
                           f"task indices are not rewritten, so the instruction strings must match (set the converter "
                           f"YAML's language_instruction to the nominal set's string)")


def write_mixed_dataset(selection: MixSelection, sources: Sequence[LeRobotSource], out: str | Path) -> Path:
    """Write the selected episodes as one LeRobot directory at ``out`` (refused when it exists)."""
    out = Path(out)
    if out.exists():
        raise MixError(f"{out} exists; the mix is written into a new directory")
    by_id = {s.id: s for s in sources}
    used = sorted({e.dataset for e in selection.episodes})
    missing = [d for d in used if d not in by_id]
    if missing:
        raise MixError(f"no LeRobot source for {missing}")
    srcs = [by_id[d] for d in used]
    _check_compatible(srcs)
    info = srcs[0].info()
    lines = {s.id: {int(e["episode_index"]): e for e in s.episode_lines()} for s in srcs}
    tmp = out.with_name(f".{out.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "meta").mkdir(parents=True)
    new_lines, mapping, frames = [], [], 0
    for new, ep in enumerate(selection.episodes):
        src = by_id[ep.dataset]
        if ep.episode_index not in lines[ep.dataset]:
            raise MixError(f"episode {ep.id} is not in {src.path}/meta/episodes.jsonl")
        for (kind, old_rel), (_, new_rel) in zip(src.episode_files(ep.episode_index, info), src.episode_files(new, info)):
            if not (src.path / old_rel).is_file():
                raise MixError(f"episode {ep.id}: missing {kind} file {src.path / old_rel}")
            (tmp / new_rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src.path / old_rel, tmp / new_rel)
        line = {**lines[ep.dataset][ep.episode_index], "episode_index": new}
        frames += int(line.get("length", 0))
        new_lines.append(line)
        mapping.append({"episode_index": new, "source": ep.dataset, "source_episode_index": ep.episode_index,
                        "condition": dict(ep.condition), "role": "shifted" if ep in selection.shifted else "nominal"})
    n = len(new_lines)
    info.update(total_episodes=n, total_frames=frames, total_videos=n * len(video_keys(info)))
    if "total_chunks" in info:
        info["total_chunks"] = (n - 1) // int(info["chunks_size"]) + 1
    if "splits" in info:
        info["splits"] = {"train": f"0:{n}"}
    (tmp / "meta" / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")
    (tmp / "meta" / "episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in new_lines), encoding="utf-8")
    for name in ("tasks.jsonl", "modality.json"):
        shutil.copyfile(srcs[0].meta(name), tmp / "meta" / name)
    (tmp / "meta" / MIX_FILE).write_text(json.dumps({**selection.to_dict(), "episodes": mapping,
                                                     "not_copied": list(NOT_COPIED)}, indent=2), encoding="utf-8")
    tmp.rename(out)
    return out
