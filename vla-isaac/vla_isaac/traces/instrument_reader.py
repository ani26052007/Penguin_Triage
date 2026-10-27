"""The policy wrapper's instrument files -> per-episode frames, step pairs and raw actions.

``policy_wrapper/instrument.py`` writes one JSON per finished episode (``env<id>_ep<n>.json``) when the
process has ``VLA_INSTRUMENT_DIR`` set; ``IsaacBackend`` sets it, and the other three variables below,
for runs that record traces. The names are the wrapper's (a test checks they agree).

``raw_actions`` here are per applied action, as ``EpisodeTrace`` wants them: for the step pair
``(obs_step, applied_step, chunk_offset)``, row ``chunk_offset`` of the chunk the server returned for
the request at ``obs_step``, its groups concatenated in the order the server sent them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from router.prereq.trace import FrameSample, StepPair

ENV_DIR = "VLA_INSTRUMENT_DIR"
ENV_FRAME_EVERY = "VLA_INSTRUMENT_FRAME_EVERY"
ENV_PROFILE_BUCKETS = "VLA_INSTRUMENT_PROFILE_BUCKETS"
ENV_OBS_KEYS = "VLA_INSTRUMENT_OBS_KEYS"
FORMAT = "policy_wrapper.instrument/1"


class InstrumentError(RuntimeError):
    """An instrument file is missing, of another format, or inconsistent."""


@dataclass(frozen=True)
class InstrumentEpisode:
    env_id: int
    episode_in_env: int
    start_index: int
    num_steps: int
    frames: tuple[FrameSample, ...]
    step_pairs: tuple[StepPair, ...]
    raw_actions: tuple[tuple[float, ...], ...]
    action_keys: tuple[str, ...]
    observations: Mapping[str, tuple[tuple[float, ...], ...]]
    path: Path


def read_episode(path: str | Path) -> InstrumentEpisode:
    path = Path(path)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise InstrumentError(f"{path}: {e}") from e
    if d.get("format") != FORMAT:
        raise InstrumentError(f"{path}: format {d.get('format')!r}, expected {FORMAT!r}")
    chunks = {}
    for c in d["chunks"]:
        chunks[int(c["obs_step"])] = c["raw"]
    pairs, raw, keys = [], [], None
    for o, a, k in d["step_pairs"]:
        try:
            pairs.append(StepPair(int(o), int(a), int(k)))
        except ValueError as e:
            raise InstrumentError(f"{path}: {e}") from e
        chunk = chunks.get(int(o))
        if chunk is None:
            raise InstrumentError(f"{path}: action applied at step {a} has no chunk logged at step {o}")
        if keys is None:
            keys = tuple(chunk)
        elif tuple(chunk) != keys:
            raise InstrumentError(f"{path}: chunk at step {o} has groups {list(chunk)}, expected {list(keys)}")
        raw.append(tuple(float(v) for key in keys for v in chunk[key][int(k)]))
    return InstrumentEpisode(
        env_id=int(d["env_id"]), episode_in_env=int(d["episode_in_env"]), start_index=int(d["start_index"]),
        num_steps=int(d["num_steps"]), frames=tuple(FrameSample.from_dict(f) for f in d["frames"]),
        step_pairs=tuple(pairs), raw_actions=tuple(raw), action_keys=keys or (),
        observations={k: tuple(tuple(float(x) for x in r) for r in rows) for k, rows in d["observations"].items()},
        path=path)


def read_instrument_dir(directory: str | Path) -> dict[tuple[int, int], InstrumentEpisode]:
    """Every episode file in ``directory``, keyed by ``(env_id, episode_in_env)``."""
    out: dict[tuple[int, int], InstrumentEpisode] = {}
    for p in sorted(Path(directory).glob("env*_ep*.json")):
        ep = read_episode(p)
        key = (ep.env_id, ep.episode_in_env)
        if key in out:
            raise InstrumentError(f"{p}: episode {key} also in {out[key].path}")
        out[key] = ep
    return out
