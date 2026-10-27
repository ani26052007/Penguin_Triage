"""Arena's HDF5 trajectory dataset -> per-episode arrays.

A run with ``record_trajectories`` writes Isaac Lab's recorder dataset into the run directory
(``dataset_<run>_rebuild<i>``, ``A:isaaclab_arena/evaluation/run_execution.py:168-169``): one group per
exported episode under ``data/`` (``demo_0``, ``demo_1``, ...). Each demo carries Arena's episode
identity (``episode_id/env_id``, ``episode_id/episode_in_env``,
``A:isaaclab_arena/terms/recorders.py:71-94``), which joins it to its JSONL record. Every other key path
comes from the task config's ``traces.hdf5`` (Isaac Lab's recorder layout is unread,
VERIFY(V-hdf5-keys); which episodes are exported and when, VERIFY(V-hdf5-export)).

Files are read through ``TrajectoryStore``, so tests use small synthetic h5py files and the reader's
logic never imports h5py itself (``H5pyStore`` does, when it opens a file).

Joint columns: ``joint_cmd`` is in the action joint order (``traces.joints.action``) and ``joint_pos``
in the state joint order (``traces.joints.state``); the tracked positions are reordered to the action
joints so the two series line up column by column (VERIFY(V-hdf5-joint-order)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

from vla_isaac.traces.layout import TraceLayout


class TrajectoryError(RuntimeError):
    """A dataset is missing an expected key or is not shaped as the layout says."""


class TrajectoryStore(Protocol):
    """One dataset file: its demo names and the array at a key path inside a demo."""

    path: Path

    def demos(self) -> list[str]: ...

    def read(self, demo: str, key: str) -> Optional[Any]:
        """The array at ``key`` in ``demo`` (anything ``tolist``-able or nested lists); None when absent."""


class H5pyStore:
    """``TrajectoryStore`` over an h5py file (imported when the store is opened)."""

    def __init__(self, path: str | Path, demo_group: str) -> None:
        import h5py

        self.path = Path(path)
        self._f = h5py.File(self.path, "r")
        self._group = demo_group

    def demos(self) -> list[str]:
        g = self._f.get(self._group)
        return sorted(g.keys(), key=_demo_order) if g is not None else []

    def read(self, demo: str, key: str) -> Optional[Any]:
        node = self._f.get(f"{self._group}/{demo}/{key}")
        return None if node is None or not hasattr(node, "shape") else node[()]

    def close(self) -> None:
        self._f.close()


def _demo_order(name: str) -> tuple:
    head, _, tail = name.rpartition("_")
    return (head, int(tail)) if tail.isdigit() else (name, -1)


def _rows(x: Any) -> list[list[float]]:
    data = x.tolist() if hasattr(x, "tolist") else x
    return [[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])] for r in data]


def _scalar(x: Any) -> int:
    data = x.tolist() if hasattr(x, "tolist") else x
    while isinstance(data, (list, tuple)):
        if len(data) != 1:
            raise TrajectoryError(f"expected one value, got {data!r}")
        data = data[0]
    return int(data)


def _finite(row: Sequence[float]) -> bool:
    return all(math.isfinite(v) for v in row)


@dataclass(frozen=True)
class EpisodeArrays:
    env_id: int
    episode_in_env: int
    source: str                                     # "<file>:<demo>"
    joint_cmd: tuple[tuple[float, ...], ...]        # action joint order
    joint_pos: tuple[tuple[float, ...], ...]        # action joint order (reordered from the state order)
    joint_pos_state: tuple[tuple[float, ...], ...]  # state joint order, as recorded
    object_pos: tuple[tuple[float, ...], ...]       # per step, xyz
    object_initial_pos: tuple[float, ...]
    ee_pos: dict                                    # hand -> rows of xyz, when the layout reads EE from HDF5
    nan_fraction: float                             # share of steps whose joints or object pose are not finite


def read_episodes(store: TrajectoryStore, layout: TraceLayout) -> dict[tuple[int, int], EpisodeArrays]:
    """Every demo of one dataset file, keyed by ``(env_id, episode_in_env)``."""
    k = layout.hdf5
    state_index = {name: i for i, name in enumerate(layout.state_joints)}
    out: dict[tuple[int, int], EpisodeArrays] = {}
    for demo in store.demos():
        where = f"{store.path}:{demo}"

        def need(key: str) -> Any:
            v = store.read(demo, layout.hdf5_path(key))
            if v is None:
                raise TrajectoryError(f"{where}: no {layout.hdf5_path(key)!r}")
            return v

        ident = (_scalar(need(k.env_id)), _scalar(need(k.episode_in_env)))
        if ident in out:
            raise TrajectoryError(f"{where}: episode {ident} also in {out[ident].source}")
        cmd, pos_state, obj = _rows(need(k.joint_cmd)), _rows(need(k.joint_pos)), _rows(need(k.object_pose))
        init = _rows(need(k.object_initial_pose))
        if len(cmd) != len(pos_state) or len(obj) != len(pos_state):
            raise TrajectoryError(f"{where}: {len(cmd)} commanded, {len(pos_state)} tracked and {len(obj)} object "
                                  f"rows; expected one per step each")
        if any(len(r) != len(layout.action_joints) for r in cmd):
            raise TrajectoryError(f"{where}: joint_cmd rows are not {len(layout.action_joints)} wide")
        if any(len(r) != len(layout.state_joints) for r in pos_state):
            raise TrajectoryError(f"{where}: joint_pos rows are not {len(layout.state_joints)} wide")
        cols = [state_index[a] for a in layout.action_joints]
        ee = {}
        if layout.ee_source == "hdf5":
            ee = {hand: tuple(tuple(r[:3]) for r in _rows(need(key))) for hand, key in layout.ee_keys.items()}
        bad = sum(not (_finite(c) and _finite(p) and _finite(o)) for c, p, o in zip(cmd, pos_state, obj))
        out[ident] = EpisodeArrays(
            env_id=ident[0], episode_in_env=ident[1], source=where,
            joint_cmd=tuple(tuple(r) for r in cmd), joint_pos=tuple(tuple(r[i] for i in cols) for r in pos_state),
            joint_pos_state=tuple(tuple(r) for r in pos_state), object_pos=tuple(tuple(r[:3]) for r in obj),
            object_initial_pos=tuple(init[0][:3]) if init else (), ee_pos=ee,
            nan_fraction=bad / len(pos_state) if pos_state else 0.0)
    return out


def dataset_files(run_dir: Path, layout: TraceLayout) -> list[Path]:
    return sorted(Path(run_dir).glob(layout.hdf5.file_glob))
