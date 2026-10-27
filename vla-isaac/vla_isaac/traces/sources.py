"""``IsaacTraceSource`` / ``IsaacBehaviorSource``: the router's trace sources over ``IsaacBackend`` runs.

An incident's log reference for a mode (``incident.log_refs[mode.value]``) is the run root that
``IsaacBackend`` wrote for that mode (``EpisodeResult.logs["paths"]["run_root"]``): one directory per
build under it, each with its ``vla_meta.json``. The sources read only what a ``vla_meta.json`` names,
and only inside its own directory:

* ``results``: the episode JSONL (success, ``progress.events``, episode identity);
* ``instrument_dir``: the policy wrapper's per-episode files (frames, step pairs, raw actions,
  observation keys such as the end-effector positions);
* ``run_dir``: Arena's HDF5 trajectory dataset (joints, object pose).

Every episode of the JSONL becomes one trace, with ``seed = build_seed * seed_stride + k`` as the
backend numbers them. ``code_key`` is the one in ``vla_meta.json``, never the incident's: the router's
template ``load`` then refuses traces of other code, and traces without a key. A run that did not
record traces, a missing or unreadable file, or an episode missing from one of the three sources is
``TraceUnavailable``: a missing log never reads as a clean one. A reference to runs of the other mode
is a ``ValueError`` (a wiring error, as in the router's in-memory sources).

Not logged: contact penetration (``SimHealth.max_penetration`` stays 0.0) and the missing-texture
fraction of frames.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from router.prereq.behavior_trace import BehaviorSource, BehaviorTrace
from router.prereq.trace import EpisodeTrace, SimHealth, TraceSource, TraceUnavailable
from router.schemas import Incident, Mode

from vla_isaac.results.episode_jsonl import EpisodeRecord, EpisodeResultsError, index_records, read_records
from vla_isaac.seeds import join_seed
from vla_isaac.traces.events import fired, grasp_event, reach_step, subtask_progress
from vla_isaac.traces.hand_closure import closure, opening
from vla_isaac.traces.hdf5_reader import (EpisodeArrays, H5pyStore, TrajectoryError, TrajectoryStore, dataset_files,
                                          read_episodes)
from vla_isaac.traces.instrument_reader import InstrumentEpisode, InstrumentError, read_instrument_dir
from vla_isaac.traces.layout import TraceLayout

META_FILE = "vla_meta.json"
StoreFactory = Callable[[Path, str], TrajectoryStore]


@dataclass(frozen=True)
class LoadedEpisode:
    seed: int
    success: bool
    code_key: Optional[str]
    record: EpisodeRecord
    instrument: InstrumentEpisode
    arrays: EpisodeArrays


def _inside(path: Any, root: Path, what: str, meta_path: Path) -> Path:
    p = Path(str(path)).resolve()
    if not p.is_relative_to(root.resolve()):
        raise TraceUnavailable(f"{meta_path}: {what} {p} is outside the run directory {root}")
    return p


class RunLoader:
    """The episodes of the runs under one log reference, joined across the three sources."""

    def __init__(self, layout: TraceLayout, store_factory: Optional[StoreFactory] = None) -> None:
        self.layout = layout
        self.store_factory = store_factory or H5pyStore

    def load(self, incident: Incident, mode: Mode) -> list[LoadedEpisode]:
        mode = Mode(mode)
        ref = incident.log_refs.get(mode.value)
        if ref is None:
            raise TraceUnavailable(f"incident {incident.incident_id} has no log_refs[{mode.value!r}]")
        root = Path(ref)
        metas = [root / META_FILE] if (root / META_FILE).is_file() else sorted(root.glob(f"*/{META_FILE}"))
        if not metas:
            raise TraceUnavailable(f"no {META_FILE} under {root} ({mode.value} mode)")
        out: list[LoadedEpisode] = []
        for m in metas:
            try:
                out.extend(self._build(m, mode))
            except (InstrumentError, TrajectoryError, EpisodeResultsError, OSError, KeyError) as e:
                raise TraceUnavailable(f"{m.parent}: logs cannot be read: {type(e).__name__}: {e}") from e
        return out

    def _build(self, meta_path: Path, mode: Mode) -> list[LoadedEpisode]:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("mode") != mode.value:
            raise ValueError(f"{meta_path} is a {meta.get('mode')!r}-mode run, not {mode.value!r}")
        if not meta.get("record_trajectories") or not meta.get("instrument_dir"):
            raise TraceUnavailable(f"{meta_path}: the run did not record traces")
        here = meta_path.parent
        results = [_inside(p, here, "results", meta_path) for p in meta["results"]]
        inst_dir = _inside(meta["instrument_dir"], here, "instrument_dir", meta_path)
        run_dir = _inside(meta["run_dir"], here, "run_dir", meta_path)
        records = [r for f in results for r in read_records(f)]
        by_k = index_records(records, num_envs=int(meta["num_envs"]), num_episodes=int(meta["num_episodes"]),
                             build_seed=int(meta["build_seed"]))
        instrument = read_instrument_dir(inst_dir)
        files = [_inside(p, here, "dataset", meta_path) for p in dataset_files(run_dir, self.layout)]
        if not files:
            raise TraceUnavailable(f"{meta_path}: no trajectory dataset {self.layout.hdf5.file_glob!r} in {run_dir}")
        arrays: dict[tuple[int, int], EpisodeArrays] = {}
        for f in files:
            store = self.store_factory(f, self.layout.hdf5.demo_group)
            try:
                for key, arr in read_episodes(store, self.layout).items():
                    if key in arrays:
                        raise TrajectoryError(f"episode {key} is in {arrays[key].source} and {arr.source}")
                    arrays[key] = arr
            finally:
                getattr(store, "close", lambda: None)()
        out = []
        for k, rec in by_k.items():
            key = (rec.env_id, rec.episode_in_env)
            if key not in instrument:
                raise TraceUnavailable(f"{inst_dir}: no instrument file for episode {key}")
            if key not in arrays:
                raise TraceUnavailable(f"{run_dir}: no trajectory for episode {key}")
            inst = instrument[key]
            length = rec.raw.get("episode_length")
            if self.layout.check_episode_length and length is not None and inst.num_steps != int(length):
                raise TraceUnavailable(f"episode {key}: the instrument logged {inst.num_steps} steps, the episode "
                                       f"record says {length} (VERIFY(V-reset-calls))")
            out.append(LoadedEpisode(join_seed(int(meta["build_seed"]), k, int(meta["seed_stride"])), rec.success,
                                     meta.get("code_key"), rec, inst, arrays[key]))
        return out


def _ee(layout: TraceLayout, ep: LoadedEpisode) -> dict[str, tuple]:
    if layout.ee_source == "hdf5":
        return ep.arrays.ee_pos
    return {hand: tuple(r[:3] for r in ep.instrument.observations[key]) for hand, key in layout.ee_keys.items()}


def episode_trace(layout: TraceLayout, ep: LoadedEpisode, mode: Mode) -> EpisodeTrace:
    a, hand = ep.arrays, layout.hands[layout.width_hand]
    rest: tuple = ()
    if layout.object_rest_event is not None:
        step = fired(ep.record.raw.get("progress", {}).get("events", []), layout.object_rest_event)
        if step is not None and step < len(a.object_pos):
            rest = a.object_pos[step]
    return EpisodeTrace(
        seed=ep.seed, mode=mode, frames=ep.instrument.frames, step_pairs=ep.instrument.step_pairs,
        raw_actions=ep.instrument.raw_actions, joint_cmd=a.joint_cmd, joint_pos=a.joint_pos,
        gripper_cmd_width=opening(a.joint_cmd, layout.action_joints, hand),
        gripper_width=opening(a.joint_pos_state, layout.state_joints, hand),
        sim=SimHealth(nan_fraction=a.nan_fraction, object_initial_pos=a.object_initial_pos, object_rest_pos=rest),
        code_key=ep.code_key)


def behavior_trace(layout: TraceLayout, ep: LoadedEpisode, mode: Mode) -> BehaviorTrace:
    a = ep.arrays
    ee = _ee(layout, ep)
    closures = {h: closure(a.joint_pos_state, layout.state_joints, joints) for h, joints in layout.hands.items()}
    grasp = grasp_event(closures, ee, a.object_pos, layout.grasp)
    completed: dict[str, bool] = {"reach": reach_step(ee, a.object_pos, layout.reach_distance_m) is not None,
                                  "grasp": grasp is not None}
    progress: Mapping[str, Any] | None = ep.record.raw.get("progress")
    if progress is not None:                    # without progress tracking, only the offline subtasks are known
        events = progress.get("events", [])
        completed.update({name: fired(events, m) is not None for name, m in layout.subtasks.items()})
    return BehaviorTrace(ep.seed, mode, ep.success, grasp.record if grasp else None,
                         subtask_progress(completed, layout.phase_map), ep.code_key)


class IsaacTraceSource(TraceSource):
    def __init__(self, layout: TraceLayout, store_factory: Optional[StoreFactory] = None) -> None:
        self.loader = RunLoader(layout, store_factory)

    def _load(self, incident: Incident, mode: Mode) -> list[EpisodeTrace]:
        return [episode_trace(self.loader.layout, ep, Mode(mode)) for ep in self.loader.load(incident, mode)]


class IsaacBehaviorSource(BehaviorSource):
    def __init__(self, layout: TraceLayout, store_factory: Optional[StoreFactory] = None) -> None:
        self.loader = RunLoader(layout, store_factory)

    def _load(self, incident: Incident, mode: Mode) -> list[BehaviorTrace]:
        return [behavior_trace(self.loader.layout, ep, Mode(mode)) for ep in self.loader.load(incident, mode)]
