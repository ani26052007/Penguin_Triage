"""FIXTURE trace writers for the Phase 3 tests.

``TraceWriter`` is a ``FakeArenaRunner.on_run`` hook. For every episode record of a fake run it writes
what a real traced run would leave behind:

* the policy wrapper's instrument files, produced by the **real** ``policy_wrapper.instrument.Instrument``
  and ``ActionChunkScheduler`` (from the policy-wrapper checkout) driven by a scripted chunk schedule,
  optionally through a ``delay``-step action buffer, with synthetic 32x32 camera frames and the
  end-effector positions as observation keys;
* a synthetic HDF5 dataset in the task config's ``traces.hdf5`` layout (joints, object pose, episode
  identity);
* the record's ``progress.events`` and ``episode_length``.

Episode scripts (local step t, ``LENGTH`` steps): the grasping hand (``hand``, right by default) moves
from 0.3 m in front of the object to the object by step 20, closes between steps 30 and 40 (closure 0.5
at step 35), and from step 35 the object rises with the hand; the other hand stays 0.4 m to its side,
open. Kinds: ``success``; ``door`` (lift and place fire, the door never closes); ``grasp`` (the hand
closes but the object stays); ``reach`` (the hand never comes close).
"""

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

PW = Path(os.environ.get("POLICY_WRAPPER_ROOT", Path(__file__).resolve().parents[2] / "policy-wrapper"))

LENGTH = 120
CHUNK = 16
OBJECT0 = np.array([0.5, 0.0, 1.0])
START = OBJECT0 + np.array([0.3, 0.0, 0.0])
CLOSE_AT, LIFT_RATE = 35, 0.002
EVENTS = {"settled": ("pick_and_place", 0, 2), "lift": ("pick_and_place", 1, 60),
          "place": ("pick_and_place", 2, 90), "door": ("close_door", 0, 110)}
FIRED = {"success": ("settled", "lift", "place", "door"), "door": ("settled", "lift", "place"),
         "grasp": ("settled",), "reach": ("settled",)}


def policy_wrapper():
    """``policy_wrapper.instrument`` and the scheduler module from the checkout."""
    if str(PW) not in sys.path:
        sys.path.insert(0, str(PW))
    from policy_wrapper import instrument
    from policy_wrapper.scheduling import ActionChunkScheduler

    return instrument, ActionChunkScheduler


def script(kind: str, length: int = LENGTH, hand: str = "right") -> dict:
    """Per-step right / left EE position, object position and the grasping hand's closure for one episode."""
    t = np.arange(length, dtype=float)
    approach = np.clip(t / 20.0, 0.0, 1.0)[:, None]
    active = START + (OBJECT0 - START) * approach if kind != "reach" else np.repeat(START[None], length, 0)
    closure = np.clip((t - 30.0) / 10.0, 0.0, 1.0)
    rise = np.where(t > CLOSE_AT, (t - CLOSE_AT) * LIFT_RATE, 0.0)[:, None] * np.array([0.0, 0.0, 1.0])
    active = active + rise if kind != "reach" else active
    obj = OBJECT0 + (rise if kind in ("success", "door") else np.zeros_like(rise))
    other = "left" if hand == "right" else "right"
    side = np.array([0.0, 0.4 if other == "left" else -0.4, 0.0])
    return {hand: active, other: np.repeat((OBJECT0 + side)[None], length, 0), "object": obj,
            "closure": closure, "hand": hand}


def progress(kind: str) -> dict:
    return {"overall_score": 1.0 if kind == "success" else 0.5, "all_complete": kind == "success",
            "criteria_by_name": {},
            "events": [{"step": EVENTS[e][2], "criteria_name": EVENTS[e][0], "sequence_name": "main",
                        "predicate_index": EVENTS[e][1], "predicate_name": e, "score_delta": 0.25}
                       for e in FIRED[kind]]}


class DelayBuffer:
    """An action buffer between the scheduler and ``env.step``: returns each tagged action ``delay``
    steps later; holds (tag -1) while it fills, and empties on reset (as a buffer inside the policy)."""

    def __init__(self, delay: int, num_envs: int):
        self.delay, self.num_envs, self.q = delay, num_envs, deque()

    def __call__(self, tagged, TaggedAction, torch):
        if self.delay == 0:
            return tagged
        self.q.append(tagged)
        if len(self.q) <= self.delay:
            return TaggedAction(torch.zeros_like(tagged.action), torch.full((self.num_envs,), -1),
                                torch.zeros(self.num_envs, dtype=torch.int64))
        return self.q.popleft()

    def reset(self):
        self.q.clear()


def _frame(step: int, env: int) -> np.ndarray:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0] = 180
    img[..., 1] = np.linspace(0, 250, 32, dtype=np.uint8)[:, None]
    img[..., 2] = (step * 7 + env * 13) % 200
    return img


def run_instrument(out_dir: Path, num_envs: int, scripts_by_env: dict[int, list[dict]], *, delay: int = 0,
                   frame_every: int = 10, obs_keys=("policy.left_eef_pos", "policy.right_eef_pos")) -> None:
    """Drives the real Instrument and ActionChunkScheduler for ``num_envs`` envs in lockstep. Each env
    runs its scripted episodes back to back (``on_reset`` after each); envs that finish early keep
    stepping into a trailing episode that ``close`` writes, as in Arena."""
    import torch

    instrument, ActionChunkScheduler = policy_wrapper()
    n = num_envs
    idle = script("reach")
    scripts_by_env = {e: scripts_by_env.get(e) or [] for e in range(n)}
    inst = instrument.Instrument(n, out_dir, frame_every=frame_every, obs_keys=obs_keys)
    sched = ActionChunkScheduler(num_envs=n, action_chunk_length=CHUNK, action_horizon=CHUNK, action_dim=2,
                                 device="cpu")
    buffers = [DelayBuffer(delay, n)]
    pos = {e: [0, 0] for e in range(n)}                     # (episode index, local step)
    total = {e: len(s) for e, s in scripts_by_env.items()}
    while any(pos[e][0] < total[e] for e in range(n)):
        def current(e):
            k, t = pos[e]
            s = scripts_by_env[e][min(k, total[e] - 1)] if total[e] else idle
            return s, min(t, len(s["right"]) - 1)

        frames = torch.from_numpy(np.stack([_frame(pos[e][1], e) for e in range(n)]))
        obs = {"camera_obs": {"cam": frames},
               "policy": {"right_eef_pos": torch.tensor(np.stack([current(e)[0]["right"][current(e)[1]] for e in range(n)])),
                          "left_eef_pos": torch.tensor(np.stack([current(e)[0]["left"][current(e)[1]] for e in range(n)]))}}

        def fetch():
            inst.on_request(obs, {"video": {"ego_view": frames.numpy()[:, None]}}, ["cam"])
            raw = {"left_arm": np.stack([np.arange(CHUNK * 2, dtype=np.float32).reshape(CHUNK, 2) + 100 * e
                                         for e in range(n)])}
            inst.on_reply(raw)
            return torch.from_numpy(raw["left_arm"].copy())

        tagged = sched.get_tagged_action(fetch, obs_step=inst.steps)
        inst.on_step(obs, buffers[0](tagged, instrument.TaggedAction, torch))
        done = []
        for e in range(n):
            pos[e][1] += 1
            k = pos[e][0]
            if k < total[e] and pos[e][1] >= len(scripts_by_env[e][k]["right"]):
                pos[e] = [k + 1, 0]
                done.append(e)
        if done:
            inst.on_reset(torch.tensor(done))
            sched.reset(done)
            if n == 1:
                buffers[0].reset()
    inst.close()


def write_hdf5(path: Path, layout, episodes: list[tuple[int, int, dict]]) -> None:
    """One synthetic dataset file in ``layout.hdf5``'s key paths: one demo per (env_id, episode_in_env, script)."""
    import h5py

    k = layout.hdf5
    state_ix = {j: i for i, j in enumerate(layout.state_joints)}
    with h5py.File(path, "w") as f:
        for i, (env_id, ep, s) in enumerate(episodes):
            g = f.create_group(f"{k.demo_group}/demo_{i}")
            T = len(s["right"])
            g[k.env_id] = np.array([env_id], dtype=np.int64)
            g[k.episode_in_env] = np.array([ep], dtype=np.int64)
            q = np.zeros((T, len(layout.state_joints)))
            q[:, state_ix["left_shoulder_pitch_joint"]] = np.linspace(0, 0.5, T)
            for hand, joints in layout.hands.items():
                closed = s["closure"] if hand == s["hand"] else np.zeros(T)
                for j, (lo, hi) in joints.items():
                    q[:, state_ix[j]] = lo + (hi - lo) * closed
            g[k.joint_pos] = q
            cmd = np.roll(q, -1, axis=0)                      # the target is one step ahead of the tracked joint
            cmd[-1] = q[-1]
            g[k.joint_cmd] = cmd[:, [state_ix[a] for a in layout.action_joints]]
            pose = np.concatenate([s["object"], np.tile([1.0, 0.0, 0.0, 0.0], (T, 1))], axis=1)
            g[layout.hdf5_path(k.object_pose)] = pose
            g[layout.hdf5_path(k.object_initial_pose)] = pose[:1]


@dataclass
class TraceWriter:
    """``FakeArenaRunner.on_run``: traces for every record of the run (see the module docstring)."""
    layout: object
    failure_kind: Callable[[int], str] = lambda k: "grasp"
    hand: Callable[[int], str] = lambda k: "right"     # the grasping hand of episode k
    delay: int = 0
    length: int = LENGTH
    runs: list = field(default_factory=list)

    def __call__(self, inv, run_dir: Path, records: list[dict]) -> None:
        inst_dir = inv.env.get("VLA_INSTRUMENT_DIR")
        by_env: dict[int, list] = {}
        demos = []
        for k, rec in enumerate(records):
            kind = "success" if rec["success"] else self.failure_kind(k)
            rec["episode_length"] = self.length
            rec["progress"] = progress(kind)
            s = script(kind, self.length, self.hand(k))
            by_env.setdefault(rec["env_id"], []).append((rec["episode_in_env"], s))
            demos.append((rec["env_id"], rec["episode_in_env"], s))
        self.runs.append(run_dir)
        if inst_dir is None:
            return
        scripts = {e: [s for _, s in sorted(v, key=lambda x: x[0])] for e, v in sorted(by_env.items())}
        run_instrument(Path(inst_dir), inv.spec.num_envs, scripts, delay=self.delay,
                       frame_every=int(inv.env.get("VLA_INSTRUMENT_FRAME_EVERY", 10)),
                       obs_keys=tuple(k for k in inv.env.get("VLA_INSTRUMENT_OBS_KEYS", "").split(",") if k))
        write_hdf5(run_dir / f"dataset_{inv.spec.run_name}_rebuild0.hdf5", self.layout, demos)
