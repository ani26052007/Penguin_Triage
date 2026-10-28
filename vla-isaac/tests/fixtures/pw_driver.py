"""FIXTURE: drive a policy-wrapper checkout's own client on synthetic observations, with instrumentation on.

    python pw_driver.py --repo R --out DIR [--episodes 3] [--steps 48] [--num-envs 1] [--scene bright|dark]
                        [--seed 0] [--server-delay-ms 0] [--step-ms 0] [--env K=V ...]

Runs in its own process, with the checkout (and its ``tests/fake_gr00t`` for the modality configs) first on
``sys.path``, so the code under test is exactly the checkout's. The GR00T server is an in-process stand-in
for ``PolicyClient``: it answers each request after ``--server-delay-ms`` with a smooth chunk per policy group
(arms: slow sines, hands: a closing ramp). Camera frames are a synthetic scene with colour balance and
vertical/horizontal structure (so channel order and orientation are measurable), scaled by the scene's
brightness. ``DIR`` receives the instrument files and ``actions_env<e>_ep<k>.json`` (the actions returned
to ``env.step``, in the 36-joint action order).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

GROUPS = {"left_arm": 7, "right_arm": 7, "left_hand": 6, "right_hand": 6}
BRIGHTNESS = {"bright": 1.0, "dark": 0.22}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--scene", choices=sorted(BRIGHTNESS), default="bright")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--server-delay-ms", type=float, default=0.0)
    ap.add_argument("--step-ms", type=float, default=0.0)
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    repo, out = Path(a.repo).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    os.environ["VLA_INSTRUMENT_DIR"] = str(out)
    os.environ["VLA_INSTRUMENT_FRAME_EVERY"] = "1"
    sys.path[:0] = [str(repo / "tests" / "fake_gr00t"), str(repo)]

    import numpy as np
    import torch

    n = a.num_envs
    state = {"requests": 0}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def ping(self):
            return True

        def get_action(self, observation, options=None):
            if a.server_delay_ms:
                time.sleep(a.server_delay_ms / 1000.0)
            r = state["requests"]
            state["requests"] += 1
            t = np.arange(16, dtype=np.float32)[None, :, None] + 16 * r
            chunk = {}
            for g, size in GROUPS.items():
                j = np.arange(size, dtype=np.float32)[None, None, :]
                if g.endswith("hand"):
                    chunk[g] = np.broadcast_to(np.clip(t / 40.0, 0, 1.2) + 0.0 * j, (n, 16, size)).astype(np.float32)
                else:
                    chunk[g] = np.broadcast_to(0.3 * np.sin(t / 9.0 + j), (n, 16, size)).astype(np.float32)
            return {k: v.copy() for k, v in chunk.items()}, {}

        def reset(self):
            return {}

    mod = types.ModuleType("gr00t.policy.server_client")
    mod.PolicyClient = Client
    sys.modules["gr00t.policy.server_client"] = mod

    from policy_wrapper.policy.gr00t_remote_closedloop_policy import (
        Gr00tRemoteClosedloopPolicy,
        Gr00tRemoteClosedloopPolicyCfg,
    )

    h = w = 512
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    scene = np.zeros((h, w, 3), dtype=np.float32)
    scene[..., 0] = 170 + 60 * xx                              # red rises left to right
    scene[..., 1] = 60 + 120 * yy                              # green rises top to bottom
    scene[..., 2] = 40 + 30 * (1 - yy)
    scene[int(0.55 * h):int(0.8 * h), int(0.2 * w):int(0.4 * w)] = (230, 210, 40)   # the object, lower left
    rng = np.random.default_rng(a.seed)

    def frames() -> torch.Tensor:
        imgs = [np.clip(scene * BRIGHTNESS[a.scene] + rng.normal(0, 3, scene.shape), 0, 255) for _ in range(n)]
        return torch.from_numpy(np.stack(imgs).astype(np.uint8))

    policy = Gr00tRemoteClosedloopPolicy(Gr00tRemoteClosedloopPolicyCfg(
        policy_config_yaml_path=str(repo / "config" / "eval.yaml"), policy_device="cpu", num_envs=n,
        remote_host="127.0.0.1", remote_port=0))
    policy.set_task_description(None)
    for ep in range(a.episodes):
        acts = [[] for _ in range(n)]
        for t in range(a.steps):
            obs = {"camera_obs": {"robot_pov_cam_rgb": frames()},
                   "policy": {"robot_joint_pos": torch.from_numpy(rng.normal(0, 0.1, (n, 54)).astype(np.float32)),
                              "left_eef_pos": torch.zeros(n, 3), "right_eef_pos": torch.zeros(n, 3)}}
            act = policy.get_action(None, obs)
            for e in range(n):
                acts[e].append([float(x) for x in act[e]])
            if a.step_ms:
                time.sleep(a.step_ms / 1000.0)
        policy.reset()
        for e in range(n):
            (out / f"actions_env{e:04d}_ep{ep:05d}.json").write_text(json.dumps(acts[e]))
    policy.close()


if __name__ == "__main__":
    main()
