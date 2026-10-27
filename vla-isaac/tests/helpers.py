"""Test helpers: a small base checkout shaped like policy-wrapper, a backend config on temp dirs, a
fake policy server, and ``make_backend``. Everything here is a FIXTURE; nothing touches Isaac,
GR00T or a GPU."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ladder.workspace import WorkspaceManager
from router.prereq.rollout_cache import RolloutCache
from router.schemas import ActionPipeline, ActorKind, ActorRef, Factor, SceneSpec

from vla_isaac.backend.isaac_backend import IsaacBackend
from vla_isaac.config import BackendConfig, ServerConfig, load_task_config
from vla_isaac.runner.subprocess_runner import ArenaRunner, FakeArenaRunner

REPO = Path(__file__).resolve().parents[1]
PILOT_TASK = REPO / "config" / "pilot_gr1_fridge.yaml"
NOMINAL = SceneSpec("gr1_fridge_ranch")
DARK = SceneSpec("gr1_fridge_ranch", (Factor("lighting.intensity", "100.0"),))
DARK_HDR = SceneSpec("gr1_fridge_ranch", (Factor("lighting.intensity", "100.0"), Factor("lighting.hdr", "home_office_robolab")))
POLICY = ActorRef(ActorKind.POLICY, "ranch_bottle_into_fridge")
REF_RAW = ActorRef(ActorKind.REFERENCE, "mimic-gr1_pink", ActionPipeline.RAW)
REF_WRAP = ActorRef(ActorKind.REFERENCE, "mimic-gr1_pink", ActionPipeline.WRAPPER)

EVAL = {
    "client": {"language_instruction": "Place the sauce bottle on the top shelf of the fridge, and close the fridge door."},
    "server": {"denoising_steps": 4},
    "env": {"task": "put_item_in_fridge_and_close_door", "embodiment": "gr1_joint",
            "object": "ranch_dressing_hope_robolab", "control_hz": 50, "actuator_overrides": {}},
}


def git(cwd: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                          capture_output=True, text=True, env=env).stdout


def make_checkout(root: Path, eval_config: Mapping[str, Any] = EVAL) -> Path:
    """A git checkout with config/eval.yaml and a stub serve module (FIXTURE)."""
    (root / "config").mkdir(parents=True)
    (root / "policy_wrapper").mkdir()
    (root / "config" / "eval.yaml").write_text(yaml.safe_dump(dict(eval_config), sort_keys=True))
    (root / "policy_wrapper" / "__init__.py").write_text("")
    (root / "policy_wrapper" / "serve.py").write_text("raise SystemExit('stub: not run in tests')\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


@dataclass
class FakeServer:
    """Stands in for ``PolicyServerProcess``: records start / stop / restart, always pings."""
    argv: list
    cwd: Path
    env: Mapping[str, str]
    cfg: ServerConfig
    log_path: Path
    events: list = field(default_factory=list)
    alive: bool = True

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")

    def restart(self) -> None:
        self.events.append("restart")

    def ping(self) -> bool:
        return self.alive


def backend_config(tmp: Path, checkout: Path, **kw) -> BackendConfig:
    (tmp / "arena").mkdir(exist_ok=True)
    (tmp / "gr00t").mkdir(exist_ok=True)
    base = dict(arena_root=tmp / "arena", gr00t_root=tmp / "gr00t", arena_python="arena-python",
                server_python="gr00t-python", output_root=tmp / "out", task_config=PILOT_TASK,
                workspace_store=tmp / "store", workspace_bases={"pw": checkout}, incident_num_envs=4)
    base.update(kw)
    return BackendConfig(**base)


@dataclass
class Rig:
    backend: IsaacBackend
    workspaces: WorkspaceManager
    runner: ArenaRunner
    servers: list
    cache: Optional[RolloutCache]

    def code(self, label: str = "pw"):
        return self.workspaces.base_ref(label)


def make_backend(tmp: Path, *, eval_config: Mapping[str, Any] = EVAL, runner: Optional[ArenaRunner] = None,
                 cache: Optional[RolloutCache] = None, reference=None, **cfg_kw) -> Rig:
    checkout = make_checkout(tmp / "checkout", eval_config)
    cfg = backend_config(tmp, checkout, **cfg_kw)
    ws = WorkspaceManager(cfg.workspace_store, cfg.workspace_bases)
    servers: list = []

    def factory(argv, **kw):
        s = FakeServer(list(argv), **kw)
        servers.append(s)
        return s
    runner = runner if runner is not None else FakeArenaRunner()
    be = IsaacBackend(cfg, load_task_config(cfg.task_config), ws, runner=runner, server_factory=factory,
                      cache=cache, reference=reference, backend_id="isaac:test")
    return Rig(be, ws, runner, servers, cache)


def with_eval(**sections) -> dict:
    """EVAL with whole sections replaced or merged one level deep."""
    out = {k: dict(v) for k, v in EVAL.items()}
    for k, v in sections.items():
        out[k] = {**out.get(k, {}), **v}
    return out


__all__ = ["DARK", "DARK_HDR", "EVAL", "NOMINAL", "PILOT_TASK", "POLICY", "REF_RAW", "REF_WRAP", "Rig", "backend_config",
           "make_backend", "make_checkout", "replace", "with_eval"]
