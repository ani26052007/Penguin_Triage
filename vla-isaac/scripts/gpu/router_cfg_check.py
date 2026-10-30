#!/usr/bin/env python3
"""Router config-contract check on real (injected) policy-wrapper repos.

From the 2026-10-29 session; ran only there, on the box. Generalised from the session's cfg_check.py: the
repos, task config and checkpoint are arguments. It was assembled outside IsaacBackend: the router loop did
not drive it. Only ConfigContractCheck is used, with router code unchanged, on the CPU path.

The TrainingSpec is built here, not extracted: the norm statistics are read from the real checkpoint and the
other values are hard-coded as in vla-isaac/tests/test_injected_bugs.py (PILOT_SPEC). The mapping from
training keys to eval-config keys comes from the task config.

usage: router_cfg_check.py --checkpoint DIR [--task-config YAML] [--checkpoint-id ID] [--embodiment gr1]
                           NAME=REPO_DIR [NAME=REPO_DIR ...]

Prints "NAME: <n> evidence item(s)" and one line per item (id, strength, location). Exit code 0.
Needs vla-router and vla-isaac importable (the vla-isaac venv).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK = ROOT / "config" / "pilot_gr1_fridge.yaml"
DEFAULT_CKPT_ID = "nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task/ranch_bottle_into_fridge"
INSTRUCTION = "Place the sauce bottle on the top shelf of the fridge, and close the fridge door."


def training_spec(stats):
    from router.schemas import TrainingSpec

    return TrainingSpec("gr00t_n1d6_gr1_ranch_bottle_into_fridge", image_size=(512, 512), norm_stats=stats,
                        action_horizon=16, replan_steps=16, denoising_steps=4, control_hz=50, render_settings={},
                        instruction_templates={"default": INSTRUCTION})


def check_repo(repo: Path, task, checkpoint: Path, checkpoint_id: str, embodiment: str = "gr1") -> list:
    """Config-contract evidence for one repo against the checkpoint's training spec."""
    from ladder.workspace import WorkspaceManager
    from router.schemas import ActorKind, ActorRef, Incident, Mode, SceneSpec
    from router.stage1.cfg_contract import ConfigContractCheck
    from vla_isaac.specs.norm_stats import load_flat_statistics
    from vla_isaac.workspace import config_mapping, eval_config_loader

    stats = load_flat_statistics(checkpoint / "statistics.json", embodiment, task.eval_config.norm_stats_groups)
    with tempfile.TemporaryDirectory() as td:
        ws = WorkspaceManager(Path(td) / "ws", {"pw": str(repo)},
                              config_loader=eval_config_loader(task, {checkpoint_id: checkpoint}))
        inc = Incident("pw", SceneSpec(task.nominal_scene), training_spec(stats), ws.base_ref("pw"),
                       ActorRef(ActorKind.POLICY, "ckpt"), (0,), Mode.CLEAN)
        return ConfigContractCheck(config_mapping(task)).run(inc)


def named_repo(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=REPO_DIR, got {text!r}")
    return name, Path(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ConfigContractCheck on policy-wrapper repos.")
    p.add_argument("repos", type=named_repo, nargs="+", metavar="NAME=REPO_DIR")
    p.add_argument("--checkpoint", type=Path, required=True, help="checkpoint directory with statistics.json")
    p.add_argument("--checkpoint-id", default=DEFAULT_CKPT_ID)
    p.add_argument("--task-config", type=Path, default=DEFAULT_TASK)
    p.add_argument("--embodiment", default="gr1")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from vla_isaac.config import load_task_config

    task = load_task_config(args.task_config)
    for name, repo in args.repos:
        ev = check_repo(repo, task, args.checkpoint, args.checkpoint_id, args.embodiment)
        print(f"{name}: {len(ev)} evidence item(s)")
        for e in ev:
            print("  ", e.id, f"{e.strength:.2f}", e.location, "|", e.detail[:110])
    return 0


if __name__ == "__main__":
    sys.exit(main())
