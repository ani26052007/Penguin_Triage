"""The code under test as vla-router's ``WorkspaceManager`` sees it.

``make_workspaces(cfg, task)`` registers the backend config's ``workspace_bases`` (policy-wrapper
checkouts) with a config loader that reads the copy's eval config (``task.policy.config_path``) and
adds two derived keys (``TaskConfig.eval_config``):

* ``derived_image_hw``: the first two entries of the image size, the (H, W) the training spec has;
* ``derived_norm_stats``: the normalisation statistics the server will use, flattened like
  ``TrainingSpec.norm_stats``: the override file named in the eval config (relative to the copy)
  when set, else the checkpoint's ``statistics.json``. The checkpoint id is looked up in the backend
  config's ``checkpoints`` map, or is a local directory. A checkpoint that cannot be found is a
  ``WorkspaceError``: the config contract must not pass for lack of the statistics.

``config_mapping(task)`` is the router's ``ConfigMapping`` for this task, and ``rung0_config(task)`` its
Rung 0 settings (the task config's forbidden paths, which include the instrumentation).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from ladder.rung0 import Rung0Config
from ladder.workspace import ConfigLoader, WorkspaceError, WorkspaceManager, file_config_loader
from router.stage1.cfg_contract import ConfigMapping

from vla_isaac.config import BackendConfig, EvalConfigSection, TaskConfig, get_dotted
from vla_isaac.specs.norm_stats import NormStatsError, load_flat_statistics

STATISTICS_FILE = "statistics.json"


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    *parents, last = dotted.split(".")
    for part in parents:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[last] = value


def checkpoint_dir(checkpoint: str, checkpoints: Mapping[str, Path], workdir: Path) -> Path:
    if checkpoint in checkpoints:
        return Path(checkpoints[checkpoint])
    local = Path(checkpoint) if Path(checkpoint).is_absolute() else workdir / checkpoint
    if local.is_dir():
        return local
    raise WorkspaceError(f"checkpoint {checkpoint!r} is not a local directory and not in the backend config's "
                         f"checkpoints map ({sorted(checkpoints)})")


def derive_keys(cfg: Mapping[str, Any], ec: EvalConfigSection, workdir: Path,
                checkpoints: Mapping[str, Path]) -> dict[str, Any]:
    out = copy.deepcopy(dict(cfg))
    found, size = get_dotted(out, ec.image_size_key)
    if found and isinstance(size, (list, tuple)) and len(size) >= 2:
        set_dotted(out, ec.derived_image_hw, list(size[:2]))
    found, embodiment = get_dotted(out, ec.embodiment_key)
    if not found or not isinstance(embodiment, str):
        raise WorkspaceError(f"eval config has no embodiment tag at {ec.embodiment_key!r}")
    _, override = get_dotted(out, ec.norm_stats_override_key)
    if override is not None:
        stats = Path(override) if Path(str(override)).is_absolute() else workdir / str(override)
    else:
        found, checkpoint = get_dotted(out, ec.checkpoint_key)
        if not found or not checkpoint:
            raise WorkspaceError(f"eval config has no checkpoint at {ec.checkpoint_key!r}")
        stats = checkpoint_dir(str(checkpoint), checkpoints, workdir) / STATISTICS_FILE
    if not stats.is_file():
        raise WorkspaceError(f"no normalisation statistics at {stats}")
    try:
        flat = load_flat_statistics(stats, embodiment.lower(), ec.norm_stats_groups)
    except (NormStatsError, ValueError) as e:
        raise WorkspaceError(f"cannot read the normalisation statistics {stats}: {e}") from e
    set_dotted(out, ec.derived_norm_stats, flat)
    return out


def eval_config_loader(task: TaskConfig, checkpoints: Mapping[str, Path]) -> ConfigLoader:
    """vla-router ``ConfigLoader``: the eval config file plus the derived keys."""
    read = file_config_loader(task.policy.config_path)

    def load(workdir: Path) -> Mapping[str, Any]:
        return derive_keys(read(workdir), task.eval_config, Path(workdir), checkpoints)
    return load


def make_workspaces(cfg: BackendConfig, task: TaskConfig) -> WorkspaceManager:
    return WorkspaceManager(cfg.workspace_store, cfg.workspace_bases,
                            config_loader=eval_config_loader(task, cfg.checkpoints))


def config_mapping(task: TaskConfig, **kw) -> ConfigMapping:
    return ConfigMapping(keys=dict(task.config_mapping), **kw)


def rung0_config(task: TaskConfig, **kw) -> Rung0Config:
    """Rung 0 for patches to this task's code: the task config's ``rung0.forbidden_prefixes`` plus ``kw``."""
    if task.rung0 is None:
        raise WorkspaceError(f"task config {task.source} has no rung0 section")
    if "forbidden_prefixes" in kw:
        raise ValueError("forbidden_prefixes come from the task config")
    return Rung0Config(forbidden_prefixes=task.rung0.forbidden_prefixes, **kw)
