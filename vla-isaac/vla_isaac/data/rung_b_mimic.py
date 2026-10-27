"""Rung B (decision 4): Mimic regeneration under the failing factor, then Arena's LeRobot converter.

``regenerate`` runs the **existing RAW reference runner** (``vla_isaac.reference.mimic.MimicReference``) on
the shifted scene for ``rung_b.episodes`` trials (50-100, config), with the scene's variation overrides
and nothing from the code's eval config. It keeps the reference's seed mode
(``reference.mimic.seed``, VERIFY(V-mimic-seed)):

* ``fixed``: the regeneration is **deterministic**. Trials 0..N-1 are the one fixed trajectory sequence
  of the datagen seed, so regenerating again repeats them. The report says so (``deterministic: true``)
  and warns that the episodes' diversity comes only from the scene variations.
* ``per_run``: the run has its own datagen seed (``run_seed`` in the report).

Mimic keeps only successful demos (``generation_guarantee``, VERIFY(V-mimic-guarantee)), so the generated
HDF5 holds ``successes_total`` episodes; zero successes is refused.

``plan_conversion`` writes a per-run copy of the converter YAML (``config/arena/...``, Arena's file) with
``data_root`` / ``hdf5_name`` pointing at the generated HDF5 and Arena-relative paths made absolute, and
refuses a template whose ``action_name_sim`` is not the configured one (``processed_actions``: the
actions Mimic records). The command is Arena's
``python isaaclab_arena_gr00t/lerobot/convert_hdf5_to_lerobot.py --yaml_file <yaml>`` from the Arena root.
The output directory is ``data_root / <hdf5 stem> / lerobot`` (``Gr00tDatasetConfig.__post_init__``); an
existing one is refused, because the converter would ask on stdin whether to delete it.
VERIFY(V-converter-noisaac): the converter's imports (torch, cv2, h5py, imageio, pandas, tqdm, ffprobe)
include nothing from Isaac, read from the code; whether it runs outside Arena's env is unverified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from router.schemas import ActionPipeline, ActorKind, ActorRef, Factor, SceneSpec

from vla_isaac.config import BackendConfig
from vla_isaac.data.config import Condition, RungBConfig, condition_key
from vla_isaac.data.mixer import LeRobotSource
from vla_isaac.runner.subprocess_runner import ArenaRunError, ArenaRunner, RunInvocation

PATH_KEYS = ("modality_template_path", "info_template_path", "policy_joints_config_path",
             "action_joints_config_path", "state_joints_config_path")
DETERMINISTIC_WARNING = ("deterministic regeneration (fixed Mimic datagen seed): the {n} episodes are one fixed "
                         "trajectory sequence, and regenerating repeats it. Their diversity comes only from the scene "
                         "variations, not from the datagen seed.")


class RungBError(ValueError):
    """Rung B cannot run or convert as asked."""


def scene_for(nominal: str, condition: Condition) -> SceneSpec:
    return SceneSpec(nominal, tuple(Factor(k, v) for k, v in condition.items()))


@dataclass(frozen=True)
class Regeneration:
    condition: Condition
    scene: str
    episodes: int                         # trials asked for
    successes: int                        # demos in the generated HDF5 (VERIFY(V-mimic-guarantee))
    attempts_total: int
    deterministic: bool
    run_seed: Optional[int]
    hdf5: Path
    log: Path
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"condition": dict(self.condition), "condition_key": condition_key(self.condition), "scene": self.scene,
                "episodes": self.episodes, "successes": self.successes, "attempts_total": self.attempts_total,
                "deterministic": self.deterministic,
                "trajectory_sequence": "single fixed sequence" if self.deterministic else "seeded per run",
                "run_seed": self.run_seed, "hdf5": str(self.hdf5), "log": str(self.log),
                "warnings": list(self.warnings)}


def regenerate(reference: Any, factors: Any, nominal: str, condition: Condition, cfg: RungBConfig,
               episodes: Optional[int] = None) -> Regeneration:
    """Mimic trials 0..N-1 on the shifted scene through ``reference`` (a ``MimicReference``)."""
    n = cfg.episodes if episodes is None else int(episodes)
    if not cfg.min_episodes <= n <= cfg.max_episodes:
        raise RungBError(f"rung B regenerates {cfg.min_episodes}-{cfg.max_episodes} episodes, asked for {n}")
    if not condition:
        raise RungBError("rung B regenerates under the failing factor; the nominal condition is not one")
    scene = scene_for(nominal, condition)
    actor = ActorRef(ActorKind.REFERENCE, reference.section.actor_checkpoint, ActionPipeline.RAW)
    results = reference.run(scene, actor, list(range(n)), variations=factors.assignments(scene))
    flags = {bool(r.logs.get("deterministic", False)) for r in results}
    if len(flags) != 1:
        raise RungBError("Mimic results mix deterministic and per-run-seeded trials")
    deterministic = flags.pop()
    paths = results[0].logs["paths"]
    meta = json.loads(Path(paths["vla_meta"]).read_text(encoding="utf-8"))
    successes = int(meta["successes_total"])
    if successes < 1:
        raise RungBError(f"Mimic generated no successful demo under {condition_key(condition)} "
                         f"({meta['attempts_total']} attempts); nothing to convert")
    warnings = (DETERMINISTIC_WARNING.format(n=n),) if deterministic else ()
    return Regeneration(dict(condition), scene.key(), n, successes, int(meta["attempts_total"]), deterministic,
                        results[0].logs.get("run_seed"), Path(paths["output_file"]), Path(paths["log"]), warnings)


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #

def lerobot_dir(hdf5: Path) -> Path:
    """Where Arena's converter writes: ``data_root / hdf5_name.replace('.hdf5', '') / 'lerobot'``."""
    hdf5 = Path(hdf5)
    return hdf5.parent / hdf5.name.replace(".hdf5", "") / "lerobot"


def converter_config(template: Path, hdf5: Path, arena_root: Path, action_name_sim: str) -> dict:
    data = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RungBError(f"{template} does not hold a mapping")
    if data.get("action_name_sim") != action_name_sim:
        raise RungBError(f"{template}: action_name_sim is {data.get('action_name_sim')!r}, expected {action_name_sim!r} "
                         f"(the actions Mimic records)")
    data["data_root"] = str(Path(hdf5).parent)
    data["hdf5_name"] = Path(hdf5).name
    for key in PATH_KEYS:
        if key in data and not Path(str(data[key])).is_absolute():
            data[key] = str(Path(arena_root) / str(data[key]))
    return data


@dataclass(frozen=True)
class ConvertSpec:
    run_name: str
    hdf5: Path
    lerobot_dir: Path


def plan_conversion(cfg: BackendConfig, rung_b: RungBConfig, hdf5: Path, work_dir: Path) -> RunInvocation:
    """Writes ``work_dir/converter.yaml`` and returns the converter command (pure apart from that file)."""
    hdf5, work_dir = Path(hdf5), Path(work_dir)
    out = lerobot_dir(hdf5)
    if out.exists():
        raise RungBError(f"{out} exists; Arena's converter would ask on stdin whether to delete it. Move it away first")
    data = converter_config(rung_b.converter.template, hdf5, cfg.arena_root, rung_b.converter.action_name_sim)
    work_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = work_dir / "converter.yaml"
    header = (f"# Written by vla_isaac.data.rung_b_mimic from {rung_b.converter.template.name}, a copy of Isaac Lab "
              f"Arena's converter YAML\n# (Apache-2.0, commit 7632eb7d5e); data_root, hdf5_name and paths changed.\n")
    yaml_path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    argv = [cfg.arena_python, str(cfg.arena_root / rung_b.converter.script), "--yaml_file", str(yaml_path)]
    return RunInvocation(argv=tuple(argv), cwd=cfg.arena_root, env={}, timeout_s=cfg.timeouts.run_s,
                         log_path=work_dir / "convert.log", output_dir=out,
                         spec=ConvertSpec(f"convert_{hdf5.stem}", hdf5, out))   # type: ignore[arg-type]


def convert(runner: ArenaRunner, inv: RunInvocation, source_id: str, condition: Condition) -> LeRobotSource:
    runner.run(inv)
    out = inv.output_dir
    missing = [n for n in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json") if not (out / "meta" / n).is_file()]
    if missing:
        raise ArenaRunError(f"converter left no {missing} under {out / 'meta'}; log {inv.log_path}", log_path=inv.log_path)
    return LeRobotSource(source_id, out, dict(condition))


def rung_b_report(regenerations: Sequence[Regeneration]) -> dict:
    det = [r.deterministic for r in regenerations]
    return {"regenerations": [r.to_dict() for r in regenerations], "deterministic": all(det) if det else None,
            "warnings": sorted({w for r in regenerations for w in r.warnings})}
