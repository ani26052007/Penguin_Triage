"""The GR00T fine-tune command (``G:gr00t/experiment/launch_finetune.py``) and its run manifest.

Presets (``finetune.presets``; PLAN_UPDATE 5.4): ``default`` = ``--no-tune-llm --no-tune-visual
--tune-projector --tune-diffusion-model`` and ``tune_visual`` = Arena's GR1 recipe
(``--no-tune-llm --tune-visual ...``, the vision-side ablation). Both get Rung A's augmentation flags
(``vla_isaac.data.rung_a``). The argument order follows Arena's documented commands
(``A:docs/.../sequential_static_manipulation/step_4_policy_training.rst:164-215``):

* 1 GPU: ``CUDA_VISIBLE_DEVICES=0 <python> gr00t/experiment/launch_finetune.py ...``;
* N GPUs: ``<python> -m torch.distributed.run --nproc_per_node=N --standalone gr00t/experiment/launch_finetune.py ...``;

run from the GR00T root, with the hardware values (batch, steps, workers) of ``finetune.hardware[N]``.

**``tune_top_llm_layers``.** The fine-tune path passes no ``tune_top_llm_layers``, so the base checkpoint's
``config.json`` decides it, and a value > 0 trains those top LLM layers even with ``--no-tune-llm``
(facts §3, ``G:gr00t/model/modules/eagle_backbone.py:75-78``). The launcher reads it before building a
command and **refuses** when it cannot (no ``config.json``, unreadable, key absent or not an integer;
VERIFY(V-ckpt-cfg)), unless ``allow_unknown_top_llm_layers`` is set. The manifest records the value, where it
came from, and whether the unknown value was allowed.

The base checkpoint is a local directory from the backend config's ``checkpoints`` map; it is never
downloaded, and an unmapped one is refused.

**Manifest** (``vla_finetune_manifest.json`` in the output directory, written before the run): both
presets' flags, the chosen one, ``tune_top_llm_layers``, the checkpoint id / revision / local directory,
Rung A, the data mix (the dataset's ``meta/vla_mix.json`` when the mixer wrote it), the held-out split,
the seeds (GR00T's training seed is ``data.seed = 42``, not on the fine-tune CLI,
``G:gr00t/configs/data/data_config.py:70``), and the command. After the run, ``record_run`` adds the
trainable and total parameter counts from the log (``G:gr00t/model/gr00t_n1d6/setup.py:105-110``;
VERIFY(V-finetune-log)) and ``experiment_cfg/final_model_config.json`` (``setup.py:101-103``,
``G:gr00t/experiment/experiment.py:135``).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from vla_isaac.config import BackendConfig
from vla_isaac.data.config import TUNE_FLAGS, PipelineConfig
from vla_isaac.data.mixer import MIX_FILE
from vla_isaac.data.rung_a import rung_a_manifest
from vla_isaac.runner.subprocess_runner import ArenaRunner, RunInvocation, SubprocessRunner

MANIFEST_FILE = "vla_finetune_manifest.json"
FINAL_CONFIG = Path("experiment_cfg") / "final_model_config.json"
UNPINNED = "TODO_PIN"
TRAINING_SEED = {"value": 42, "on_cli": False,
                 "source": "G:gr00t/configs/data/data_config.py:70 (data.seed); launch_finetune.py does not set it"}
TOP_LAYERS_NOTE = ("not passed by the fine-tune path, so the base checkpoint's config.json decides it; > 0 trains the "
                   "top LLM layers even with --no-tune-llm (G:gr00t/model/modules/eagle_backbone.py:75-78)")
# setup.py:105-110 logs these two lines through logging.info; the prefix depends on the logging setup.
TOTAL_RX = re.compile(r"Total parameters: ([\d,]+)")
TRAINABLE_RX = re.compile(r"Trainable parameters: ([\d,]+) \(([\d.]+)%\)")


class LauncherError(ValueError):
    """The fine-tune command cannot be built as configured."""


@dataclass(frozen=True)
class TopLayers:
    value: Optional[int]
    source: str
    reason: Optional[str] = None          # why the value is unknown

    @property
    def readable(self) -> bool:
        return self.value is not None


def read_tune_top_llm_layers(checkpoint_dir: Path) -> TopLayers:
    path = Path(checkpoint_dir) / "config.json"     # VERIFY(V-ckpt-cfg): the key is in the pilot checkpoint
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        return TopLayers(None, str(path), f"cannot read {path}: {e.strerror or e}")
    except ValueError as e:
        return TopLayers(None, str(path), f"{path} is not JSON: {e}")
    if not isinstance(data, dict) or "tune_top_llm_layers" not in data:
        return TopLayers(None, str(path), f"{path} has no tune_top_llm_layers")
    v = data["tune_top_llm_layers"]
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        return TopLayers(None, str(path), f"{path} tune_top_llm_layers is {v!r}, not an integer >= 0")
    return TopLayers(v, str(path))


def preset_args(flags: Mapping[str, bool]) -> list[str]:
    """``--tune-x`` / ``--no-tune-x`` for every tune flag, in ``TUNE_FLAGS`` order."""
    return [f"--{'' if flags[k] else 'no-'}{k.replace('_', '-')}" for k in TUNE_FLAGS]


@dataclass(frozen=True)
class FinetunePlan:
    argv: tuple[str, ...]
    env: Mapping[str, str]
    cwd: Path
    output_dir: Path
    manifest: Mapping[str, Any]

    @property
    def command(self) -> str:
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(self.env.items()))
        return (prefix + " " if prefix else "") + shlex.join(self.argv)

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / MANIFEST_FILE


def _data_mix(dataset: Path) -> dict:
    mix = Path(dataset) / "meta" / MIX_FILE
    if mix.is_file():
        data = json.loads(mix.read_text(encoding="utf-8"))
        return {"dataset_path": str(dataset), "mixed": True, "mix_file": str(mix),
                **{k: data[k] for k in ("ratio", "seed", "nominal_count", "shifted_count", "achieved_ratio", "available",
                                        "nominal", "shifted") if k in data}}
    return {"dataset_path": str(dataset), "mixed": False}


def plan_finetune(cfg: BackendConfig, pipe: PipelineConfig, *, dataset_path: str | Path, output_dir: str | Path,
                  heldout: Optional[Mapping[str, Any]] = None, seeds: Optional[Mapping[str, Any]] = None,
                  preset: Optional[str] = None, num_gpus: Optional[int] = None,
                  allow_unknown_top_llm_layers: Optional[bool] = None) -> FinetunePlan:
    """The command and the manifest for one fine-tune run. Pure apart from reading the checkpoint's
    ``config.json`` / ``processor_config.json`` and the dataset's ``meta/vla_mix.json``."""
    ft = pipe.finetune
    preset = preset or ft.preset
    n = ft.num_gpus if num_gpus is None else int(num_gpus)
    allow = ft.allow_unknown_top_llm_layers if allow_unknown_top_llm_layers is None else bool(allow_unknown_top_llm_layers)
    if preset not in ft.presets:
        raise LauncherError(f"no fine-tune preset {preset!r} (have {sorted(ft.presets)})")
    if n not in ft.hardware:
        raise LauncherError(f"no finetune.hardware entry for {n} GPUs (have {sorted(ft.hardware)})")
    local = cfg.checkpoints.get(ft.base_checkpoint)
    if local is None or not Path(local).is_dir():
        raise LauncherError(f"base checkpoint {ft.base_checkpoint!r} has no local directory in the backend config's "
                            f"checkpoints map (got {local}); it is never downloaded")
    top = read_tune_top_llm_layers(Path(local))
    if not top.readable and not allow:
        raise LauncherError(f"tune_top_llm_layers cannot be read ({top.reason}); without it the 'frozen LLM' recipe "
                            f"may train LLM layers unnoticed. Set finetune.allow_unknown_top_llm_layers to run anyway")
    dataset, out = Path(dataset_path), Path(output_dir)
    hw = ft.hardware[n]
    rung_a = rung_a_manifest(pipe.rung_a, Path(local))
    script = ft.launcher
    args = ["--dataset-path", str(dataset), "--output-dir", str(out),
            "--modality-config-path", str(cfg.arena_root / ft.modality_config_path),
            "--global-batch-size", str(hw["global_batch_size"]), "--max-steps", str(hw["max_steps"]),
            "--num-gpus", str(n), "--save-steps", str(hw["save_steps"]), "--save-total-limit", str(hw["save_total_limit"]),
            "--base-model-path", str(local), *preset_args(ft.presets[preset]),
            "--dataloader-num-workers", str(hw["dataloader_num_workers"]), "--embodiment-tag", ft.embodiment_tag,
            *rung_a["args"]]
    if n == 1:
        argv, env = [cfg.server_python, script, *args], {"CUDA_VISIBLE_DEVICES": "0"}
    else:
        argv, env = [cfg.server_python, "-m", "torch.distributed.run", f"--nproc_per_node={n}", "--standalone",
                     script, *args], {}
    manifest = {
        "format": "vla_isaac.finetune/1",
        "name": pipe.name,
        "preset": preset,
        "presets": {name: {"flags": dict(flags), "args": preset_args(flags)} for name, flags in ft.presets.items()},
        "tune_top_llm_layers": {"value": top.value, "readable": top.readable, "source": top.source,
                                "reason": top.reason, "allow_unknown_top_llm_layers": allow,
                                "allowed_unknown": not top.readable, "note": TOP_LAYERS_NOTE},
        "checkpoint": {"id": ft.base_checkpoint, "revision": ft.base_revision,
                       "revision_pinned": ft.base_revision != UNPINNED, "local_dir": str(local)},
        "embodiment_tag": ft.embodiment_tag,
        "hardware": {"num_gpus": n, **hw},
        "rung_a": rung_a,
        "data_mix": _data_mix(dataset),
        "heldout": dict(heldout) if heldout is not None else None,
        "seeds": {**(dict(seeds) if seeds else {}), "training": dict(TRAINING_SEED)},
        "dataset_path": str(dataset), "output_dir": str(out),
        "cwd": str(cfg.gr00t_root), "argv": list(argv), "env": dict(env),
        "result": None,
    }
    plan = FinetunePlan(tuple(argv), env, cfg.gr00t_root, out, manifest)
    manifest["command"] = plan.command
    return plan


# --------------------------------------------------------------------------- #
# after the run
# --------------------------------------------------------------------------- #

def _count(s: str) -> int:
    return int(s.replace(",", ""))


def parse_parameter_counts(log: str) -> dict:
    """``{trainable, total, trainable_percent}`` from the launcher's log (None for a line not found).
    Every rank logs the same counts; differing lines are an error."""
    trainable = {(_count(m[1]), float(m[2])) for m in TRAINABLE_RX.finditer(log)}
    total = {_count(m[1]) for m in TOTAL_RX.finditer(log)}
    if len(trainable) > 1 or len(total) > 1:
        raise ValueError(f"the log reports different parameter counts: trainable {sorted(trainable)}, total {sorted(total)}")
    t = trainable.pop() if trainable else (None, None)
    return {"trainable": t[0], "trainable_percent": t[1], "total": total.pop() if total else None}


def record_run(manifest: Mapping[str, Any], log_path: Path, output_dir: Path) -> dict:
    """``manifest`` with ``result`` filled from the log and ``final_model_config.json``."""
    log_path, output_dir = Path(log_path), Path(output_dir)
    missing = []
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    if not log:
        missing.append(f"no log at {log_path}")
    counts = parse_parameter_counts(log)
    if log and counts["trainable"] is None:
        missing.append("no 'Trainable parameters' line in the log (VERIFY(V-finetune-log))")
    final_path = output_dir / FINAL_CONFIG
    final = None
    if final_path.is_file():
        final = json.loads(final_path.read_text(encoding="utf-8"))
    else:
        missing.append(f"no {final_path}")
    before = manifest["tune_top_llm_layers"]["value"]
    after = final.get("tune_top_llm_layers") if isinstance(final, dict) else None
    result = {"parameters": counts, "final_model_config_path": str(final_path), "final_model_config": final,
              "tune_top_llm_layers_final": after,
              "tune_top_llm_layers_consistent": None if before is None or after is None else before == after,
              "log": str(log_path), "complete": not missing, "missing": missing}
    return {**manifest, "result": result}


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    from vla_isaac.experiments.tables import write_json

    write_json(Path(path), manifest)
    return Path(path)


def run_finetune(plan: FinetunePlan, runner: Optional[ArenaRunner] = None, timeout_s: float = 172800.0) -> dict:
    """Writes the manifest, runs the command, then records the parameter counts and final config."""
    write_manifest(plan.manifest_path, plan.manifest)
    log = plan.output_dir / "finetune.log"
    inv = RunInvocation(argv=plan.argv, cwd=plan.cwd, env=dict(plan.env), timeout_s=timeout_s, log_path=log,
                        output_dir=plan.output_dir, spec=_Named(f"finetune_{plan.manifest['preset']}"))  # type: ignore[arg-type]
    (runner or SubprocessRunner()).run(inv)
    done = record_run(plan.manifest, log, plan.output_dir)
    write_manifest(plan.manifest_path, done)
    return done


@dataclass(frozen=True)
class _Named:
    run_name: str


def main(argv: Optional[Sequence[str]] = None) -> int:
    from vla_isaac.config import load_backend_config

    ap = argparse.ArgumentParser(prog="python -m vla_isaac.train.finetune_launcher")
    ap.add_argument("action", choices=("plan", "run"))
    ap.add_argument("--backend", required=True)
    ap.add_argument("--pipeline", required=True, help="data pipeline / fine-tune YAML")
    ap.add_argument("--dataset", required=True, help="LeRobot directory (the mixer's output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset")
    ap.add_argument("--num-gpus", type=int)
    ap.add_argument("--heldout", help="JSON file of the held-out split (vla_isaac.data.heldout)")
    ap.add_argument("--allow-unknown-top-llm-layers", action="store_true", default=None)
    a = ap.parse_args(argv)
    pipe = PipelineConfig.load(a.pipeline)
    heldout = json.loads(Path(a.heldout).read_text(encoding="utf-8")) if a.heldout else None
    plan = plan_finetune(load_backend_config(a.backend), pipe, dataset_path=a.dataset, output_dir=a.out,
                         heldout=heldout, seeds={"heldout": pipe.heldout.seed, "mix": pipe.mix.seed},
                         preset=a.preset, num_gpus=a.num_gpus, allow_unknown_top_llm_layers=a.allow_unknown_top_llm_layers)
    print(plan.command)
    if a.action == "plan":
        print(write_manifest(plan.manifest_path, plan.manifest))
        return 0
    run_finetune(plan, timeout_s=pipe.finetune.timeout_s)
    print(plan.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
