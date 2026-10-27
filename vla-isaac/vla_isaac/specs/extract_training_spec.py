"""The policy's ``TrainingSpec`` from its checkpoint and Arena's configs, as JSON.

Inputs (paths in the task config's ``training_spec:`` section and the backend config):

* the checkpoint directory (backend ``checkpoints`` map): ``config.json`` (model config),
  ``processor_config.json`` (processor settings and the modality configs it was trained with),
  ``statistics.json`` (normalisation statistics), ``embodiment_id.json`` (embodiment tag -> id), as
  written by ``G:gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py:433-475``;
* Arena's closed-loop policy YAML, its GR00T data config module, its LeRobot ``modality.json`` and its
  HDF5 -> LeRobot converter YAML (paths relative to the Arena root);
* optionally the local LeRobot training dataset (backend ``datasets`` map): ``meta/info.json`` and
  ``meta/tasks.jsonl``. Nothing is downloaded.

Precedence: **the checkpoint's modality / processor config wins over Arena's data config** (facts §8:
the server ignores ``--modality-config-path`` when a model path is given). Every field that differs
between them is listed in ``extra["disagreements"]`` with every value and the one used. The dataset's
own metadata wins over Arena's converter YAML for the frame rate and image size.

Every ``TrainingSpec`` field is either filled, with its source in ``extra["sources"]``, or explicitly
missing, with the reason in ``extra["missing"]`` (the router's config contract skips a missing field, so
the reason is the only record of it). In particular ``instruction_templates`` comes only from the
dataset's ``meta/tasks.jsonl``; without a local dataset it is missing (VERIFY(V-train-instr)), never
filled from the eval instruction or from Arena's YAMLs (which disagree with each other, IMPL_PLAN D4;
both strings are kept in ``extra["arena_instructions"]`` for the reader).

``norm_stats`` is ``vla_isaac.specs.norm_stats.training_norm_stats``, the function the eval-config
loader uses for ``server.effective_norm_stats``.

CLI: ``python -m vla_isaac.specs.extract_training_spec --backend backend.yaml --output spec.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from router.schemas import TrainingSpec

from vla_isaac.config import ConfigError, read_yaml
from vla_isaac.specs.arena_data_config import read_data_config
from vla_isaac.specs.norm_stats import STATISTICS_FILE, training_norm_stats

MODEL_CONFIG_FILE = "config.json"
PROCESSOR_CONFIG_FILE = "processor_config.json"
EMBODIMENT_ID_FILE = "embodiment_id.json"
CHECKPOINT_FILES = (MODEL_CONFIG_FILE, PROCESSOR_CONFIG_FILE, STATISTICS_FILE, EMBODIMENT_ID_FILE)

# TrainingSpec fields the extraction fills or marks missing (all but the name and ``extra``).
SPEC_FIELDS = tuple(f.name for f in fields(TrainingSpec) if f.name not in ("policy_name", "extra"))
# Fields no checkpoint or Arena file records; the task config may declare them (with a VERIFY id).
DECLARABLE = ("channel_order", "render_settings", "camera_poses")
MODALITY_FIELDS = ("delta_indices", "modality_keys", "sin_cos_embedding_keys", "mean_std_embedding_keys",
                   "action_configs")
PROCESSOR_KEYS = ("image_crop_size", "image_target_size", "shortest_image_edge", "crop_fraction",
                  "use_albumentations", "random_rotation_angle", "color_jitter_params", "formalize_language",
                  "apply_sincos_state_encoding", "use_relative_action", "use_percentiles", "clip_outliers",
                  "max_state_dim", "max_action_dim", "max_action_horizon", "model_name")
MODEL_KEYS = ("model_type", "model_name", "action_horizon", "num_inference_timesteps", "tune_top_llm_layers",
              "tune_llm", "tune_visual", "tune_projector", "tune_diffusion_model", "max_state_dim", "max_action_dim")


ABSENT = object()        # a source that has no value for a field (None is a value)


class SpecError(ValueError):
    """Inputs the training spec cannot be extracted from (a missing checkpoint file, an unknown embodiment)."""


@dataclass(frozen=True)
class TrainingSpecSection:
    """The task config's ``training_spec:`` section."""
    policy_name: str
    checkpoint: str                          # id in the backend config's ``checkpoints`` map
    embodiment_tag: str
    arena_closedloop_config: str             # relative to the Arena root
    arena_data_config: str
    arena_modality_json: str
    arena_converter_config: str
    video_key: str                           # modality video key whose dataset feature gives the image size
    dataset: Optional[str] = None            # id in the backend config's ``datasets`` map (local LeRobot dir)
    declared: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bad = set(self.declared) - set(DECLARABLE)
        if bad:
            raise ConfigError(f"training_spec.declared may only name {DECLARABLE}, not {sorted(bad)}")


def section_from_dict(d: Mapping) -> TrainingSpecSection:
    known = {f.name for f in fields(TrainingSpecSection)}
    if not isinstance(d, Mapping):
        raise ConfigError("task config [training_spec] must be a mapping")
    unknown = set(d) - known
    missing = {f.name for f in fields(TrainingSpecSection) if f.name not in ("dataset", "declared")} - set(d)
    if unknown or missing:
        raise ConfigError(f"task config [training_spec]: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    kw = dict(d)
    kw["declared"] = dict(kw.get("declared") or {})
    return TrainingSpecSection(**kw)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SpecError(f"{path}: {e}") from e


class _Record:
    """Values, sources, missing reasons and disagreements, collected field by field."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.sources: dict[str, str] = {}
        self.missing: dict[str, str] = {}
        self.disagreements: list[dict[str, Any]] = []

    def fill(self, name: str, value: Any, source: str) -> None:
        self.values[name], self.sources[name] = value, source

    def miss(self, name: str, reason: str) -> None:
        self.missing[name] = reason

    def compare(self, name: str, used: str, values: Mapping[str, Any]) -> None:
        """Record ``name`` as a disagreement when the sources in ``values`` that have a value (not ``ABSENT``)
        do not all agree."""
        present = {k: v for k, v in values.items() if v is not ABSENT}
        if len({json.dumps(v, sort_keys=True) for v in present.values()}) > 1:
            self.disagreements.append({"field": name, "used": used, "values": dict(present)})


def read_tasks_jsonl(path: Path) -> dict[str, str]:
    """``meta/tasks.jsonl`` of a LeRobot dataset -> ``{"task_<index>": task}``."""
    out = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            out[f"task_{int(row['task_index'])}"] = str(row["task"])
        except (ValueError, KeyError, TypeError) as e:
            raise SpecError(f"{path}:{n}: not a task row ({e})") from e
    if not out:
        raise SpecError(f"{path}: no tasks")
    return out


def _layout(groups: Sequence[str], stats: Mapping[str, Any], where: str) -> dict[str, list[int]]:
    """``{group: [start, end)}`` from the per-group statistics lengths, in group order."""
    out, start = {}, 0
    for g in groups:
        n = len(stats[g]["min"]) if "min" in stats[g] else len(next(iter(stats[g].values())))
        out[g] = [start, start + n]
        start += n
    if not out:
        raise SpecError(f"{where}: no groups")
    return out


def extract_training_spec(section: TrainingSpecSection, *, checkpoint_dir: str | Path, arena_root: str | Path,
                          fallback_groups: Mapping[str, Sequence[str]],
                          dataset_dir: Optional[str | Path] = None) -> TrainingSpec:
    ckpt, arena = Path(checkpoint_dir), Path(arena_root)
    absent = [f for f in CHECKPOINT_FILES if not (ckpt / f).is_file()]
    if absent:
        raise SpecError(f"checkpoint {ckpt} lacks {absent}")
    tag = section.embodiment_tag.lower()
    model = _json(ckpt / MODEL_CONFIG_FILE)
    processor = _json(ckpt / PROCESSOR_CONFIG_FILE).get("processor_kwargs", {})
    statistics = _json(ckpt / STATISTICS_FILE)
    ids = _json(ckpt / EMBODIMENT_ID_FILE)
    ckpt_modalities = processor.get("modality_configs", {}).get(tag)
    if ckpt_modalities is None or tag not in ids:
        raise SpecError(f"checkpoint {ckpt} has no embodiment {tag!r} (modality configs "
                        f"{sorted(processor.get('modality_configs', {}))}, ids {sorted(ids)})")
    closedloop = read_yaml(arena / section.arena_closedloop_config)
    converter = read_yaml(arena / section.arena_converter_config)
    arena_modalities = read_data_config(arena / section.arena_data_config)
    modality_json = _json(arena / section.arena_modality_json)
    files = {"checkpoint": str(ckpt), "arena_closedloop": section.arena_closedloop_config,
             "arena_data_config": section.arena_data_config, "arena_modality_json": section.arena_modality_json,
             "arena_converter": section.arena_converter_config}
    r = _Record()

    # modality configs: the checkpoint's, compared with Arena's data config field by field
    for m in sorted(set(ckpt_modalities) | set(arena_modalities)):
        c, a = ckpt_modalities.get(m), arena_modalities.get(m)
        for f in MODALITY_FIELDS:
            r.compare(f"modality.{m}.{f}", "checkpoint",
                      {"checkpoint": c.get(f) if c is not None else "<no such modality>",
                       "arena_data_config": a.get(f) if a is not None else "<no such modality>"})
    r.compare("embodiment_tag", "checkpoint",
              {"checkpoint": tag, "arena_closedloop": str(closedloop["embodiment_tag"]).lower()
                       if "embodiment_tag" in closedloop else ABSENT})

    # chunk length, replan steps, denoising
    delta = (ckpt_modalities.get("action") or {}).get("delta_indices")
    if delta:
        r.fill("action_horizon", len(delta), f"checkpoint:{PROCESSOR_CONFIG_FILE} modality_configs.{tag}.action.delta_indices")
    else:
        r.miss("action_horizon", "the checkpoint's action modality config has no delta_indices")
    arena_delta = (arena_modalities.get("action") or {}).get("delta_indices")
    r.compare("action_horizon", "checkpoint", {"checkpoint": len(delta) if delta else ABSENT,
                                               "arena_data_config": len(arena_delta) if arena_delta else ABSENT,
                                               "arena_closedloop": closedloop.get("action_horizon", ABSENT)})
    if closedloop.get("action_chunk_length") is not None:
        r.fill("replan_steps", int(closedloop["action_chunk_length"]), "arena_closedloop:action_chunk_length")
    else:
        r.miss("replan_steps", "neither the checkpoint nor Arena's closed-loop config records the replan steps")
    if model.get("num_inference_timesteps") is not None:
        r.fill("denoising_steps", int(model["num_inference_timesteps"]), f"checkpoint:{MODEL_CONFIG_FILE} num_inference_timesteps")
    else:
        r.miss("denoising_steps", f"checkpoint {MODEL_CONFIG_FILE} has no num_inference_timesteps")

    # dataset metadata (local only) vs Arena's converter: frame rate and image size
    info = tasks = None
    meta = Path(dataset_dir) / "meta" if dataset_dir is not None else None
    if meta is not None:
        files["dataset"] = str(dataset_dir)
        info = _json(meta / "info.json") if (meta / "info.json").is_file() else None
    feature = (modality_json.get("video", {}).get(section.video_key) or {}).get("original_key")
    ds_fps = info.get("fps") if info else None
    ds_shape = ((info.get("features") or {}).get(feature) or {}).get("shape") if info and feature else None
    conv_fps, conv_size = converter.get("fps"), converter.get("target_image_size")
    if ds_fps is not None:
        r.fill("control_hz", float(ds_fps), "dataset:meta/info.json fps")
    elif conv_fps is not None:
        r.fill("control_hz", float(conv_fps), "arena_converter:fps")
    else:
        r.miss("control_hz", "no frame rate in the dataset metadata or Arena's converter config")
    r.compare("control_hz", r.sources.get("control_hz", "").split(":")[0] or "none",
              {"dataset": float(ds_fps) if ds_fps is not None else ABSENT,
               "arena_converter": float(conv_fps) if conv_fps is not None else ABSENT})
    if ds_shape:
        r.fill("image_size", (int(ds_shape[0]), int(ds_shape[1])), f"dataset:meta/info.json features.{feature}.shape")
    elif conv_size:
        r.fill("image_size", (int(conv_size[0]), int(conv_size[1])), "arena_converter:target_image_size")
    else:
        r.miss("image_size", f"no image size for {feature!r} in the dataset metadata or Arena's converter config")
    r.compare("image_size", r.sources.get("image_size", "").split(":")[0] or "none",
              {"dataset": list(ds_shape[:2]) if ds_shape else ABSENT,
               "arena_converter": list(conv_size[:2]) if conv_size else ABSENT})

    # instruction: the dataset's tasks only (V-train-instr)
    if meta is None:
        r.miss("instruction_templates", "no local training dataset configured (training_spec.dataset / backend "
                                        "datasets); never taken from the eval instruction. VERIFY(V-train-instr)")
    elif not (meta / "tasks.jsonl").is_file():
        r.miss("instruction_templates", f"no meta/tasks.jsonl in {dataset_dir}. VERIFY(V-train-instr)")
    else:
        tasks = read_tasks_jsonl(meta / "tasks.jsonl")
        r.fill("instruction_templates", tasks, "dataset:meta/tasks.jsonl")

    # normalisation statistics and layouts: the checkpoint's, in its group order
    ns = training_norm_stats(ckpt, tag, fallback_groups)
    r.fill("norm_stats", ns.stats, f"checkpoint:{STATISTICS_FILE} (groups from {ns.groups_source})")
    r.compare("norm_stats_groups", "checkpoint", {"checkpoint": {m: list(g) for m, g in ns.groups.items()},
                                                  "task_config": {m: list(g) for m, g in fallback_groups.items()}})
    for m, name in (("state", "state_layout"), ("action", "action_layout")):
        layout = _layout(ns.groups[m], statistics[tag][m], f"{STATISTICS_FILE} {tag}.{m}")
        r.fill(name, layout, f"checkpoint:{STATISTICS_FILE} {tag}.{m} group lengths in modality order")
        arena_layout = {g: [int(v["start"]), int(v["end"])] for g, v in modality_json.get(m, {}).items()}
        r.compare(name, "checkpoint", {"checkpoint": layout, "arena_modality_json": arena_layout or ABSENT})

    # declared values: fields no file records
    for name in DECLARABLE:
        if name in section.declared:
            value = section.declared[name]
            r.fill(name, value, "task config:training_spec.declared")
        elif name not in r.values:
            r.miss(name, "not recorded in the checkpoint or Arena's configs and not declared in the task config")

    unaccounted = set(SPEC_FIELDS) - set(r.values) - set(r.missing)
    assert not unaccounted, f"fields neither filled nor missing: {sorted(unaccounted)}"
    extra = {
        "checkpoint": section.checkpoint, "embodiment_tag": tag, "embodiment_id": ids[tag], "files": files,
        "sources": r.sources, "missing": r.missing, "disagreements": r.disagreements,
        "processor": {k: processor.get(k) for k in PROCESSOR_KEYS},
        "model": {k: model.get(k) for k in MODEL_KEYS},
        "arena_instructions": {"arena_closedloop": closedloop.get("language_instruction"),
                               "arena_converter": converter.get("language_instruction")},
    }
    return TrainingSpec(policy_name=section.policy_name, extra=extra,
                        **{k: r.values[k] for k in SPEC_FIELDS if k in r.values})


def extract_for_backend(backend_config: str | Path, *, checkpoint_dir: Optional[str | Path] = None) -> TrainingSpec:
    """The task's training spec with the paths of a backend config (checkpoint and dataset maps)."""
    from vla_isaac.config import load_backend_config, load_task_config

    cfg = load_backend_config(backend_config)
    task = load_task_config(cfg.task_config)
    s = task.training_spec
    if s is None:
        raise ConfigError(f"task config {task.source} has no training_spec section")
    if checkpoint_dir is None:
        if s.checkpoint not in cfg.checkpoints:
            raise ConfigError(f"checkpoint {s.checkpoint!r} is not in the backend config's checkpoints map")
        checkpoint_dir = cfg.checkpoints[s.checkpoint]
    dataset_dir = None
    if s.dataset is not None:
        dataset_dir = cfg.datasets.get(s.dataset)      # absent: instruction_templates is missing (V-train-instr)
    return extract_training_spec(s, checkpoint_dir=checkpoint_dir, arena_root=cfg.arena_root,
                                 fallback_groups=task.eval_config.norm_stats_groups, dataset_dir=dataset_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", required=True, help="backend config YAML")
    p.add_argument("--output", required=True, help="where to write the TrainingSpec JSON")
    p.add_argument("--checkpoint-dir", default=None, help="local checkpoint directory (default: the checkpoints map)")
    a = p.parse_args(argv)
    spec = extract_for_backend(a.backend, checkpoint_dir=a.checkpoint_dir)
    Path(a.output).write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    missing = spec.extra["missing"]
    print(f"wrote {a.output}: {len(SPEC_FIELDS) - len(missing)} fields filled, {len(missing)} missing "
          f"({', '.join(sorted(missing)) or 'none'}), {len(spec.extra['disagreements'])} disagreements")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
