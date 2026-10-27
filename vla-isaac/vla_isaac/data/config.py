"""The data pipeline and fine-tune config (``config/finetune_gr1_fridge.yaml``), read strictly.

Sections: ``rung_a`` (augmentation flags), ``rung_b`` (Mimic regeneration and conversion), ``heldout``
(condition split), ``mix`` (nominal vs shifted ratio) and ``finetune`` (launcher, presets, hardware).
Unknown or missing keys raise ``ConfigError``; nothing task-specific lives in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from vla_isaac.config import ConfigError, read_yaml

JITTER_KEYS = ("brightness", "contrast", "saturation", "hue")   # G:gr00t/configs/finetune_config.py:56-68
TUNE_FLAGS = ("tune_llm", "tune_visual", "tune_projector", "tune_diffusion_model")
HARDWARE_KEYS = ("global_batch_size", "max_steps", "save_steps", "save_total_limit", "dataloader_num_workers")


def _keys(d: Any, where: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict:
    if not isinstance(d, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    unknown, missing = set(d) - set(required) - set(optional), set(required) - set(d)
    if unknown or missing:
        raise ConfigError(f"{where}: unknown keys {sorted(map(str, unknown))}, missing keys {sorted(missing)}")
    return dict(d)


def _mapping(d: Any, where: str) -> dict:
    if not isinstance(d, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    return dict(d)


Condition = Mapping[str, str]             # factor name -> value; {} is the nominal condition


def condition_key(condition: Condition) -> str:
    """A canonical name for a condition: ``name=value`` pairs sorted by name, or ``nominal``."""
    return ";".join(f"{k}={condition[k]}" for k in sorted(condition)) or "nominal"


@dataclass(frozen=True)
class RungAConfig:
    color_jitter_params: Optional[Mapping[str, float]]
    random_rotation_angle: Optional[int]

    def __post_init__(self) -> None:
        j = self.color_jitter_params
        if j is not None:
            if not j or set(j) - set(JITTER_KEYS):
                raise ConfigError(f"rung_a.color_jitter_params: keys must be a non-empty subset of {JITTER_KEYS}, "
                                  f"got {sorted(j)}")
            if any(not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0 for v in j.values()):
                raise ConfigError("rung_a.color_jitter_params: values must be numbers >= 0")
        r = self.random_rotation_angle
        if r is not None and (not isinstance(r, int) or isinstance(r, bool) or r < 0):
            raise ConfigError("rung_a.random_rotation_angle must be null or an integer >= 0 (degrees)")


@dataclass(frozen=True)
class ConverterConfig:
    script: str                           # relative to the Arena root
    template: Path                        # the copied converter YAML
    action_name_sim: str


@dataclass(frozen=True)
class RungBConfig:
    episodes: int
    min_episodes: int
    max_episodes: int
    conditions: tuple[Condition, ...]
    converter: ConverterConfig

    def __post_init__(self) -> None:
        if not 1 <= self.min_episodes <= self.episodes <= self.max_episodes:
            raise ConfigError(f"rung_b: need 1 <= min_episodes <= episodes <= max_episodes, got "
                              f"{self.min_episodes} / {self.episodes} / {self.max_episodes}")
        keys = [condition_key(c) for c in self.conditions]
        if "nominal" in keys or len(set(keys)) != len(keys):
            raise ConfigError(f"rung_b.conditions must be distinct shifted conditions, got {keys}")


@dataclass(frozen=True)
class HeldoutConfig:
    seed: int
    fraction: float

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction < 1.0:
            raise ConfigError("heldout.fraction must be in (0, 1)")


@dataclass(frozen=True)
class MixConfig:
    seed: int
    nominal_dataset: str                  # backend config datasets id
    ratio: float = 0.5                    # fraction of shifted episodes

    def __post_init__(self) -> None:
        if not 0.0 < self.ratio < 1.0:
            raise ConfigError("mix.ratio must be in (0, 1): both sources are always in the mix")


@dataclass(frozen=True)
class FinetuneConfig:
    launcher: str                         # relative to the GR00T root
    modality_config_path: str             # relative to the Arena root
    embodiment_tag: str
    base_checkpoint: str                  # backend config checkpoints id
    base_revision: str
    presets: Mapping[str, Mapping[str, bool]]
    hardware: Mapping[int, Mapping[str, int]]
    preset: str = "default"
    num_gpus: int = 1
    allow_unknown_top_llm_layers: bool = False
    timeout_s: float = 172800.0           # one fine-tune run (Arena: 2-3 h for 30k steps on one GPU)

    def __post_init__(self) -> None:
        for name, flags in self.presets.items():
            if set(flags) != set(TUNE_FLAGS) or not all(isinstance(v, bool) for v in flags.values()):
                raise ConfigError(f"finetune.presets.{name} must set exactly {TUNE_FLAGS} to true/false")
        if self.preset not in self.presets:
            raise ConfigError(f"finetune.preset {self.preset!r} is not one of {sorted(self.presets)}")
        for n, hw in self.hardware.items():
            if set(hw) != set(HARDWARE_KEYS):
                raise ConfigError(f"finetune.hardware.{n} must set exactly {HARDWARE_KEYS}")
        if self.num_gpus not in self.hardware:
            raise ConfigError(f"finetune.num_gpus {self.num_gpus} has no finetune.hardware entry "
                              f"(have {sorted(self.hardware)})")


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    nominal: str
    rung_a: RungAConfig
    rung_b: RungBConfig
    heldout: HeldoutConfig
    mix: MixConfig
    finetune: FinetuneConfig
    source: Optional[Path] = field(default=None, compare=False)

    @classmethod
    def from_dict(cls, d: Mapping, base: Path = Path(".")) -> "PipelineConfig":
        top = _keys(d, "pipeline config", ("name", "nominal", "rung_a", "rung_b", "heldout", "mix", "finetune"))
        a = _keys(top["rung_a"], "rung_a", ("color_jitter_params", "random_rotation_angle"))
        jitter = a["color_jitter_params"]
        rung_a = RungAConfig(None if jitter is None else _mapping(jitter, "rung_a.color_jitter_params"),
                             a["random_rotation_angle"])
        b = _keys(top["rung_b"], "rung_b", ("episodes", "min_episodes", "max_episodes", "conditions", "converter"))
        c = _keys(b["converter"], "rung_b.converter", ("script", "template", "action_name_sim"))
        if not isinstance(b["conditions"], list):
            raise ConfigError("rung_b.conditions must be a list of {factor: value} mappings")
        conditions = tuple({str(k): str(v) for k, v in _mapping(x, "rung_b.conditions[]").items()}
                           for x in b["conditions"])
        tpl = Path(str(c["template"]))
        rung_b = RungBConfig(int(b["episodes"]), int(b["min_episodes"]), int(b["max_episodes"]), conditions,
                             ConverterConfig(str(c["script"]), tpl if tpl.is_absolute() else (base / tpl).resolve(),
                                             str(c["action_name_sim"])))
        h = _keys(top["heldout"], "heldout", ("seed", "fraction"))
        m = _keys(top["mix"], "mix", ("seed", "nominal_dataset"), ("ratio",))
        f = _keys(top["finetune"], "finetune",
                  ("launcher", "modality_config_path", "embodiment_tag", "base_checkpoint", "base_revision", "presets",
                   "hardware"), ("preset", "num_gpus", "allow_unknown_top_llm_layers", "timeout_s"))
        presets = {str(n): _keys(p, f"finetune.presets.{n}", TUNE_FLAGS)
                   for n, p in _mapping(f["presets"], "finetune.presets").items()}
        hardware = {int(n): {k: int(v) for k, v in _keys(hw, f"finetune.hardware.{n}", HARDWARE_KEYS).items()}
                    for n, hw in _mapping(f["hardware"], "finetune.hardware").items()}
        allow = f.get("allow_unknown_top_llm_layers", False)
        if not isinstance(allow, bool):
            raise ConfigError("finetune.allow_unknown_top_llm_layers must be true or false")
        finetune = FinetuneConfig(str(f["launcher"]), str(f["modality_config_path"]), str(f["embodiment_tag"]),
                                  str(f["base_checkpoint"]), str(f["base_revision"]), presets, hardware,
                                  str(f.get("preset", "default")), int(f.get("num_gpus", 1)), allow,
                                  float(f.get("timeout_s", 172800.0)))
        return cls(str(top["name"]), str(top["nominal"]), rung_a, rung_b,
                   HeldoutConfig(int(h["seed"]), float(h["fraction"])),
                   MixConfig(int(m["seed"]), str(m["nominal_dataset"]), float(m.get("ratio", 0.5))), finetune)

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        cfg = cls.from_dict(read_yaml(path), path.parent.resolve())
        object.__setattr__(cfg, "source", path.resolve())
        return cfg
