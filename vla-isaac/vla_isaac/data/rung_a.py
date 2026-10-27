"""Rung A: GR00T's built-in image augmentation as fine-tune arguments (plan 5.3; facts §5).

``rung_a_args(cfg)`` gives ``--color-jitter-params k v ...`` (tyro's dict form, in config order, as in
Arena's recipe) and ``--random-rotation-angle N``. A value of ``None`` passes **no flag**, and that does not
mean "no augmentation": ``Gr00tN1d6Processor.from_pretrained`` overrides ``random_rotation_angle`` and
``color_jitter_params`` only when the passed value is not None (``G:.../processing_gr00t_n1d6.py:513-522``),
so the checkpoint's own setting is kept. ``rung_a_manifest`` says so for every ``None``, and gives the
checkpoint's value when its ``processor_config.json`` is readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from vla_isaac.data.config import RungAConfig

FIELDS = (("color_jitter_params", "--color-jitter-params"), ("random_rotation_angle", "--random-rotation-angle"))
KEEPS_CHECKPOINT = ("None: the flag is not passed, so GR00T keeps the checkpoint's own setting "
                    "(Gr00tN1d6Processor.from_pretrained overrides it only when the value is not None; facts §5, "
                    "G:gr00t/model/gr00t_n1d6/processing_gr00t_n1d6.py:513-522). It does not switch augmentation off.")


def _num(v: float) -> str:
    return repr(float(v)) if isinstance(v, float) else str(v)


def rung_a_args(cfg: RungAConfig) -> list[str]:
    args: list[str] = []
    if cfg.color_jitter_params is not None:
        args.append("--color-jitter-params")
        for k, v in cfg.color_jitter_params.items():
            args += [k, _num(v)]
    if cfg.random_rotation_angle is not None:
        args += ["--random-rotation-angle", str(cfg.random_rotation_angle)]
    return args


def checkpoint_augmentation(checkpoint_dir: Optional[Path]) -> Optional[dict]:
    """``{random_rotation_angle, color_jitter_params}`` from the checkpoint's ``processor_config.json``
    (``processor_kwargs`` or the top level), or None when it cannot be read."""
    if checkpoint_dir is None:
        return None
    path = Path(checkpoint_dir) / "processor_config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    kw = data.get("processor_kwargs", data) if isinstance(data, dict) else {}
    if not isinstance(kw, dict):
        return None
    return {name: kw.get(name) for name, _ in FIELDS if name in kw} or None


def rung_a_manifest(cfg: RungAConfig, checkpoint_dir: Optional[Path] = None) -> dict:
    ckpt = checkpoint_augmentation(checkpoint_dir)
    out: dict[str, Any] = {"args": rung_a_args(cfg), "fields": {}}
    for name, flag in FIELDS:
        value = getattr(cfg, name)
        entry: dict[str, Any] = {"value": None if value is None else (dict(value) if isinstance(value, dict) else value),
                                 "flag": flag, "passed": value is not None}
        if value is None:
            entry["note"] = KEEPS_CHECKPOINT
            entry["checkpoint_value"] = ckpt.get(name) if ckpt and name in ckpt else "unread"
        out["fields"][name] = entry
    return out
