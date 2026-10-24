"""Config-key mapper (plan.md 3.2.12, Nemotron 3 Nano; contract in 3.2.4).

The config contract (``router.stage1.cfg_contract``) compares the training spec
with the eval config through ``ConfigMapping.keys``: TrainingSpec field ->
dotted path into the eval config. The plan allows a small model to *propose*
that mapping once per policy; a human checks it. This module is that proposal
step. ``cfg_contract`` itself still never calls a model.

    map_config_keys(spec, eval_config, model, cache_path=...) -> MapperResult

How it stays hand-checkable
---------------------------
* The proposal is written to a JSON file next to its inputs' structure:

      {"schema": 1, "structure_key": "...", "model": "...", "approved": false,
       "keys": {"action_horizon": "policy.chunk", ...},      <- edit this
       "unmapped": ["denoising_steps"],
       "review": {"action_horizon": {"path": "policy.chunk",
                                     "training_value": "16", "eval_value": "16"}}}

  ``review`` is for the reader (both values side by side) and is ignored on
  load. A person edits ``keys`` and sets ``approved`` to true.
* On the next call the file is read back and the model is **not** called. The
  keys are re-validated against the eval config, so a typo in a hand edit
  fails loudly instead of silently skipping a field.
* ``structure_key`` hashes the *names* in the two configs (which spec fields are
  set, which dotted paths exist), not their values. If the structure changes,
  the cache is stale and ``StaleMapping`` is raised, whether or not it was
  approved: a hand-checked mapping is never overwritten silently. Pass
  ``refresh=True`` to propose again.

What is checked on a model reply (any failure: reject, retry once, then
``MappingError``; there is no fallback mapping, because a wrong mapping makes
the contract check miss a real violation):

* exactly one key, ``keys``; every field is a TrainingSpec field that is set;
* every path exists in the eval config (a hallucinated path is rejected);
* no path used twice, and no path inside another mapped path.

``factor_covers``, ``rtol`` and ``atol`` of ``ConfigMapping`` are policy
settings, not model output; they are passed through unchanged.

TODO(verify): the real eval-config layout for the chosen policy; whether the
model id behind ``[mapper]`` follows the "JSON only" instruction.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from hagent.models import Model

from router.llm import LLMFailure, LLMOutputError, query_json
from router.schemas import TrainingSpec, short_hash
from router.stage1.cfg_contract import FIELD_ORDER, ConfigMapping

log = logging.getLogger("router.config_mapper")
SCHEMA = 1


class MappingError(ValueError):
    """No valid mapping could be produced or read."""


class StaleMapping(MappingError):
    """The cached mapping was made for a different config structure."""


@dataclass(frozen=True)
class MapperResult:
    mapping: ConfigMapping
    source: str                       # "model" or "cache"
    path: Optional[Path]
    approved: bool
    unmapped: tuple[str, ...]
    attempts: int = 0
    cost: float = 0.0
    tokens: int = 0


# --------------------------------------------------------------------------- #
# what the two configs contain
# --------------------------------------------------------------------------- #

def spec_fields(spec: TrainingSpec) -> dict[str, Any]:
    """The contract fields the training spec actually sets, in ``FIELD_ORDER``."""
    return {f: getattr(spec, f) for f in FIELD_ORDER if getattr(spec, f) is not None}


def config_paths(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Dotted path -> value for every mapping node and every leaf. Lists are leaves."""
    out: dict[str, Any] = {}

    def walk(node: Mapping, prefix: str) -> None:
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out[path] = v
            if isinstance(v, Mapping):
                walk(v, path)

    walk(cfg, "")
    return out


def structure_key(spec: TrainingSpec, eval_config: Mapping[str, Any]) -> str:
    return short_hash({"fields": list(spec_fields(spec)), "paths": sorted(config_paths(eval_config))})


def _brief(v: Any, n: int = 80) -> str:
    s = json.dumps(v, default=str, sort_keys=True)
    return s if len(s) <= n else s[: n - 3] + "..."


# --------------------------------------------------------------------------- #
# validation (model replies and hand-edited caches go through the same rules)
# --------------------------------------------------------------------------- #

def validate_keys(keys: Any, fields: Mapping[str, Any], paths: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(keys, dict):
        raise LLMOutputError("'keys' must be an object mapping a TrainingSpec field to an eval-config path or null")
    out: dict[str, str] = {}
    for field, path in keys.items():
        if field not in fields:
            raise LLMOutputError(f"'{field}' is not a TrainingSpec field that is set; "
                                 f"choose from {sorted(fields)}")
        if path is None:
            continue
        if not isinstance(path, str) or path not in paths:
            raise LLMOutputError(f"the path {path!r} for '{field}' does not exist in the eval config; "
                                 f"use one of the listed paths exactly")
        out[field] = path
    used: dict[str, str] = {}
    for field, path in out.items():
        if path in used:
            raise LLMOutputError(f"'{field}' and '{used[path]}' both map to '{path}'")
        used[path] = field
    for a, fa in used.items():
        for b, fb in used.items():
            if a != b and b.startswith(a + "."):
                raise LLMOutputError(f"'{fb}' maps to '{b}', which is inside '{a}' mapped by '{fa}'")
    return out


# --------------------------------------------------------------------------- #
# the model call
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You map configuration keys. A policy has a TRAINING spec and an EVAL config in different \
formats. For each training field, name the eval-config path that holds the same setting (the value the \
evaluation actually uses).

Reply with ONE JSON object and nothing else:
{"keys": {"<training field>": "<eval path>" or null, ...}}

Rules:
- Use only the training fields and eval paths listed below, spelled exactly.
- Use null when the eval config has no counterpart; never guess a path to fill a gap.
- Use each eval path at most once, and never a path inside another path you used.
- Match by meaning and by value (the values are shown); a sub-dictionary may match a whole training field."""


def _messages(fields: Mapping[str, Any], paths: Mapping[str, Any]) -> list[dict]:
    f = "\n".join(f"- {k}: {_brief(v)}" for k, v in fields.items())
    p = "\n".join(f"- {k}: {_brief(v)}" for k, v in paths.items())
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"## Training fields (name: value)\n{f}\n\n## Eval config paths "
                                        f"(path: value)\n{p}\n\nReturn the JSON mapping now."}]


# --------------------------------------------------------------------------- #
# the cache file
# --------------------------------------------------------------------------- #

def _review(keys: Mapping[str, str], fields: Mapping[str, Any], paths: Mapping[str, Any]) -> dict:
    return {f: {"path": p, "training_value": _brief(fields[f]), "eval_value": _brief(paths[p])}
            for f, p in keys.items()}


def _write(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def _read(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise MappingError(f"{path} is not valid JSON ({e}); fix it by hand or delete it") from None
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or not {"structure_key", "keys"} <= set(doc):
        raise MappingError(f"{path} is not a schema-{SCHEMA} mapping file")
    return doc


def map_config_keys(spec: TrainingSpec, eval_config: Mapping[str, Any], model: Optional[Model] = None, *,
                    cache_path: str | Path, factor_covers: Optional[Mapping[str, str]] = None,
                    rtol: float = 1e-3, atol: float = 1e-6, refresh: bool = False) -> MapperResult:
    """The mapping for ``spec`` and ``eval_config``: from the cache file if it
    is there and current, otherwise proposed by ``model`` and written to it."""
    cache = Path(cache_path)
    fields, paths = spec_fields(spec), config_paths(eval_config)
    key = structure_key(spec, eval_config)

    def result(keys: Mapping[str, str], source: str, approved: bool, **kw) -> MapperResult:
        return MapperResult(
            ConfigMapping(keys=dict(keys), factor_covers=dict(factor_covers or {}), rtol=rtol, atol=atol),
            source, cache, approved, tuple(f for f in fields if f not in keys), **kw)

    if cache.exists() and not refresh:
        doc = _read(cache)
        if doc["structure_key"] != key:
            raise StaleMapping(
                f"{cache} was made for a different config structure ({doc['structure_key']} vs {key}). "
                f"Review it, or pass refresh=True to propose again"
                + (" (it is marked approved: a hand-checked mapping will be replaced)" if doc.get("approved") else ""))
        try:
            keys = validate_keys(doc["keys"], fields, paths)
        except LLMOutputError as e:
            raise MappingError(f"{cache}: {e}") from None
        return result(keys, "cache", bool(doc.get("approved", False)))

    if model is None:
        raise MappingError(f"no mapping file at {cache} and no model to propose one")
    try:
        q = query_json(model, _messages(fields, paths),
                       lambda obj: _parse(obj, fields, paths))
    except LLMFailure as e:
        raise MappingError(f"the mapper model gave no valid mapping: {e}") from e
    keys: dict[str, str] = q.value
    _write(cache, {"schema": SCHEMA, "structure_key": key, "model": getattr(model, "name", "model"),
                   "approved": False, "keys": keys, "unmapped": [f for f in fields if f not in keys],
                   "review": _review(keys, fields, paths)})
    log.info("proposed config mapping written to %s (%d of %d fields mapped); check it and set approved=true",
             cache, len(keys), len(fields))
    return result(keys, "model", False, attempts=q.attempts, cost=q.cost, tokens=q.tokens)


def _parse(obj: dict, fields: Mapping[str, Any], paths: Mapping[str, Any]) -> dict[str, str]:
    if set(obj) != {"keys"}:
        raise LLMOutputError(f"the reply must have exactly one key, 'keys'; got {sorted(obj)}")
    return validate_keys(obj["keys"], fields, paths)
