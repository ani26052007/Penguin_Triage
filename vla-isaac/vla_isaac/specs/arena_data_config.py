"""Arena's GR00T data config (a Python module) read as data, without importing it.

The module (e.g. ``A:isaaclab_arena_gr00t/embodiments/gr1/gr1_arms_only_data_config.py``) builds a dict
of GR00T ``ModalityConfig`` objects and registers it with ``register_modality_config(<dict>, ...)``.
Importing it needs GR00T, so this reads its syntax tree instead: the dict passed to
``register_modality_config`` (or, without that call, the module's only top-level dict literal), with
``ModalityConfig(...)`` / ``ActionConfig(...)`` calls turned into dicts of their keyword arguments and
the GR00T action enums into their member names. That is the shape GR00T writes into a checkpoint's
``processor_config.json`` (``to_json_serializable``: dataclasses as dicts, enums by name,
``G:gr00t/data/utils.py:236-277``), so the two can be compared field by field.

Anything else in the literal (a variable, an arithmetic expression, positional arguments) is a
``DataConfigError``: the reader does not guess.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

# Keyword defaults of the GR00T dataclasses (G:gr00t/data/types.py:61-84).
DEFAULTS = {
    "ModalityConfig": {"sin_cos_embedding_keys": None, "mean_std_embedding_keys": None, "action_configs": None},
    "ActionConfig": {"state_key": None},
}
ENUMS = ("ActionRepresentation", "ActionType", "ActionFormat")


class DataConfigError(ValueError):
    """A data config module this reader cannot turn into data."""


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _value(node: ast.AST, where: str) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_value(e, where) for e in node.elts]
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            if not isinstance(k, ast.Constant):
                raise DataConfigError(f"{where}: non-constant dict key at line {node.lineno}")
            out[k.value] = _value(v, where)
        return out
    if isinstance(node, ast.Attribute) and _name(node.value) in ENUMS:
        return node.attr
    if isinstance(node, ast.Call) and _name(node.func) in DEFAULTS:
        if node.args:
            raise DataConfigError(f"{where}: positional arguments to {_name(node.func)} at line {node.lineno}")
        kw = {k.arg: _value(k.value, where) for k in node.keywords}
        return {**DEFAULTS[_name(node.func)], **kw}
    raise DataConfigError(f"{where}: cannot read {ast.dump(node)[:80]} at line {getattr(node, 'lineno', '?')}")


def read_data_config(path: str | Path) -> dict[str, dict[str, Any]]:
    """``{modality: {field: value}}`` of the data config module at ``path``."""
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    dicts: dict[str, ast.Dict] = {}
    registered = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Dict) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            dicts[stmt.targets[0].id] = stmt.value
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) \
                and _name(stmt.value.func) == "register_modality_config" and stmt.value.args \
                and isinstance(stmt.value.args[0], ast.Name):
            registered = stmt.value.args[0].id
    if registered is not None:
        if registered not in dicts:
            raise DataConfigError(f"{path}: {registered!r} is registered but is not a top-level dict literal")
        node = dicts[registered]
    elif len(dicts) == 1:
        node = next(iter(dicts.values()))
    else:
        raise DataConfigError(f"{path}: no register_modality_config call and {len(dicts)} top-level dicts")
    data = _value(node, str(path))
    bad = [m for m, v in data.items() if not isinstance(v, dict)]
    if bad:
        raise DataConfigError(f"{path}: modalities {bad} are not ModalityConfig calls")
    return data
