"""Stage 1 check 1: config contract (plan.md 3.2.4).

Compares the eval config (``incident.code_ref.eval_config``) with the training
spec. A difference that is not a declared factor of the scene spec is a
violation. Fields compared, as listed in 3.2.4: chunk length (``action_horizon``),
replan steps, denoising steps, control frequency, normalisation stats, camera
poses, image size, render settings and the instruction string.

This is a plain rule. How keys in the two config formats map onto each other
is data (``ConfigMapping.keys``), hand-checked once per policy; a small model
may propose it (3.2.12) but nothing here calls one. TODO(verify): the real eval
config layout and key names for the chosen policy.

Signals
-------
* ``contract_violation``      any field except render settings.
* ``render_settings_differ``  render settings only. Plan 3.2.4 calls every
  undeclared difference a violation, but a render difference is row 5 of the
  decision table (a config fix verified by ``C_render``), not row 4 (code), so
  it gets its own signal (DECISIONS.md, Step 3).

Strength: 1.0 for a structural or non-numeric mismatch. For a number outside
its tolerance, ``saturating(deviation / tolerance)``. The no-fault config
matches exactly, so there is no spread to calibrate against; the tolerances
are the nominal band and are passed in as data.

A mapped key that is missing from the eval config is skipped with a warning:
the effective value is then a default the check cannot see.

A mapped field the training spec does not have (None) cannot be compared
(router change 6). It is skipped as before, and the check says so in its notes,
one line per field: "not compared: <field> (missing in spec)". A note, never a
skip: the evidence, and therefore the verdict, are exactly as before.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from router import vocab as V
from router.schemas import Evidence, Incident
from router.stage1.base import Check, log, saturating

FIELD_ORDER = ("action_horizon", "replan_steps", "denoising_steps", "control_hz", "norm_stats",
               "camera_poses", "image_size", "render_settings", "instruction_templates")


@dataclass(frozen=True)
class ConfigMapping:
    """keys:          TrainingSpec field -> dotted path into the eval config.
    factor_covers: scene-spec factor name -> TrainingSpec field it may
                   legitimately change (``camera_offset`` -> ``camera_poses``).
                   A difference in a covered field is the declared test
                   condition, not a violation.
    rtol / atol:   numeric tolerance, ``|a - e| <= atol + rtol * |e|``."""
    keys: Mapping[str, str]
    factor_covers: Mapping[str, str] = field(default_factory=dict)
    rtol: float = 1e-3
    atol: float = 1e-6

    def __post_init__(self) -> None:
        bad = (set(self.keys) | set(self.factor_covers.values())) - set(FIELD_ORDER)
        if bad:
            raise ValueError(f"unsupported TrainingSpec fields {sorted(bad)}; supported: {FIELD_ORDER}")
        if self.rtol < 0 or self.atol < 0 or (self.rtol == 0 and self.atol == 0):
            raise ValueError("need rtol, atol >= 0, not both zero")


@dataclass(frozen=True)
class Diff:
    path: str
    expected: Any
    actual: Any
    strength: float

    def text(self) -> str:
        return f"{self.path or '<value>'}: expected {self.expected!r}, got {self.actual!r}"


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_seq(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def _get_path(cfg: Mapping, dotted: str) -> tuple[bool, Any]:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _diff(expected: Any, actual: Any, path: str, rtol: float, atol: float, out: list[Diff]) -> None:
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not (isinstance(expected, Mapping) and isinstance(actual, Mapping)):
            out.append(Diff(path, expected, actual, 1.0))
            return
        for k in sorted(set(expected) | set(actual), key=str):
            sub = f"{path}.{k}" if path else str(k)
            if k not in expected:
                out.append(Diff(sub, "<absent>", actual[k], 1.0))
            elif k not in actual:
                out.append(Diff(sub, expected[k], "<absent>", 1.0))
            else:
                _diff(expected[k], actual[k], sub, rtol, atol, out)
        return
    if _is_seq(expected) or _is_seq(actual):
        if not (_is_seq(expected) and _is_seq(actual)) or len(expected) != len(actual):
            out.append(Diff(path, expected, actual, 1.0))
            return
        for i, (e, a) in enumerate(zip(expected, actual)):
            _diff(e, a, f"{path}[{i}]", rtol, atol, out)
        return
    if _is_num(expected) and _is_num(actual):
        if math.isnan(expected) or math.isnan(actual):
            if not (math.isnan(expected) and math.isnan(actual)):
                out.append(Diff(path, expected, actual, 1.0))
            return
        tol = atol + rtol * abs(expected)
        dev = abs(actual - expected)
        if dev > tol:
            out.append(Diff(path, expected, actual, saturating(dev / tol)))
        return
    if expected != actual:
        out.append(Diff(path, expected, actual, 1.0))


def _join(base: str, rel: str) -> str:
    """Config path of a difference found at ``rel`` inside the value at ``base``."""
    if not rel:
        return base
    return f"{base}{rel}" if rel.startswith("[") else f"{base}.{rel}"


def _template_regex(template: str) -> "re.Pattern[str]":
    """``"pick up the {object}"`` -> regex matching ``"pick up the mug"``."""
    parts = re.split(r"\{[^{}]*\}", template)
    return re.compile(".+?".join(re.escape(p) for p in parts), re.DOTALL)


def _instruction_diffs(templates: Mapping[str, str], actual: Any, path: str) -> list[Diff]:
    """Every instruction string in the eval config must match some training
    template. ``actual`` is one string or a mapping of strings."""
    if isinstance(actual, str):
        found = {path: actual}
    elif isinstance(actual, Mapping) and all(isinstance(v, str) for v in actual.values()):
        found = {f"{path}.{k}": v for k, v in actual.items()}
    else:
        return [Diff(path, "<instruction string(s)>", actual, 1.0)]
    patterns = [_template_regex(t) for t in templates.values()]
    return [Diff(p, "a string matching one of " + ", ".join(repr(t) for t in templates.values()), s, 1.0)
            for p, s in found.items() if not any(rx.fullmatch(s.strip()) for rx in patterns)]


class ConfigContractCheck(Check):
    """Needs no traces: the eval config and the training spec are on the incident."""

    source = "contract_cfg"

    def __init__(self, mapping: ConfigMapping) -> None:
        self.mapping = mapping
        self._missing: dict[str, tuple[str, ...]] = {}   # incident id -> mapped fields its spec lacked (last run)

    def run(self, incident: Incident) -> list[Evidence]:
        spec, cfg = incident.training_spec, incident.code_ref.eval_config
        covered = {self.mapping.factor_covers[f] for f in incident.scene.factor_names
                   if f in self.mapping.factor_covers}
        out: list[Evidence] = []
        missing: list[str] = []
        for name in FIELD_ORDER:
            path = self.mapping.keys.get(name)
            if path is None:
                continue
            expected = getattr(spec, name)
            if expected is None:
                missing.append(name)
                continue
            if name in covered:
                log.debug("cfg contract: %s may differ (declared factor in the scene spec)", name)
                continue
            found, actual = _get_path(cfg, path)
            if not found:
                log.warning("cfg contract: eval config has no %r (for %s); not compared", path, name)
                continue
            if name == "instruction_templates":
                diffs = _instruction_diffs(expected, actual, path)
            else:
                diffs = []
                _diff(expected, actual, "", self.mapping.rtol, self.mapping.atol, diffs)
                diffs = [Diff(_join(path, d.path), d.expected, d.actual, d.strength) for d in diffs]
            if not diffs:
                continue
            shown = "; ".join(d.text() for d in diffs[:3]) + (f"; (+{len(diffs) - 3} more)" if len(diffs) > 3 else "")
            render = name == "render_settings"
            out.append(Evidence(
                id=f"s1.cfg.{name}", source=self.source,
                signal=V.SIG_RENDER_SETTINGS_DIFFER if render else V.SIG_CONTRACT_VIOLATION,
                strength=max(d.strength for d in diffs), location=f"eval_config.{path}",
                detail=f"eval config differs from the training spec ({name}) and no scene factor declares it: {shown}"))
        self._missing[incident.incident_id] = tuple(missing)
        return out

    def notes(self, incident: Incident) -> list[str]:
        """The mapped fields the last run could not compare because the training spec lacks them."""
        return [f"contract_cfg: not compared: {name} (missing in spec)" for name in self._missing.get(incident.incident_id, ())]
