"""Parse ``--list_variations`` output into a factor table, so the table can be regenerated on the
GPU box (``scripts/gpu/R3_list_variations.sh``).

The format follows the code that prints it (``A:isaaclab_arena/variations/variations_printing.py:
64-181``), not its docstring::

    Variations (Hydra-configurable)
    ================================

    Asset: light
      intensity (LightIntensityVariation, build-time)
        Enable: light.intensity.enabled=true  (default: False)
        Fields:
          light.intensity.sampler_cfg.high = [2000.0]
          light.intensity.sampler_cfg.low = [100.0]

Isaac Sim prints log lines around it; anything that is not part of an asset block is ignored.
VERIFY(V-list-var): the exact output for the pilot environment (R3).

Sampler inference, per variation:
  uniform    fields ``sampler_cfg.low`` and ``sampler_cfg.high`` (lists of equal length = dims)
  bernoulli  field ``sampler_cfg.probability``
  choice     ``sampler_cfg`` printed as ``{}`` (an empty ``ChoiceSamplerCfg``) and exactly one other
             list-valued field, which becomes ``values_field`` (``hdr_names`` for HDRs)
Anything else is returned as unsupported and left out of the table, never guessed.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from vla_isaac.scene.factors import FactorSpec, FactorTable

HEADER = "Variations (Hydra-configurable)"
_ASSET = re.compile(r"^Asset: (?P<asset>\S+)\s*$")
_VAR = re.compile(r"^  (?P<name>[A-Za-z0-9_]+) \((?P<cls>[A-Za-z0-9_]+), (?P<kind>build-time|run-time|unknown)\)\s*$")
_ENABLE = re.compile(r"^    Enable: (?P<path>\S+)\.enabled=true\s+\(default: (?P<default>[^)]*)\)\s*$")
_FIELD = re.compile(r"^      (?P<path>\S+) = (?P<value>.*)$")


@dataclass
class CatalogEntry:
    asset: str
    variation: str
    cls: str
    kind: str                                   # build_time | run_time | unknown
    enabled_default: Optional[str] = None
    fields: dict[str, Any] = field(default_factory=dict)   # field path relative to <asset>.<variation>

    @property
    def path(self) -> str:
        return f"{self.asset}.{self.variation}"


def _value(text: str) -> Any:
    try:
        return ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        return text.strip()


def parse_catalog(text: str) -> list[CatalogEntry]:
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == HEADER)
    except StopIteration:
        raise ValueError(f"no {HEADER!r} header in the --list_variations output") from None
    out: list[CatalogEntry] = []
    asset: Optional[str] = None
    cur: Optional[CatalogEntry] = None
    for ln in lines[start + 1:]:
        if m := _ASSET.match(ln):
            asset, cur = m["asset"], None
        elif asset and (m := _VAR.match(ln)):
            cur = CatalogEntry(asset, m["name"], m["cls"], m["kind"].replace("-", "_"))
            out.append(cur)
        elif cur and (m := _ENABLE.match(ln)):
            if m["path"] != cur.path:
                raise ValueError(f"enable line for {m['path']!r} under variation {cur.path!r}")
            cur.enabled_default = m["default"]
        elif cur and (m := _FIELD.match(ln)):
            prefix = cur.path + "."
            if not m["path"].startswith(prefix):
                raise ValueError(f"field {m['path']!r} under variation {cur.path!r}")
            cur.fields[m["path"][len(prefix):]] = _value(m["value"])
        elif ln.strip() and not ln.startswith(" ") and not ln.startswith("="):
            asset, cur = None, None           # a log line ends the current block
    return out


def infer_spec(entry: CatalogEntry, name: str) -> Optional[FactorSpec]:
    f = entry.fields
    if entry.kind not in ("build_time", "run_time"):
        return None
    low, high = f.get("sampler_cfg.low"), f.get("sampler_cfg.high")
    if isinstance(low, list) and isinstance(high, list) and len(low) == len(high) and low:
        return FactorSpec(name, entry.asset, entry.variation, entry.kind, "uniform", dims=len(low))
    if "sampler_cfg.probability" in f:
        return FactorSpec(name, entry.asset, entry.variation, entry.kind, "bernoulli")
    if f.get("sampler_cfg") == {}:
        lists = [k for k, v in f.items() if k != "sampler_cfg" and isinstance(v, list) and "." not in k]
        if len(lists) == 1:
            return FactorSpec(name, entry.asset, entry.variation, entry.kind, "choice", values_field=lists[0])
    return None


def catalog_to_table(entries: list[CatalogEntry], names: Optional[Mapping[str, str]] = None
                     ) -> tuple[FactorTable, list[str]]:
    """(factor table, unsupported variation paths). ``names`` maps ``<asset>.<variation>`` to a
    factor name (default: the path itself); a name for a path the catalogue lacks raises."""
    names = dict(names or {})
    paths = {e.path for e in entries}
    stale = sorted(set(names) - paths)
    if stale:
        raise ValueError(f"factor names given for variations the environment does not have: {stale}")
    specs: dict[str, FactorSpec] = {}
    unsupported: list[str] = []
    for e in entries:
        spec = infer_spec(e, names.get(e.path, e.path))
        if spec is None:
            unsupported.append(e.path)
        else:
            specs[spec.name] = spec
    return FactorTable(specs), unsupported


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True, help="captured stdout of policy_runner.py --list_variations")
    ap.add_argument("--output", required=True, help="factor table YAML to write")
    ap.add_argument("--names", help="YAML mapping <asset>.<variation> -> factor name")
    a = ap.parse_args(argv)
    names = yaml.safe_load(Path(a.names).read_text()) if a.names else None
    table, unsupported = catalog_to_table(parse_catalog(Path(a.input).read_text()), names)
    Path(a.output).write_text(yaml.safe_dump(table.to_dict(), sort_keys=True))
    for p in unsupported:
        print(f"unsupported (left out): {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
