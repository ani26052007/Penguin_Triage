"""Small shared pieces of the experiment tools: proportions as table cells, and atomic JSON / CSV output.

Intervals always come from ``ladder.stats.wilson_interval``; nothing here computes one."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ladder.stats import wilson_interval


def proportion(k: int, n: int, confidence: float) -> dict:
    """``{k, n, rate, lower, upper, confidence}`` with the Wilson interval of ``ladder.stats``."""
    p = wilson_interval(k, n, confidence)
    return {"k": k, "n": n, "rate": p.rate if n else None, "lower": p.lo, "upper": p.hi, "confidence": confidence}


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file and a rename, so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, data: Any) -> None:
    write_atomic(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow({c: _cell(row.get(c)) for c in columns})
    return buf.getvalue()


def _cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return ";".join(map(str, v))
    return v


def flat(prefix: str, d: Optional[Mapping[str, Any]], keys: Sequence[str]) -> dict:
    """``{prefix_key: d[key]}`` for the CSV columns; all None when ``d`` is None."""
    return {f"{prefix}_{k}": (None if d is None else d.get(k)) for k in keys}
