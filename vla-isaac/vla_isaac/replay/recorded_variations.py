"""Replay a recording with one factor restored: strip that factor's key from an Arena episode JSONL.

Arena replays the ``variations`` recorded in an episode-result JSONL
(``--recorded_variation_samples_path``, ``A:isaaclab_arena/variations/recorded_variation_replay.py``).
Restoring factor i means disabling its variation, but Arena **asserts** when a recorded key has no
enabled variation on the replay build (``A:isaaclab_arena/variations/recorded_variation_samples.py:
122-142``). So the key has to leave the file first. ``strip_key`` writes a copy of the JSONL with
``variations[key]`` removed from every record and everything else (placement, the other keys, the
other record fields) unchanged.

It refuses, naming the key, before writing anything:

* a key that no record holds (nothing to strip; the wrong file or a typo);
* a key that is still enabled on the replay build (Arena would sample it live, not restore it);
* **any remaining key that does not belong to an enabled variation on the replay build**, which Arena
  would assert on. The message names every such key and the first record that holds it.

``strip_factor`` does the same from a scene and the factor table: the key is the factor's
``<asset>.<variation>`` and the enabled keys are those of ``scene.restore(factor)`` (variations start
disabled, so a replay build enables exactly the scene's factors). The recorded key format
``host.variation`` equals the factor table's path (``A:isaaclab_arena/variations/recorded_variation_samples.py:25,36``); the value
format is VERIFY(V-var-record).

Records are read as Arena reads them: one JSON object per non-empty line, duplicate keys rejected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from router.schemas import SceneSpec

from vla_isaac.experiments.tables import write_atomic
from vla_isaac.scene.factors import FactorTable


class RecordedVariationError(ValueError):
    """A recording that Arena would not accept for replay with the factor restored."""


def _unique(items: list[tuple[str, Any]]) -> dict:
    d = dict(items)
    if len(d) != len(items):
        raise RecordedVariationError("duplicate key in a JSON object")
    return d


def read_records(path: str | Path) -> list[dict]:
    records = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line, object_pairs_hook=_unique)
        except (json.JSONDecodeError, RecordedVariationError) as exc:
            raise RecordedVariationError(f"{path}, line {n}: {exc}") from exc
        if not isinstance(rec, dict):
            raise RecordedVariationError(f"{path}, line {n}: expected a JSON object")
        variations = rec.get("variations", {})
        if not isinstance(variations, dict):
            raise RecordedVariationError(f"{path}, line {n}: variations must be a mapping")
        records.append(rec)
    if not records:
        raise RecordedVariationError(f"no episode records in {path}")
    return records


def strip_records(records: Sequence[dict], key: str, enabled_keys: Iterable[str]) -> list[dict]:
    """``records`` with ``variations[key]`` removed; refuses what Arena would refuse (module docstring)."""
    enabled = set(enabled_keys)
    if key in enabled:
        raise RecordedVariationError(f"{key!r} is still enabled on the replay build; disable its variation to "
                                     f"restore it (Arena would sample it live)")
    if not any(key in r.get("variations", {}) for r in records):
        raise RecordedVariationError(f"no record holds variation {key!r}; recorded keys: "
                                     f"{sorted({k for r in records for k in r.get('variations', {})})}")
    out, unknown = [], {}
    for i, rec in enumerate(records):
        rec = json.loads(json.dumps(rec))
        variations = rec.get("variations", {})
        variations.pop(key, None)
        for k in variations:
            if k not in enabled:
                unknown.setdefault(k, i)
        out.append(rec)
    if unknown:
        listed = ", ".join(f"{k!r} (first in record {i})" for k, i in sorted(unknown.items()))
        raise RecordedVariationError(f"after stripping {key!r}, the recording still holds keys with no enabled "
                                     f"variation on the replay build: {listed}; Arena asserts on them "
                                     f"(recorded_variation_samples.py:122-142)")
    return out


def strip_key(src: str | Path, dst: str | Path, key: str, enabled_keys: Iterable[str]) -> Path:
    """Write ``src`` without ``variations[key]`` to ``dst`` (nothing is written when it refuses)."""
    out = strip_records(read_records(src), key, enabled_keys)
    dst = Path(dst)
    if dst.resolve() == Path(src).resolve():
        raise RecordedVariationError("the stripped copy must not overwrite the recording")
    write_atomic(dst, "".join(json.dumps(r, sort_keys=False) + "\n" for r in out))
    return dst


def enabled_keys(table: FactorTable, scene: SceneSpec) -> set[str]:
    """The ``<asset>.<variation>`` keys a build of ``scene`` enables: exactly its factors'."""
    return {table.spec(f.name).path for f in scene.factors}


def strip_factor(src: str | Path, dst: str | Path, table: FactorTable, scene: SceneSpec, factor: str) -> Path:
    """The recording of ``scene`` made replayable on ``scene.restore(factor)``."""
    restored = scene.restore(factor)
    return strip_key(src, dst, table.spec(factor).path, enabled_keys(table, restored))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vla_isaac.replay.recorded_variations",
                                 description="strip one variation key from an Arena episode-result JSONL")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--strip", required=True, help="the recorded key, <asset>.<variation>")
    ap.add_argument("--enabled", nargs="*", default=[], help="the keys enabled on the replay build")
    a = ap.parse_args(argv)
    print(strip_key(a.input, a.output, a.strip, a.enabled))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
