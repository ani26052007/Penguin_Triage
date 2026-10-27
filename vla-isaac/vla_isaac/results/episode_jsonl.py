"""Arena episode JSONL -> ``EpisodeResult``.

One record per finished episode (``A:isaaclab_arena/recording/episode_recorder_manager.py:
273-303``): ``job_name``, the core fields ``env_id``, ``episode_in_env``, ``seed`` (the build seed),
``success``, ``episode_length``, ``language_instruction``, ``timestamp``
(``common_terms.py:131-148``), and when present ``variations``, ``placement``
(``common_terms.py:151-172``) and ``progress`` (``progress_terms.py:18-67``).

Episode identity (IMPL_PLAN D10). The record has no global start index. In clean mode (one env)
episode k of the build is ``episode_in_env == k``. With N envs the start order is not recorded, so
records are ranked by ``(episode_in_env, env_id)``; those results carry ``paired: False`` and must
not be paired seed by seed with clean-mode results.

``success`` is ``None`` when the task has no ``success`` termination term
(``common_terms.py:133-135``, VERIFY(V-jsonl-success)): that is an error here, never a failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from router.schemas import EpisodeResult

REQUIRED = ("env_id", "episode_in_env", "seed", "success")
KEPT = ("variations", "placement", "progress", "episode_length", "language_instruction", "job_name")


class EpisodeResultsError(RuntimeError):
    """Arena's episode records are missing, malformed or do not match the run that was asked for."""


@dataclass(frozen=True)
class EpisodeRecord:
    env_id: int
    episode_in_env: int
    build_seed: int
    success: bool
    raw: Mapping[str, Any]
    path: Path
    line: int


def _unique(items: list[tuple[str, Any]]) -> dict:
    d = dict(items)
    if len(d) != len(items):
        raise ValueError("duplicate key in JSON object")
    return d


def read_records(path: str | Path) -> list[EpisodeRecord]:
    """Strict read: malformed lines, duplicate keys and missing core fields raise."""
    path = Path(path)
    out: list[EpisodeRecord] = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw, object_pairs_hook=_unique)
        except ValueError as e:
            raise EpisodeResultsError(f"{path}:{i}: {e}") from e
        if not isinstance(rec, dict):
            raise EpisodeResultsError(f"{path}:{i}: not a JSON object")
        missing = [k for k in REQUIRED if k not in rec]
        if missing:
            raise EpisodeResultsError(f"{path}:{i}: missing {missing}")
        if not isinstance(rec["success"], bool):
            raise EpisodeResultsError(f"{path}:{i}: success is {rec['success']!r}, not a bool (does the task "
                                      f"have a 'success' termination term?)")
        out.append(EpisodeRecord(int(rec["env_id"]), int(rec["episode_in_env"]), int(rec["seed"]),
                                 rec["success"], rec, path, i))
    return out


def find_result_files(run_dir: Path) -> list[Path]:
    files = sorted(run_dir.glob("episode_results_rebuild*.jsonl")) or sorted(run_dir.glob("episode_results_rank*.jsonl"))
    if not files:
        raise EpisodeResultsError(f"no episode_results_*.jsonl in {run_dir}")
    return files


def index_records(records: list[EpisodeRecord], *, num_envs: int, num_episodes: int, build_seed: int
                  ) -> dict[int, EpisodeRecord]:
    """Episode index (start order) -> record, for the first ``num_episodes`` episodes."""
    wrong_seed = sorted({r.build_seed for r in records} - {build_seed})
    if wrong_seed:
        raise EpisodeResultsError(f"records for build seeds {wrong_seed}, expected {build_seed}")
    keys = [(r.env_id, r.episode_in_env) for r in records]
    if len(set(keys)) != len(keys):
        raise EpisodeResultsError("duplicate (env_id, episode_in_env) records")
    if num_envs == 1:
        bad = [r for r in records if r.env_id != 0]
        if bad:
            raise EpisodeResultsError(f"clean-mode run has records for env ids {sorted({r.env_id for r in bad})}")
        by_k = {r.episode_in_env: r for r in records}
    else:
        ranked = sorted(records, key=lambda r: (r.episode_in_env, r.env_id))
        by_k = dict(enumerate(ranked))
    missing = [k for k in range(num_episodes) if k not in by_k]
    if missing:
        raise EpisodeResultsError(f"the run finished {len(records)} episode(s); missing episode indices "
                                  f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")
    return {k: by_k[k] for k in range(num_episodes)}


def to_episode_result(rec: EpisodeRecord, *, seed: int, episode_index: int, paired: bool,
                      extra_logs: Mapping[str, Any]) -> EpisodeResult:
    logs: dict[str, Any] = {k: rec.raw[k] for k in KEPT if k in rec.raw}
    logs.update(build_seed=rec.build_seed, episode_index=episode_index, env_id=rec.env_id,
                episode_in_env=rec.episode_in_env, paired=paired)
    logs.update(extra_logs)
    logs["paths"] = {**dict(extra_logs.get("paths", {})), "episode_results": str(rec.path),
                     "episode_results_line": rec.line}
    return EpisodeResult(seed=seed, success=rec.success, logs=logs)
