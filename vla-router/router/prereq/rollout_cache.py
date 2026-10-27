"""Rollout cache (plan.md 3.2.2).

Episodes are keyed by (code+eval-config hash, actor, scene spec, seed, mode).
The plan's key is (code hash, policy checkpoint, scene spec, seed, mode); the
actor key carries the checkpoint plus the actor kind and action pipeline, so
reference-controller runs (G2 / G3 / C1) never collide with policy runs.

Router replays, Rung 1 and the data pipeline share one cache.

The key also holds the producing backend's ``backend_id`` (ISSUES #19), so two backends sharing
one cache never return each other's episodes. A backend without an id (the mock) has ``""``, and
a key with an empty id is stored without the field, so caches written before the id existed load
and match unchanged.

Note on non-determinism: a cached episode is returned as-is on later calls.
That is the point of the cache, but it means the replay-agreement measurement
(running the same seed twice) must call the wrapped backend directly
(``CachedBackend.backend``), not through the cache.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

from ladder.backend import RolloutBackend
from router.schemas import ActorRef, CodeRef, EpisodeResult, Mode, SceneSpec


@dataclass(frozen=True)
class CacheKey:
    code: str
    actor: str
    scene: str
    seed: int
    mode: str
    backend: str = ""

    @classmethod
    def make(cls, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef, seed: int, mode: Mode,
             backend: str = "") -> "CacheKey":
        return cls(code_ref.key(), actor.key(), scene.key(), int(seed), Mode(mode).value, backend)

    def to_dict(self) -> dict:
        d = {"code": self.code, "actor": self.actor, "scene": self.scene, "seed": self.seed, "mode": self.mode}
        if self.backend:
            d["backend"] = self.backend
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CacheKey":
        return cls(d["code"], d["actor"], d["scene"], int(d["seed"]), d["mode"], d.get("backend", ""))


class RolloutCache:
    """In-memory cache with optional append-only JSONL persistence.

    With ``path`` set, existing entries are loaded on construction and each
    new entry is appended, so a crash loses at most the episode in flight.
    The first entry for a key wins; later puts for the same key are ignored.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._store: dict[CacheKey, EpisodeResult] = {}
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = CacheKey.from_dict(rec["key"])
                    self._store.setdefault(key, EpisodeResult.from_dict(rec["episode"]))

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: CacheKey) -> bool:
        return key in self._store

    def get(self, key: CacheKey) -> Optional[EpisodeResult]:
        return self._store.get(key)

    def put(self, key: CacheKey, episode: EpisodeResult) -> bool:
        """Store an episode. Returns False if the key was already present."""
        if episode.seed != key.seed:
            raise ValueError(f"episode seed {episode.seed} does not match key seed {key.seed}")
        if key in self._store:
            return False
        self._store[key] = episode
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key.to_dict(), "episode": episode.to_dict()},
                                    sort_keys=True) + "\n")
        return True

    def keys(self) -> Iterator[CacheKey]:
        return iter(self._store)


class CachedBackend(RolloutBackend):
    """Wraps a backend: cached seeds are returned, only missing ones are run."""

    def __init__(self, backend: RolloutBackend, cache: RolloutCache) -> None:
        self.backend = backend
        self.cache = cache
        self.hits = 0
        self.misses = 0

    def run(self, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef,
            seeds: Sequence[int], mode: Mode) -> list[EpisodeResult]:
        mode = Mode(mode)
        seeds = [int(s) for s in seeds]
        if len(seeds) != len(set(seeds)):
            raise ValueError("duplicate seeds in one run")
        backend_id = getattr(self.backend, "backend_id", "")
        keys = {s: CacheKey.make(scene, code_ref, actor, s, mode, backend_id) for s in seeds}
        missing = [s for s in seeds if keys[s] not in self.cache]
        fresh = self.backend.run(scene, code_ref, actor, missing, mode) if missing else []
        if sorted(e.seed for e in fresh) != sorted(missing):
            raise RuntimeError("backend returned episodes for the wrong seeds")
        # counted after the run: a run the backend refused (ReplayUnavailable) simulated nothing
        self.hits += len(seeds) - len(missing)
        self.misses += len(missing)
        for ep in fresh:
            self.cache.put(keys[ep.seed], ep)
        return [self.cache.get(keys[s]) for s in seeds]
