"""The ``RolloutBackend`` contract as a reusable suite (IMPL_PLAN D1).

vla-router states the contract in ``R:ladder/backend.py`` ("one EpisodeResult per seed, in the
order of ``seeds``") and relies on it in ``R:router/prereq/rollout_cache.py`` (``CachedBackend``
asks only for missing seeds and rejects a reply for other seeds) and in the router's paired
replays (same configuration and seed -> same episode, for a deterministic backend). Its own tests
check these on ``MockBackend`` only (``R:tests/test_backend_cache.py``). This suite turns them into
one class that any backend's test subclasses; ``tests/test_backend_contract.py`` runs it on
``MockBackend`` (to show it is the same contract) and on ``IsaacBackend`` over the fakes.

A subclass implements ``make(tmp_path) -> BackendUnderTest``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from ladder.backend import RolloutBackend
from router.prereq.rollout_cache import CachedBackend, RolloutCache
from router.schemas import ActorRef, CodeRef, EpisodeResult, Mode, SceneSpec


@dataclass
class BackendUnderTest:
    backend: RolloutBackend
    scene: SceneSpec
    other_scene: SceneSpec
    code: CodeRef
    actor: ActorRef
    seeds: tuple[int, ...]               # a seed list in a deliberately unsorted order
    deterministic: bool = True


class RolloutBackendContract:
    def make(self, tmp_path) -> BackendUnderTest:            # pragma: no cover - abstract
        raise NotImplementedError

    def test_one_result_per_seed_in_the_callers_order(self, tmp_path):
        b = self.make(tmp_path)
        out = b.backend.run(b.scene, b.code, b.actor, list(b.seeds), Mode.CLEAN)
        assert [e.seed for e in out] == list(b.seeds)
        assert all(isinstance(e, EpisodeResult) and isinstance(e.success, bool) for e in out)

    def test_a_subset_returns_the_same_episodes_as_the_full_list(self, tmp_path):
        b = self.make(tmp_path)
        if not b.deterministic:
            pytest.skip("backend is not deterministic")
        full = {e.seed: e.success for e in b.backend.run(b.scene, b.code, b.actor, list(b.seeds), Mode.CLEAN)}
        part = list(reversed(b.seeds[: len(b.seeds) // 2]))
        sub = b.backend.run(b.scene, b.code, b.actor, part, Mode.CLEAN)
        assert [e.seed for e in sub] == part and all(e.success == full[e.seed] for e in sub)

    def test_cached_backend_round_trip(self, tmp_path):
        b = self.make(tmp_path)
        cb = CachedBackend(b.backend, RolloutCache(str(tmp_path / "cache.jsonl")))
        first = cb.run(b.scene, b.code, b.actor, list(b.seeds), Mode.CLEAN)
        again = cb.run(b.scene, b.code, b.actor, list(b.seeds), Mode.CLEAN)
        assert [e.to_dict() for e in again] == [e.to_dict() for e in first]
        assert cb.hits >= len(b.seeds)

    def test_results_survive_json(self, tmp_path):
        b = self.make(tmp_path)
        for e in b.backend.run(b.scene, b.code, b.actor, list(b.seeds[:3]), Mode.CLEAN):
            d = e.to_dict()                          # the rollout cache stores exactly this
            assert EpisodeResult.from_dict(json.loads(json.dumps(d))).to_dict() == d

    def test_incident_mode_also_returns_one_result_per_seed(self, tmp_path):
        b = self.make(tmp_path)
        out = b.backend.run(b.other_scene, b.code, b.actor, list(b.seeds), Mode.INCIDENT)
        assert [e.seed for e in out] == list(b.seeds)

    def test_backend_id_is_a_string(self, tmp_path):
        assert isinstance(getattr(self.make(tmp_path).backend, "backend_id", ""), str)
