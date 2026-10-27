import pytest

from ladder.backend import MockBackend, MockRule
from ladder.stats import agreement_rate
from router.prereq.rollout_cache import CacheKey, CachedBackend, RolloutCache
from router.schemas import (
    ActionPipeline, ActorKind, ActorRef, CodeRef, EpisodeResult, Factor, Mode, SceneSpec,
)

NOMINAL = SceneSpec("pick_mug")
DARK = SceneSpec("pick_mug", (Factor("lighting", "dark"),))
DARK_GLOSSY = SceneSpec("pick_mug", (Factor("lighting", "dark"), Factor("material(mug)", "glossy")))
CODE = CodeRef("abc123", {"replan_steps": 8})
CODE_BUGGY = CodeRef("abc123", {"replan_steps": 16})       # same commit, different eval config
POLICY = ActorRef(ActorKind.POLICY, "ckpt-1")
REF_RAW = ActorRef(ActorKind.REFERENCE, "ik-v1", ActionPipeline.RAW)
REF_WRAP = ActorRef(ActorKind.REFERENCE, "ik-v1", ActionPipeline.WRAPPER)
SEEDS = list(range(2000))


def _rate(eps):
    return sum(e.success for e in eps) / len(eps)


def _table():
    return MockBackend(
        rules=[
            MockRule(rate=1.0, actor=REF_RAW),                        # reference controller always works
            MockRule(rate=0.2, code=CODE_BUGGY),                      # injected config bug
            MockRule(rate=0.6, mode=Mode.INCIDENT),                   # server staleness
            MockRule(rate=0.3, factors_present=("lighting",)),        # vision gap, any scene with dark lighting
        ],
        default_rate=0.85,
    )


# ------------------------------------------------------------------ mock ----

def test_mock_rule_table_first_match_wins():
    be = _table()
    assert be.rate_for(DARK, CODE_BUGGY, REF_RAW, Mode.INCIDENT) == 1.0
    assert be.rate_for(DARK, CODE_BUGGY, POLICY, Mode.CLEAN) == 0.2
    assert be.rate_for(NOMINAL, CODE, POLICY, Mode.INCIDENT) == 0.6
    assert be.rate_for(DARK_GLOSSY, CODE, POLICY, Mode.CLEAN) == 0.3
    assert be.rate_for(NOMINAL, CODE, POLICY, Mode.CLEAN) == 0.85      # default
    assert be.rate_for(NOMINAL, CODE, REF_WRAP, Mode.CLEAN) == 0.85    # raw-only rule does not match wrapper


def test_mock_exact_scene_rule():
    be = MockBackend([MockRule(rate=0.1, scene=DARK)], default_rate=0.9)
    assert be.rate_for(DARK, CODE, POLICY, Mode.CLEAN) == 0.1
    assert be.rate_for(DARK_GLOSSY, CODE, POLICY, Mode.CLEAN) == 0.9   # exact match only


@pytest.mark.parametrize("scene,code,actor,mode,rate", [
    (NOMINAL, CODE, POLICY, Mode.CLEAN, 0.85),
    (DARK, CODE, POLICY, Mode.CLEAN, 0.3),
    (NOMINAL, CODE_BUGGY, POLICY, Mode.CLEAN, 0.2),
])
def test_mock_empirical_rate_matches_table(scene, code, actor, mode, rate):
    eps = _table().run(scene, code, actor, SEEDS, mode)
    assert [e.seed for e in eps] == SEEDS
    assert _rate(eps) == pytest.approx(rate, abs=0.035)    # ~4 sd at n=2000


def test_mock_is_deterministic_without_noise():
    a = _table().run(DARK, CODE, POLICY, SEEDS[:200], Mode.CLEAN)
    b = _table().run(DARK, CODE, POLICY, SEEDS[:200], Mode.CLEAN)
    assert [e.success for e in a] == [e.success for e in b]
    be = _table()
    first = {e.seed: e.success for e in be.run(DARK, CODE, POLICY, SEEDS[:200], Mode.CLEAN)}
    again = {e.seed: e.success for e in be.run(DARK, CODE, POLICY, SEEDS[:200], Mode.CLEAN)}
    assert agreement_rate(first, again) == 1.0


def test_mock_replay_noise_lowers_agreement_but_not_first_run():
    quiet = MockBackend([], default_rate=0.5)
    noisy = MockBackend([], default_rate=0.5, replay_noise=0.2)
    q = {e.seed: e.success for e in quiet.run(NOMINAL, CODE, POLICY, SEEDS, Mode.CLEAN)}
    first = {e.seed: e.success for e in noisy.run(NOMINAL, CODE, POLICY, SEEDS, Mode.CLEAN)}
    assert first == q                                  # first run unaffected
    again = {e.seed: e.success for e in noisy.run(NOMINAL, CODE, POLICY, SEEDS, Mode.CLEAN)}
    assert agreement_rate(first, again) == pytest.approx(0.8, abs=0.035)


def test_mock_counts_calls_per_configuration():
    be = _table()
    be.run(NOMINAL, CODE, POLICY, range(10), Mode.CLEAN)
    be.run(NOMINAL, CODE, POLICY, range(5), Mode.CLEAN)
    assert be.calls[MockBackend.config_key(NOMINAL, CODE, POLICY, Mode.CLEAN)] == 15


def test_mock_rejects_bad_rates():
    with pytest.raises(ValueError):
        MockRule(rate=1.5)
    with pytest.raises(ValueError):
        MockBackend([], default_rate=-0.1)


# ----------------------------------------------------------------- cache ----

def test_cache_runs_only_missing_seeds_and_preserves_order():
    be = _table()
    cb = CachedBackend(be, RolloutCache())
    first = cb.run(DARK, CODE, POLICY, [0, 1, 2, 3, 4], Mode.CLEAN)
    assert (cb.hits, cb.misses) == (0, 5)
    second = cb.run(DARK, CODE, POLICY, [4, 2, 7, 0, 9], Mode.CLEAN)
    assert (cb.hits, cb.misses) == (3, 7)
    assert [e.seed for e in second] == [4, 2, 7, 0, 9]
    assert be.calls[MockBackend.config_key(DARK, CODE, POLICY, Mode.CLEAN)] == 7
    by_seed = {e.seed: e for e in first}
    assert second[0] == by_seed[4] and second[3] == by_seed[0]


@pytest.mark.parametrize("other", [
    dict(scene=DARK_GLOSSY),
    dict(code=CODE_BUGGY),                      # eval config differs, same commit
    dict(actor=REF_RAW),                        # reference controller vs policy
    dict(actor=ActorRef(ActorKind.POLICY, "ckpt-2")),   # different checkpoint
    dict(mode=Mode.INCIDENT),
])
def test_cache_key_separates_every_component(other):
    base = dict(scene=DARK, code=CODE, actor=POLICY, mode=Mode.CLEAN)
    cb = CachedBackend(_table(), RolloutCache())
    cb.run(base["scene"], base["code"], base["actor"], [0, 1, 2], base["mode"])
    alt = {**base, **other}
    cb.run(alt["scene"], alt["code"], alt["actor"], [0, 1, 2], alt["mode"])
    assert cb.misses == 6 and cb.hits == 0
    assert len(cb.cache) == 6


def test_reference_raw_and_wrapper_do_not_collide():
    cb = CachedBackend(_table(), RolloutCache())
    raw = cb.run(NOMINAL, CODE, REF_RAW, range(50), Mode.CLEAN)
    wrap = cb.run(NOMINAL, CODE, REF_WRAP, range(50), Mode.CLEAN)
    assert _rate(raw) == 1.0
    assert cb.hits == 0
    assert _rate(wrap) < 1.0          # default 0.85 for the wrapper path in this table


def test_cache_persists_to_jsonl_and_reloads(tmp_path):
    path = str(tmp_path / "rollouts.jsonl")
    be1 = _table()
    cb1 = CachedBackend(be1, RolloutCache(path))
    out1 = cb1.run(DARK, CODE, POLICY, range(20), Mode.CLEAN)

    be2 = _table()
    cb2 = CachedBackend(be2, RolloutCache(path))
    assert len(cb2.cache) == 20
    out2 = cb2.run(DARK, CODE, POLICY, range(20), Mode.CLEAN)
    assert [e.to_dict() for e in out2] == [e.to_dict() for e in out1]
    assert sum(be2.calls.values()) == 0 and cb2.hits == 20


def test_cache_first_put_wins_and_checks_seed():
    cache = RolloutCache()
    key = CacheKey.make(NOMINAL, CODE, POLICY, 3, Mode.CLEAN)
    assert cache.put(key, EpisodeResult(3, True))
    assert not cache.put(key, EpisodeResult(3, False))
    assert cache.get(key).success is True
    with pytest.raises(ValueError):
        cache.put(key, EpisodeResult(4, True))


def test_cached_backend_hides_noise_and_raw_backend_exposes_it():
    noisy = MockBackend([], default_rate=0.5, replay_noise=0.5)
    cb = CachedBackend(noisy, RolloutCache())
    a = {e.seed: e.success for e in cb.run(NOMINAL, CODE, POLICY, SEEDS, Mode.CLEAN)}
    b = {e.seed: e.success for e in cb.run(NOMINAL, CODE, POLICY, SEEDS, Mode.CLEAN)}
    assert agreement_rate(a, b) == 1.0                  # cache reuses episodes
    c = {e.seed: e.success for e in cb.backend.run(NOMINAL, CODE, POLICY, SEEDS, Mode.CLEAN)}
    assert agreement_rate(a, c) < 0.6                   # agreement measurement must bypass the cache


def test_cached_backend_rejects_duplicate_seeds():
    with pytest.raises(ValueError):
        CachedBackend(_table(), RolloutCache()).run(NOMINAL, CODE, POLICY, [1, 1], Mode.CLEAN)


# ------------------------------------------------- backend id (ISSUES #19) ----

class _Named(MockBackend):
    def __init__(self, backend_id, rate):
        super().__init__([], default_rate=rate)
        self.backend_id = backend_id


def test_cache_key_includes_the_backend_so_two_backends_never_share_episodes(tmp_path):
    path = str(tmp_path / "shared.jsonl")
    always = CachedBackend(_Named("sim-a", 1.0), RolloutCache(path))
    assert all(e.success for e in always.run(NOMINAL, CODE, POLICY, range(20), Mode.CLEAN))
    never = CachedBackend(_Named("sim-b", 0.0), RolloutCache(path))      # same file, other backend
    out = never.run(NOMINAL, CODE, POLICY, range(20), Mode.CLEAN)
    assert never.hits == 0 and not any(e.success for e in out)
    again = CachedBackend(_Named("sim-a", 0.0), RolloutCache(path))      # same id: served from the cache
    assert all(e.success for e in again.run(NOMINAL, CODE, POLICY, range(20), Mode.CLEAN)) and again.hits == 20


def test_cache_lines_from_before_the_backend_id_still_load_and_match(tmp_path):
    import json
    path = tmp_path / "old.jsonl"
    key = {"code": CODE.key(), "actor": POLICY.key(), "scene": NOMINAL.key(), "seed": 3, "mode": "clean"}
    path.write_text(json.dumps({"key": key, "episode": EpisodeResult(3, True).to_dict()}) + "\n")
    cache = RolloutCache(str(path))
    assert CacheKey.make(NOMINAL, CODE, POLICY, 3, Mode.CLEAN) in cache
    assert CacheKey.make(NOMINAL, CODE, POLICY, 3, Mode.CLEAN, "isaac:x") not in cache
    cb = CachedBackend(_table(), cache)                                   # MockBackend: backend_id ""
    assert cb.run(NOMINAL, CODE, POLICY, [3], Mode.CLEAN)[0].success is True and cb.hits == 1
    assert "backend" not in CacheKey.make(NOMINAL, CODE, POLICY, 3, Mode.CLEAN).to_dict()
    assert CacheKey.from_dict(CacheKey.make(NOMINAL, CODE, POLICY, 3, Mode.CLEAN, "b").to_dict()).backend == "b"
