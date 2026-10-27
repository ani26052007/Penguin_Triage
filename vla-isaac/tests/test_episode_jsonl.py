from __future__ import annotations

import json

import pytest

from vla_isaac.results.episode_jsonl import EpisodeResultsError, index_records, read_records, to_episode_result


def write(tmp_path, recs, name="episode_results_rebuild0.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return p


def rec(env_id, ep, success=True, seed=0, **kw):
    return {"job_name": "j", "env_id": env_id, "episode_in_env": ep, "seed": seed, "success": success,
            "episode_length": 10, "language_instruction": "x", "timestamp": "t", **kw}


def test_clean_mode_indexes_by_episode_in_env(tmp_path):
    recs = read_records(write(tmp_path, [rec(0, 1, False), rec(0, 0, True)]))
    by_k = index_records(recs, num_envs=1, num_episodes=2, build_seed=0)
    assert [by_k[k].success for k in (0, 1)] == [True, False]


def test_incident_mode_ranks_by_episode_then_env(tmp_path):
    recs = read_records(write(tmp_path, [rec(1, 0), rec(0, 1), rec(0, 0), rec(1, 1)]))
    by_k = index_records(recs, num_envs=2, num_episodes=4, build_seed=0)
    assert [(by_k[k].env_id, by_k[k].episode_in_env) for k in range(4)] == [(0, 0), (1, 0), (0, 1), (1, 1)]


def test_extra_episodes_beyond_the_limit_are_ignored_and_missing_ones_raise(tmp_path):
    recs = read_records(write(tmp_path, [rec(0, k) for k in range(5)]))
    assert len(index_records(recs, num_envs=1, num_episodes=3, build_seed=0)) == 3
    with pytest.raises(EpisodeResultsError, match="missing"):
        index_records(recs, num_envs=1, num_episodes=6, build_seed=0)


@pytest.mark.parametrize("bad,match", [
    ([rec(0, 0, success=None)], "not a bool"),
    ([{"env_id": 0, "seed": 0, "success": True}], "missing"),
])
def test_malformed_records_raise(tmp_path, bad, match):
    with pytest.raises(EpisodeResultsError, match=match):
        read_records(write(tmp_path, bad))


def test_duplicate_keys_wrong_seed_and_duplicate_episodes_raise(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"env_id": 0, "env_id": 1, "episode_in_env": 0, "seed": 0, "success": true}\n')
    with pytest.raises(EpisodeResultsError, match="duplicate"):
        read_records(p)
    with pytest.raises(EpisodeResultsError, match="build seeds"):
        index_records(read_records(write(tmp_path, [rec(0, 0, seed=7)])), num_envs=1, num_episodes=1, build_seed=0)
    with pytest.raises(EpisodeResultsError, match="duplicate"):
        index_records(read_records(write(tmp_path, [rec(0, 0), rec(0, 0)])), num_envs=1, num_episodes=1, build_seed=0)


def test_episode_result_keeps_variations_placement_progress_and_paths(tmp_path):
    (r,) = read_records(write(tmp_path, [rec(0, 0, variations={"light.intensity": [100.0]}, placement={"o": 1},
                                             progress={"events": []})]))
    ep = to_episode_result(r, seed=42, episode_index=0, paired=True, extra_logs={"code_key": "k", "paths": {"x": "y"}})
    assert ep.seed == 42 and ep.success is True
    assert ep.logs["variations"] == {"light.intensity": [100.0]} and ep.logs["placement"] == {"o": 1}
    assert ep.logs["progress"] == {"events": []} and ep.logs["code_key"] == "k"
    assert ep.logs["paths"]["x"] == "y" and ep.logs["paths"]["episode_results_line"] == 1
