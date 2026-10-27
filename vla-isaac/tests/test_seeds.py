from __future__ import annotations

import pytest

from vla_isaac.seeds import EpisodeSeed, join_seed, plan_seeds, split_seed


def test_split_and_join_round_trip():
    assert split_seed(30007, 10000) == EpisodeSeed(3, 7)
    assert join_seed(3, 7, 10000) == 30007
    assert all(join_seed(*vars(split_seed(s, 100)).values(), 100) == s for s in range(0, 1000, 37))


def test_plan_groups_by_build_and_keeps_the_callers_order():
    plan = plan_seeds([20003, 5, 10001, 2, 10000], 10000)
    assert list(plan.builds) == [0, 1, 2]
    assert plan.builds == {0: (2, 5), 1: (0, 1), 2: (3,)}
    assert [plan.episodes_needed(b) for b in plan.builds] == [6, 2, 4]
    assert plan.produced_seeds(1) == (10000, 10001)
    assert plan.in_caller_order({s: s * 10 for s in [2, 5, 10000, 10001, 20003, 99]}) == [200030, 50, 100010, 20, 100000]
    with pytest.raises(KeyError):
        plan.in_caller_order({5: 1})


@pytest.mark.parametrize("bad", [[1, 1], [], [-1], [1.5], [True]])
def test_bad_seed_lists_raise(bad):
    with pytest.raises(ValueError):
        plan_seeds(bad, 10)


def test_episode_index_out_of_range():
    with pytest.raises(ValueError):
        join_seed(0, 10, 10)
