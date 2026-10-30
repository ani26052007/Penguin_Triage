"""``transform_statistics`` on a multi-embodiment statistics file (no policy-wrapper checkout needed).

The real checkpoint file holds ``gr1`` and embodiments without ``left_arm`` / ``right_arm`` action groups
(``behavior_r1_pro``, ``robocasa_panda_omron``). The 2026-10-29 session injected BUG-07 from a gr1-only copy
because the tool raised on the others.
"""

from __future__ import annotations

import pytest

from tests.bugs_support import inject

SWAP = {"transform": "swap_arm_groups"}


def group(v: float) -> dict:
    return {"min": [v], "max": [v + 1.0], "mean": [v + 0.5]}


def stats() -> dict:
    return {
        "gr1": {"action": {"left_arm": group(0.0), "right_arm": group(10.0)}, "state": {}},
        "behavior_r1_pro": {"action": {"base": group(5.0), "torso": group(6.0)}, "state": {}},
        "robocasa_panda_omron": {"action": {"end_effector_position": group(7.0)}, "state": {}},
        "_comment": "dropped",
    }


def test_embodiments_without_arm_groups_are_copied_unchanged_and_the_others_swapped():
    src = stats()
    out = inject.transform_statistics(src, SWAP)
    assert "_comment" not in out
    assert out["gr1"]["action"]["left_arm"] == group(10.0) and out["gr1"]["action"]["right_arm"] == group(0.0)
    assert out["behavior_r1_pro"] == src["behavior_r1_pro"]
    assert out["robocasa_panda_omron"] == src["robocasa_panda_omron"]


def test_the_input_is_not_modified():
    src = stats()
    inject.transform_statistics(src, SWAP)
    assert src["gr1"]["action"]["left_arm"] == group(0.0)


def test_an_embodiment_flag_keeps_only_that_embodiment():
    out = inject.transform_statistics(stats(), SWAP, embodiment="gr1")
    assert list(out) == ["gr1"] and out["gr1"]["action"]["left_arm"] == group(10.0)


def test_a_named_embodiment_without_arm_groups_is_refused():
    with pytest.raises(inject.InjectError, match="no left_arm / right_arm"):
        inject.transform_statistics(stats(), SWAP, embodiment="behavior_r1_pro")
    with pytest.raises(inject.InjectError, match="no embodiment 'nope'"):
        inject.transform_statistics(stats(), SWAP, embodiment="nope")


def test_a_file_with_no_arm_groups_at_all_is_refused():
    only_others = {k: v for k, v in stats().items() if k != "gr1"}
    with pytest.raises(inject.InjectError, match="no embodiment in the statistics"):
        inject.transform_statistics(only_others, SWAP)


def test_the_shift_transform_also_skips_embodiments_without_arm_groups():
    out = inject.transform_statistics(stats(), {"transform": "shift_arm_location", "fraction": 0.5})
    assert out["gr1"]["action"]["left_arm"]["min"] == [0.5]
    assert out["behavior_r1_pro"] == stats()["behavior_r1_pro"]
