"""Phase 3 parts: the GR1 hand-closure scalar, offline reach / grasp events, the HDF5 reader over
synthetic h5py files, the instrument-file reader and the ``traces:`` layout checks."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import yaml

from router.prereq.behavior_trace import SubtaskProgress

from vla_isaac.config import ConfigError, load_task_config
from vla_isaac.traces.events import choose_hand, fired, grasp_event, reach_step, subtask_progress
from vla_isaac.traces.hand_closure import closure, opening
from vla_isaac.traces.hdf5_reader import H5pyStore, TrajectoryError, read_episodes
from vla_isaac.traces.instrument_reader import InstrumentError, read_episode
from vla_isaac.traces.layout import EventMatch, layout_from_dict

from tests.helpers import PILOT_TASK
from tests.trace_fixtures import CLOSE_AT, script, write_hdf5

LAYOUT = load_task_config(PILOT_TASK).traces
RIGHT = LAYOUT.hands["right"]


# ---- hand closure ---------------------------------------------------------------------------------

def joint_rows(values_by_joint: dict, order=LAYOUT.state_joints, n: int = 1) -> list[list[float]]:
    rows = [[0.0] * len(order) for _ in range(n)]
    for j, vs in values_by_joint.items():
        for t in range(n):
            rows[t][order.index(j)] = vs[t] if isinstance(vs, (list, tuple)) else vs
    return rows


def test_hand_closure_is_the_mean_normalised_closure_of_the_listed_finger_joints():
    joints = list(RIGHT)
    lo, hi = RIGHT[joints[0]]
    traj = [lo, (lo + hi) / 2, hi, hi + 1.0, lo - 1.0]                 # open, half, closed, past both limits
    rows = joint_rows({j: traj for j in joints}, n=5)
    assert closure(rows, LAYOUT.state_joints, RIGHT) == pytest.approx([0.0, 0.5, 1.0, 1.0, 0.0])
    assert opening(rows, LAYOUT.state_joints, RIGHT) == pytest.approx([1.0, 0.5, 0.0, 0.0, 1.0])
    one = joint_rows({joints[0]: hi})                                   # one of four fingers closed
    assert closure(one, LAYOUT.state_joints, RIGHT) == pytest.approx([0.25])
    assert closure(one, LAYOUT.state_joints, LAYOUT.hands["left"]) == pytest.approx([0.0])
    flipped = {j: (hi, lo) for j in joints}                             # a joint that closes towards lower values
    assert closure(rows, LAYOUT.state_joints, flipped)[:3] == pytest.approx([1.0, 0.5, 0.0])
    with pytest.raises(ValueError):
        closure(rows, ("a", "b"), RIGHT)


# ---- events ---------------------------------------------------------------------------------------

def _hands(s):
    return {"left": s["left"].tolist(), "right": s["right"].tolist()}


def _closures(s):
    idle = [0.0] * len(s["closure"])
    return {h: s["closure"].tolist() if h == s["hand"] else idle for h in ("left", "right")}


def test_reach_is_the_first_step_an_end_effector_is_within_the_distance():
    s = script("success")
    t = reach_step(_hands(s), s["object"].tolist(), LAYOUT.reach_distance_m)
    assert t == 14 and math.dist(s["right"][t], s["object"][t]) <= 0.1 < math.dist(s["right"][t - 1], s["object"][t - 1])
    r = script("reach")
    assert reach_step(_hands(r), r["object"].tolist(), LAYOUT.reach_distance_m) is None


def _grasp(s):
    return grasp_event(_closures(s), _hands(s), s["object"].tolist(), LAYOUT.grasp, LAYOUT.reach_distance_m)


def _choice(s):
    return choose_hand(_closures(s), _hands(s), s["object"].tolist(), LAYOUT.grasp, LAYOUT.reach_distance_m)


def test_grasp_needs_the_closure_crossing_and_the_object_moving_with_the_hand():
    s = script("success")
    g = _grasp(s)
    assert (g.step, g.hand) == (CLOSE_AT, "right") and g.record.ee_pos == pytest.approx(g.record.target_pos)
    still = script("grasp")                                             # closes, the object stays
    assert _grasp(still) is None
    short = {k: v[:CLOSE_AT + LAYOUT.grasp.window] if k != "hand" else v for k, v in s.items()}   # window past the end
    assert _grasp(short) is None
    away = dict(s, object=s["object"] + np.array([0.0, 0.0, 0.0]))
    away["right"] = s["right"] + np.linspace(0, 0.2, len(s["right"]))[:, None] * np.array([1.0, 0, 0])
    assert _grasp(away) is None


# ---- the grasping hand, per episode ---------------------------------------------------------------

@pytest.mark.parametrize("hand", ["left", "right"])
def test_the_grasping_hand_is_the_one_that_closes_within_reach_and_moves_the_object(hand):
    s = script("success", hand=hand)
    c = _choice(s)
    assert (c.hand, c.basis, c.grasp.hand, c.grasp.step) == (hand, "grasp", hand, CLOSE_AT)
    assert c.notes() == {"grasp_hand": hand, "width_hand": hand, "hand_basis": "grasp"}
    assert c.grasp.record.ee_pos == pytest.approx(tuple(s[hand][CLOSE_AT]))


def test_no_grasp_names_no_grasping_hand_and_takes_the_nearest_hand_for_the_width():
    r = script("reach")                                                 # the right hand never comes within reach
    c = _choice(r)
    assert c.grasp is None and c.grasp_hand is None and c.basis == "nearest" and c.hand == "right"
    assert c.notes() == {"grasp_hand": None, "width_hand": "right", "hand_basis": "nearest"}
    away = script("reach", hand="left")                                 # the left hand comes nearest, closes
    away["left"] = away["left"] - np.array([0.15, 0.0, 0.0])             # 0.15 m away: nearer, out of reach
    assert _choice(away).hand == "left" and _choice(away).grasp_hand is None


def test_a_closure_within_reach_that_does_not_move_the_object_is_an_attempt_with_that_hand():
    c = _choice(script("grasp", hand="left"))
    assert (c.hand, c.basis, c.grasp, c.grasp_hand) == ("left", "attempt", None, "left")


def test_a_hand_that_closes_out_of_reach_is_not_the_grasping_hand():
    s = script("success", hand="right")
    closures = _closures(s)
    closures["left"] = [0.0] * 10 + [1.0] * (len(s["closure"]) - 10)   # the left hand closes first, 0.4 m away
    c = choose_hand(closures, _hands(s), s["object"].tolist(), LAYOUT.grasp, LAYOUT.reach_distance_m)
    assert (c.hand, c.basis, c.grasp.step) == ("right", "grasp", CLOSE_AT)


def test_two_hands_crossing_at_the_same_step_go_to_the_nearer_hand():
    s = script("success", hand="right")
    hands = _hands(s)
    on = np.array(hands["right"])                                       # on the object from step 20
    hands["right"] = (on + np.array([0.0, -0.03, 0.0])).tolist()        # 0.03 m from the object
    for nearer, offset in (("right", 0.05), ("left", 0.01)):
        hands["left"] = (on + np.array([0.0, offset, 0.0])).tolist()    # both within reach, moving with the object
        closures = {h: s["closure"].tolist() for h in ("left", "right")}
        c = choose_hand(closures, hands, s["object"].tolist(), LAYOUT.grasp, LAYOUT.reach_distance_m)
        assert (c.hand, c.basis, c.grasp.step) == (nearer, "grasp", CLOSE_AT)


def test_events_mark_subtasks_and_the_phase_map_orders_them_as_router_phases():
    events = [{"step": 60, "criteria_name": "pick_and_place", "predicate_index": 1},
              {"step": 40, "criteria_name": "pick_and_place", "predicate_index": 1},
              {"step": 90, "criteria_name": "close_door", "predicate_index": 0}]
    assert fired(events, EventMatch("pick_and_place", 1)) == 40
    assert fired(events, EventMatch("pick_and_place", 2)) is None
    got = subtask_progress({"close_door": False, "place": True, "reach": True, "lift": True, "grasp": True},
                           LAYOUT.phase_map)
    assert got == (SubtaskProgress("reach", True), SubtaskProgress("grasp", True), SubtaskProgress("lift", True),
                   SubtaskProgress("place", True), SubtaskProgress("post_place", False))
    assert subtask_progress({"reach": True, "grasp": False}, LAYOUT.phase_map) == (
        SubtaskProgress("reach", True), SubtaskProgress("grasp", False))


# ---- HDF5 reader ----------------------------------------------------------------------------------

def test_the_reader_keys_demos_by_episode_and_lines_the_joints_up(tmp_path):
    path = tmp_path / "dataset_x_rebuild0.hdf5"
    s = script("success")
    write_hdf5(path, LAYOUT, [(0, 0, s)] + [(1, i, script("grasp")) for i in range(11)])
    store = H5pyStore(path, LAYOUT.hdf5.demo_group)
    assert store.demos()[:3] == ["demo_0", "demo_1", "demo_2"] and store.demos()[-1] == "demo_11"
    eps = read_episodes(store, LAYOUT)
    store.close()
    assert sorted(eps)[:2] == [(0, 0), (1, 0)] and len(eps) == 12
    a = eps[(0, 0)]
    assert len(a.joint_cmd) == len(a.joint_pos) == len(s["right"])
    assert len(a.joint_cmd[0]) == len(a.joint_pos[0]) == 36 and len(a.joint_pos_state[0]) == 54
    col_a, col_s = LAYOUT.action_joints.index("R_ring_proximal_joint"), LAYOUT.state_joints.index("R_ring_proximal_joint")
    assert [r[col_a] for r in a.joint_pos] == [r[col_s] for r in a.joint_pos_state]
    assert a.object_pos[0] == pytest.approx(tuple(s["object"][0])) and a.object_initial_pos == a.object_pos[0]
    assert a.nan_fraction == 0.0 and a.ee_pos == {}


def test_nan_steps_are_counted_and_missing_or_malformed_keys_raise(tmp_path):
    import h5py

    path = tmp_path / "d.hdf5"
    write_hdf5(path, LAYOUT, [(0, 0, script("success"))])
    with h5py.File(path, "a") as f:
        q = f[f"data/demo_0/{LAYOUT.hdf5.joint_pos}"]
        q[3:9, 0] = np.nan
    store = H5pyStore(path, "data")
    assert read_episodes(store, LAYOUT)[(0, 0)].nan_fraction == pytest.approx(6 / len(script("success")["right"]))
    store.close()
    with h5py.File(path, "a") as f:
        del f[f"data/demo_0/{LAYOUT.hdf5_path(LAYOUT.hdf5.object_pose)}"]
    with pytest.raises(TrajectoryError, match="root_pose"):
        read_episodes(H5pyStore(path, "data"), LAYOUT)
    dup = tmp_path / "dup.hdf5"
    write_hdf5(dup, LAYOUT, [(0, 0, script("success")), (0, 0, script("door"))])
    with pytest.raises(TrajectoryError, match="also in"):
        read_episodes(H5pyStore(dup, "data"), LAYOUT)


# ---- instrument reader ----------------------------------------------------------------------------

def _instrument_file(tmp_path, **edits):
    d = {"format": "policy_wrapper.instrument/1", "env_id": 0, "episode_in_env": 0, "start_index": 0, "num_steps": 3,
         "frames": [], "step_pairs": [[0, 0, 0], [0, 1, 1], [2, 2, 0]], "observations": {},
         "chunks": [{"obs_step": 0, "raw": {"a": [[1, 2], [3, 4]], "b": [[5], [6]]}},
                    {"obs_step": 2, "raw": {"a": [[7, 8], [9, 9]], "b": [[0], [0]]}}]}
    d.update(edits)
    p = tmp_path / "env0000_ep00000.json"
    p.write_text(json.dumps(d))
    return p


def test_raw_actions_are_the_chunk_rows_the_step_pairs_point_at(tmp_path):
    ep = read_episode(_instrument_file(tmp_path))
    assert ep.raw_actions == ((1.0, 2.0, 5.0), (3.0, 4.0, 6.0), (7.0, 8.0, 0.0)) and ep.action_keys == ("a", "b")
    with pytest.raises(InstrumentError, match="no chunk"):
        read_episode(_instrument_file(tmp_path, step_pairs=[[1, 1, 0]]))
    with pytest.raises(InstrumentError, match="applied before"):
        read_episode(_instrument_file(tmp_path, step_pairs=[[0, 0, 1]]))
    with pytest.raises(InstrumentError, match="format"):
        read_episode(_instrument_file(tmp_path, format="other/9"))


# ---- layout checks --------------------------------------------------------------------------------

def _traces() -> dict:
    return copy.deepcopy(yaml.safe_load(PILOT_TASK.read_text())["traces"])


@pytest.mark.parametrize("edit,match", [
    (lambda d: d["phase_map"].pop("close_door"), "does not map"),
    (lambda d: d["phase_map"].update(close_door="door"), "router phases"),
    (lambda d: d["phase_map"].update(close_door="place"), "two subtasks to one phase"),
    (lambda d: d["phase_map"].update(wave="post_place"), "unknown subtasks"),
    (lambda d: d["instrument"].update(obs_keys=["policy.left_eef_pos"]), "not in instrument.obs_keys"),
    (lambda d: d["hand_closure"]["hands"]["right"].update(R_nonexistent_joint=[0, 1]), "state and an action joint"),
    (lambda d: d["hand_closure"].update(width_hand="right"), "chosen per episode"),
    (lambda d: d["ee"].update(source="camera"), "'instrument' or 'hdf5'"),
    (lambda d: d["hdf5"].pop("joint_cmd"), "missing keys"),
    (lambda d: d["joints"].update(state="nope.yaml"), "no joint-space file"),
])
def test_bad_traces_sections_raise(edit, match):
    d = _traces()
    edit(d)
    with pytest.raises(ConfigError, match=match):
        layout_from_dict(d, PILOT_TASK.parent)


def test_the_pilot_layout_maps_the_door_to_post_place():
    assert LAYOUT.phase_map["close_door"] == "post_place" and LAYOUT.ee_source == "instrument"
    assert LAYOUT.hdf5_path(LAYOUT.hdf5.object_pose) == "states/rigid_object/ranch_dressing_hope_robolab/root_pose"
