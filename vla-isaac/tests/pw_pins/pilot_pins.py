"""The pilot pins of policy-wrapper: exact values and exact behaviour of the clean base on the repo's own
``config/eval.yaml``. They were policy-wrapper tests before Phase 8 and moved here, outside the agent's
container (docs/IMPL_PLAN.md, "Decisions during Phase 8"): inside the repo they would catch the injected
bugs that are meant to be silent.

Not collected by vla-isaac's pytest. ``tests/test_policy_wrapper_pins.py`` copies this file into a copy of
the checkout as ``tests/test_pilot_pins.py`` and runs it there, so it imports the checkout's own test helpers.
"""

from __future__ import annotations

import numpy as np
import torch

from policy_wrapper.eval_config import read_eval_config, repo_root
from policy_wrapper.policy.gr00t_remote_closedloop_policy import (
    Gr00tRemoteClosedloopPolicy,
    Gr00tRemoteClosedloopPolicyCfg,
)
from policy_wrapper.serve import ServerSection, load_server_section
from tests.helpers import EVAL_YAML, REPO, ServerThread
from tests.test_wire import make_policy, observation, requests

NUM_ENVS = 2
CHUNK = 16
CAMERA, VIDEO_KEY = "robot_pov_cam_rgb", "ego_view"

PILOT = {
    "client": {
        "language_instruction": "Place the sauce bottle on the top shelf of the fridge, and close the fridge door.",
        "embodiment_tag": "GR1", "action_horizon": 16, "action_chunk_length": 16, "scheduler": "chunk",
        "modality_config_path": "policy_wrapper/embodiments/gr1/gr1_arms_only_data_config.py",
        "policy_joints_config_path": "policy_wrapper/embodiments/gr1/gr00t_26dof_joint_space.yaml",
        "action_joints_config_path": "policy_wrapper/embodiments/gr1/36dof_joint_space.yaml",
        "state_joints_config_path": "policy_wrapper/embodiments/gr1/54dof_joint_space.yaml",
        "task_mode_name": "gr1_tabletop_manipulation", "pov_cam_name_sim": "robot_pov_cam_rgb",
        "original_image_size": [512, 512, 3], "target_image_size": [512, 512, 3], "bgr_conversion": False},
    "server": {
        "checkpoint": "nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task/ranch_bottle_into_fridge",
        "checkpoint_revision": "TODO_PIN", "embodiment_tag": "GR1", "denoising_steps": 4,
        "seed_per_request": True, "norm_stats_override": None},
    "env": {
        "task": "put_item_in_fridge_and_close_door", "embodiment": "gr1_joint", "object": "ranch_dressing_hope_robolab",
        "control_hz": 50, "actuator_overrides": {}, "render": {}},
}


def repo_client(port: int) -> Gr00tRemoteClosedloopPolicy:
    policy = Gr00tRemoteClosedloopPolicy(Gr00tRemoteClosedloopPolicyCfg(
        policy_config_yaml_path=str(EVAL_YAML), policy_device="cpu", num_envs=NUM_ENVS,
        remote_host="127.0.0.1", remote_port=port))
    policy.set_task_description(None)
    return policy


def test_the_eval_config_holds_exactly_the_pilot_values():
    assert read_eval_config(EVAL_YAML) == PILOT
    assert repo_root(EVAL_YAML) == REPO


def test_the_server_section_is_the_pilot():
    assert load_server_section(EVAL_YAML) == ServerSection(
        "nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task/ranch_bottle_into_fridge", "TODO_PIN", "GR1", 4, True, None)


def test_the_image_on_the_wire_is_the_env_frame_unchanged():
    with ServerThread(make_policy) as server:
        policy = repo_client(server.port)
        obs = observation()
        policy.get_action(None, obs)
        policy.close()
        sent = server.server.policy.policy.observations[0]
    np.testing.assert_array_equal(sent["video"][VIDEO_KEY][:, 0], obs["camera_obs"][CAMERA].numpy())


def test_the_scheduler_replays_one_chunk_step_by_step_then_fetches_the_next():
    from gr00t.policy.gr00t_policy import fixed_chunk

    with ServerThread(make_policy) as server:
        policy = repo_client(server.port)
        actions = [policy.get_action(None, observation(i)) for i in range(CHUNK + 1)]
        policy.close()
        assert len(requests(server, "get_action")) == 2
    chunk = fixed_chunk(NUM_ENVS)
    # action index 0 is left_shoulder_pitch_joint (policy left_arm[0]); 1 is right_shoulder_pitch_joint
    for t in range(CHUNK):
        np.testing.assert_allclose(actions[t][:, 0].numpy(), chunk["left_arm"][:, t, 0])
        np.testing.assert_allclose(actions[t][:, 1].numpy(), chunk["right_arm"][:, t, 0])
    np.testing.assert_allclose(actions[CHUNK][:, 0].numpy(), chunk["left_arm"][:, 0, 0])   # a new chunk


def test_step_pairs_follow_the_chunk_with_no_extra_delay_and_policy_input_is_the_render(tmp_path, monkeypatch):
    import json

    from policy_wrapper import instrument as I

    monkeypatch.setenv(I.ENV_DIR, str(tmp_path))
    with ServerThread(make_policy) as server:
        policy = repo_client(server.port)
        for t in range(2 * CHUNK + 4):
            policy.get_action(None, observation(t))
        policy.close()
    ep = json.loads((tmp_path / "env0000_ep00000.json").read_text())
    pairs = ep["step_pairs"]
    assert pairs[:CHUNK + 1] == [[0, t, t] for t in range(CHUNK)] + [[CHUNK, CHUNK, 0]]
    assert {a - o - k for o, a, k in pairs} == {0} and len(pairs) == 2 * CHUNK + 4
    assert all(f["policy_input"] == f["render"] for f in ep["frames"])   # same size, no conversion: identical


def test_the_hand_actions_are_the_policy_hand_outputs_in_the_action_joint_order():
    """Every action column is the policy output of the joint with that name (the action YAML's order)."""
    from gr00t.policy.gr00t_policy import fixed_chunk

    import yaml

    client = PILOT["client"]
    action_idx = yaml.safe_load((REPO / client["action_joints_config_path"]).read_text())["joints"]
    groups = yaml.safe_load((REPO / client["policy_joints_config_path"]).read_text())["joints"]
    with ServerThread(make_policy) as server:
        policy = repo_client(server.port)
        actions = [policy.get_action(None, observation(i)) for i in range(CHUNK)]
        policy.close()
    chunk = fixed_chunk(NUM_ENVS)
    for t in (0, 5, CHUNK - 1):
        for group, names in groups.items():
            for j, name in enumerate(names):
                if name in action_idx:
                    np.testing.assert_allclose(actions[t][:, action_idx[name]].numpy(), chunk[group][:, t, j])
    assert all(isinstance(a, torch.Tensor) for a in actions)
