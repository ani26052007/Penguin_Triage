"""Actuator overrides for policy runs (IMPL_PLAN D7). Runs inside Isaac's Python on the GPU box
only; the GPU-free tests never import this module.

Gains belong to the code under test (``env.actuator_overrides`` in policy-wrapper's
``config/eval.yaml``), but they live in Arena's embodiment code, which no Hydra variation reaches.
``IsaacBackend`` therefore runs a policy episode with actuator overrides through
``launch_experiment``, which registers a subclass of the task's environment factory that applies
them after the base factory builds the environment. The reference controller (Mimic, RAW) never
comes through here, so a gain bug cannot make G2 / C1 fail.

Inputs, set by ``IsaacBackend`` (``vla_isaac.runner.arena_cmd``):
  VLA_ISAAC_ACTUATOR_OVERRIDES  JSON ``{actuator_group: {field: value}}``
  VLA_ISAAC_BASE_ENV            ``module:Class`` of the task's environment factory
  VLA_ISAAC_ACTUATED_ENV_NAME   name to register the subclass under
"""

from __future__ import annotations

import importlib
import json
import os
from typing import Any, Mapping

ENV_OVERRIDES = "VLA_ISAAC_ACTUATOR_OVERRIDES"
ENV_BASE = "VLA_ISAAC_BASE_ENV"
ENV_NAME = "VLA_ISAAC_ACTUATED_ENV_NAME"


def apply_actuator_overrides(arena_env: Any, overrides: Mapping[str, Mapping[str, Any]]) -> None:
    """Set ``overrides`` on the robot's actuator configs, before the env config is compiled."""
    # VERIFY(V-actuator): the robot's ArticulationCfg sits at embodiment.scene_config.robot and its
    # ``actuators`` dict holds per-group cfgs with fields such as stiffness / damping (Isaac Lab, unread).
    robot = arena_env.embodiment.scene_config.robot
    for group, values in overrides.items():
        assert group in robot.actuators, f"no actuator group {group!r}; groups: {sorted(robot.actuators)}"
        actuator = robot.actuators[group]
        for name, value in values.items():
            assert hasattr(actuator, name), f"actuator group {group!r} has no field {name!r}"
            setattr(actuator, name, value)


def register_actuated_environment() -> type:
    module_path, class_name = os.environ[ENV_BASE].split(":", 1)
    base = getattr(importlib.import_module(module_path), class_name)
    overrides = json.loads(os.environ[ENV_OVERRIDES])

    def build(self, cfg):
        arena_env = base.build(self, cfg)
        apply_actuator_overrides(arena_env, overrides)
        return arena_env

    cls = type(f"Actuated{class_name}", (base,), {"name": os.environ[ENV_NAME], "build": build})
    from isaaclab_arena.assets.register import register_environment
    # VERIFY(V-ext-env): a dynamic subclass registers, the registry recovers its cfg type from the base,
    # and importing the base factory before SimulationApp starts is safe.
    return register_environment(cls)
