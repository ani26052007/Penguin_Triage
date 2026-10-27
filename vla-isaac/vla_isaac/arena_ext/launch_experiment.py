"""``python -m vla_isaac.arena_ext.launch_experiment <experiment-runner args>``: register the
actuator-override environment, then hand over to Arena's experiment runner with the same argv.
Isaac's Python on the GPU box only (see ``actuator_env``)."""

from __future__ import annotations


def main() -> None:
    from vla_isaac.arena_ext.actuator_env import register_actuated_environment

    register_actuated_environment()
    # VERIFY(V-ext-env): Arena's experiment_runner.main() reads sys.argv itself
    # (A:isaaclab_arena/evaluation/experiment_runner.py:98-99).
    from isaaclab_arena.evaluation.experiment_runner import main as arena_main

    arena_main()


if __name__ == "__main__":
    main()
