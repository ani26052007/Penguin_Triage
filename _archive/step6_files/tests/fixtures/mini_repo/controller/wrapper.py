"""Policy wrapper: observation in, action out (fixture repo for the report tests)."""

from controller import frames

GRIPPER_THRESHOLD = 0.5


class PolicyWrapper:
    def __init__(self, policy):
        self.policy = policy

    def to_policy_input(self, obs):
        return {k: frames.normalize_image(v) for k, v in obs.items()}

    def act(self, obs):
        raw = self.policy(self.to_policy_input(obs))
        return self.postprocess(raw)

    def postprocess(self, raw):
        return raw


def gripper_command(raw):
    return 1.0 if raw > GRIPPER_THRESHOLD else 0.0
