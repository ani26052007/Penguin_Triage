"""GR1 hand-closure scalar: how far one hand has closed, 0 open to 1 closed.

GR1's hands have no single gripper joint, so "gripper width" and "the moment the gripper closes" need
a scalar. The default (``closure``) is the mean normalised closure of the finger joints listed in the
task config (``traces.hand_closure.hands``): each joint ``q`` maps to ``(q - open) / (closed - open)``,
clipped to [0, 1]. The router's gripper width is the opposite direction (wide = open), so
``opening = 1 - closure`` is what goes into ``gripper_cmd_width`` / ``gripper_width``.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def closure(rows: Sequence[Sequence[float]], joint_order: Sequence[str],
            joints: Mapping[str, tuple[float, float]]) -> list[float]:
    """Per row, the mean normalised closure of ``joints`` (name -> (open, closed)); ``joint_order``
    names the columns of ``rows``."""
    missing = [j for j in joints if j not in joint_order]
    if missing:
        raise ValueError(f"hand-closure joints {missing} are not in the joint order")
    cols = [(list(joint_order).index(j), lo, hi) for j, (lo, hi) in joints.items()]
    out = []
    for row in rows:
        vals = [min(1.0, max(0.0, (float(row[i]) - lo) / (hi - lo))) for i, lo, hi in cols]
        out.append(sum(vals) / len(vals))
    return out


def opening(rows: Sequence[Sequence[float]], joint_order: Sequence[str],
            joints: Mapping[str, tuple[float, float]]) -> list[float]:
    """``1 - closure``: the router's gripper width, normalised (1 open, 0 closed)."""
    return [1.0 - c for c in closure(rows, joint_order, joints)]
