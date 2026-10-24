"""Rollout backend interface and a table-driven mock (plan.md 3.2.14).

Everything simulator-related goes behind ``RolloutBackend``. The router, Rung 1
and the data pipeline only ever call ``run``.

Backend contract for code (Fix F2a): a ``CodeRef`` is a name, not a path. A
backend that needs the code's files gets them from
``ladder.workspace.WorkspaceManager``: ``view(code_ref)`` for a cached copy it
only reads, ``materialize(code_ref)`` for a fresh copy it may build or run in.
It never reads the original checkout directly; the name is the content (base
snapshot + cumulative diff), so equal names are equal files.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from router.schemas import ActorRef, CodeRef, EpisodeResult, Mode, SceneSpec


class RolloutBackend(ABC):
    """Runs episodes. One call = one (scene, code, actor, mode) over many seeds.
    The files of ``code_ref`` come from a ``WorkspaceManager`` (module docstring)."""

    @abstractmethod
    def run(self, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef,
            seeds: Sequence[int], mode: Mode) -> list[EpisodeResult]:
        """Return one EpisodeResult per seed, in the order of ``seeds``."""


class IsaacBackend(RolloutBackend):
    """Placeholder for the real Isaac Sim / Isaac Lab backend.

    Deliberately unimplemented: no Isaac, GR00T or policy-server API is
    assumed here. Items to settle in build-order steps 1-2:
      TODO(verify): how a SceneSpec factor maps onto Isaac Lab scene / material
                    / light / camera config (and that a material restore changes
                    only the visual material, 3.2.7).
      TODO(verify): how to load the policy checkpoint and run it headless.
      TODO(verify): how the reference controller's actions are injected RAW
                    (bypassing the wrapper) vs through the WRAPPER.
      TODO(verify): how clean mode (serial, one env, no queueing) vs incident
                    mode (parallel, shared policy server) are configured.
      TODO(verify): how per-episode seeding fixes both the simulator and the
                    policy's sampling seed (needed for replay agreement).
      TODO(verify): which instrumentation fields (3.2.2) can be logged and how.
      TODO(verify): run the controller / wrapper from
                    ``WorkspaceManager.materialize(code_ref)``.
    """

    def run(self, scene, code_ref, actor, seeds, mode):  # pragma: no cover
        raise NotImplementedError("IsaacBackend is not implemented; see TODO(verify) notes")


# --------------------------------------------------------------------------- #
# mock
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MockRule:
    """One row of the mock's success-rate table. ``None`` fields match anything.

    scene / code / actor match on their ``key()``; ``mode`` matches exactly.
    ``factors_present`` matches any scene that contains all listed factor
    names (useful for "material(mug) present -> 20%" without listing every
    combination).

    ``files`` is a predicate on the code's files (e.g.
    ``ladder.workspace.FileContains``), evaluated by the backend on
    ``WorkspaceManager.view(code_ref)``: "code that normalises the inputs ->
    90%", whatever name a real repair agent's patch gets.
    """
    rate: float
    scene: Optional[SceneSpec] = None
    factors_present: tuple[str, ...] = ()
    code: Optional[CodeRef] = None
    actor: Optional[ActorRef] = None
    mode: Optional[Mode] = None
    files: Optional[Callable[..., bool]] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.rate <= 1.0):
            raise ValueError(f"rate must be in [0, 1], got {self.rate}")
        object.__setattr__(self, "factors_present", tuple(self.factors_present))

    def matches(self, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef, mode: Mode,
                files: Optional[Callable[[Callable], bool]] = None) -> bool:
        """``files(predicate)`` evaluates a file predicate on ``code_ref``'s files; the backend
        supplies it. A rule with ``files`` set cannot be matched without it."""
        if self.scene is not None and self.scene.key() != scene.key():
            return False
        if self.factors_present and not set(self.factors_present) <= set(scene.factor_names):
            return False
        if self.code is not None and self.code.key() != code_ref.key():
            return False
        if self.actor is not None and self.actor.key() != actor.key():
            return False
        if self.mode is not None and Mode(self.mode) is not mode:
            return False
        if self.files is not None:
            if files is None:
                raise ValueError("a MockRule with a files predicate needs a backend with workspaces")
            return files(self.files)
        return True


def _uniform(*parts: object) -> float:
    """Deterministic pseudo-uniform in [0, 1) from the given parts."""
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


class MockBackend(RolloutBackend):
    """Success rates come from a rule table; first matching rule wins.

    Determinism: an episode's outcome is a hash of (configuration, seed,
    salt), so the same seed in the same configuration gives the same result,
    i.e. replay agreement 1.0. ``replay_noise`` > 0 flips a repeated run of the
    same (configuration, seed) with that probability, to mimic non-deterministic
    GPU physics or sampled action heads (3.2.2). The first run of each seed is
    never flipped, so ``replay_noise`` only affects repeats.

    ``calls`` counts episodes actually simulated, per configuration key, so
    tests can check that the cache avoided re-running seeds.

    ``workspaces`` (a ``ladder.workspace.WorkspaceManager``) is needed only by
    rules with a ``files`` predicate; each (predicate, code name) pair is
    evaluated once, on ``workspaces.view(code_ref)``.
    """

    def __init__(self, rules: Sequence[MockRule], default_rate: float,
                 replay_noise: float = 0.0, salt: str = "mock", workspaces=None) -> None:
        if not (0.0 <= default_rate <= 1.0):
            raise ValueError("default_rate must be in [0, 1]")
        if not (0.0 <= replay_noise <= 1.0):
            raise ValueError("replay_noise must be in [0, 1]")
        self.rules = list(rules)
        self.default_rate = default_rate
        self.replay_noise = replay_noise
        self.salt = salt
        self.workspaces = workspaces
        self.calls: Counter = Counter()
        self._repeat: Counter = Counter()
        self._file_checks: dict[tuple[Callable, str], bool] = {}

    def _files(self, code_ref: CodeRef) -> Optional[Callable[[Callable], bool]]:
        if self.workspaces is None:
            return None

        def check(pred: Callable) -> bool:
            key = (pred, code_ref.code_hash)
            if key not in self._file_checks:
                self._file_checks[key] = bool(pred(self.workspaces.view(code_ref)))
            return self._file_checks[key]
        return check

    def rate_for(self, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef, mode: Mode) -> float:
        files = self._files(code_ref)
        for rule in self.rules:
            if rule.matches(scene, code_ref, actor, mode, files):
                return rule.rate
        return self.default_rate

    @staticmethod
    def config_key(scene: SceneSpec, code_ref: CodeRef, actor: ActorRef, mode: Mode) -> str:
        return f"{code_ref.key()}|{actor.key()}|{scene.key()}|{Mode(mode).value}"

    def run(self, scene: SceneSpec, code_ref: CodeRef, actor: ActorRef,
            seeds: Sequence[int], mode: Mode) -> list[EpisodeResult]:
        mode = Mode(mode)
        rate = self.rate_for(scene, code_ref, actor, mode)
        cfg = self.config_key(scene, code_ref, actor, mode)
        out = []
        for seed in seeds:
            seed = int(seed)
            success = _uniform(self.salt, cfg, seed) < rate
            repeat = self._repeat[(cfg, seed)]
            self._repeat[(cfg, seed)] += 1
            if repeat > 0 and self.replay_noise > 0:
                if _uniform(self.salt, "noise", cfg, seed, repeat) < self.replay_noise:
                    success = not success
            self.calls[cfg] += 1
            out.append(EpisodeResult(seed=seed, success=success,
                                     logs={"backend": "mock", "rate": rate, "repeat": repeat}))
        return out
