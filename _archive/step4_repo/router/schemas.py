"""Shared data records for the router (plan.md 3.2.2, 3.2.3).

Everything here is plain data: frozen dataclasses that round-trip through JSON
via ``to_dict`` / ``from_dict``. No routing logic lives in this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def canonical_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, no whitespace) used for hashing keys."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def short_hash(obj: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:length]


# --------------------------------------------------------------------------- #
# enums
# --------------------------------------------------------------------------- #

class Mode(str, Enum):
    """Inference mode (3.2.2). Incident = how the eval was run; clean = serial."""
    CLEAN = "clean"
    INCIDENT = "incident"


class ActorKind(str, Enum):
    POLICY = "policy"          # the VLA under test
    REFERENCE = "reference"    # scripted / planner reference controller (5.2 B)


class ActionPipeline(str, Enum):
    """Where the actor's actions go.

    WRAPPER: through the current wrapper's action pipeline and controller (the
             normal path; G3 for the reference controller).
    RAW:     straight to the simulator, bypassing the wrapper (G2 / C1).
    """
    WRAPPER = "wrapper"
    RAW = "raw"


class Outcome(str, Enum):
    """Result of one replay against the two anchors (3.2.9).

    INCONCLUSIVE is not a fourth verdict: it means "the interval straddles a
    boundary, extend the seeds before deciding".
    """
    RESTORED = "restored"
    PARTIAL = "partial"
    NO_EFFECT = "no_effect"
    INCONCLUSIVE = "inconclusive"


class VerdictLabel(str, Enum):
    """One label per row of the decision table (3.2.8)."""
    NO_SIGNIFICANT_FAILURE = "no_significant_failure"   # row 1
    SIM_ASSET = "sim_asset"                             # row 2
    INFRA = "infra"                                     # row 3
    CODE = "code"                                       # row 4
    RENDERING_MISMATCH = "rendering_mismatch"           # row 5
    VISION_FACTOR = "vision_factor"                     # row 6
    VISION_INTERACTING = "vision_interacting"           # row 7
    ACTION_GAP = "action_gap"                           # row 8
    PERCEPTION_UNRESOLVED = "perception_unresolved"     # row 9
    UNDETERMINED = "undetermined"                       # row 10


class Destination(str, Enum):
    STOP = "stop"
    HAGENT = "hagent"
    CONFIG_FIX = "config_fix"
    DATA_LADDER = "data_ladder"
    REPORT = "report"


# --------------------------------------------------------------------------- #
# scene spec
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Factor:
    """One named change from nominal, e.g. Factor("material(mug)", "glossy_02")."""
    name: str
    value: str

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Factor":
        return cls(name=d["name"], value=d["value"])


@dataclass(frozen=True)
class SceneSpec:
    """A scene is a nominal scene plus a list of named factor changes (3.2.2).

    Factors are stored sorted by name so two specs with the same changes in a
    different order compare and hash equal.
    """
    nominal: str
    factors: tuple[Factor, ...] = ()

    def __post_init__(self) -> None:
        facs = tuple(sorted(self.factors, key=lambda f: f.name))
        names = [f.name for f in facs]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate factor names in scene spec: {names}")
        if not self.nominal:
            raise ValueError("scene spec needs a nominal scene id")
        object.__setattr__(self, "factors", facs)

    # -- queries -------------------------------------------------------------
    @property
    def is_nominal(self) -> bool:
        return not self.factors

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.factors)

    def factor(self, name: str) -> Optional[str]:
        for f in self.factors:
            if f.name == name:
                return f.value
        return None

    # -- derived specs (pure; used later by one-factor restores) -------------
    def restore(self, name: str) -> "SceneSpec":
        if name not in self.factor_names:
            raise KeyError(f"factor {name!r} not in scene spec")
        return SceneSpec(self.nominal, tuple(f for f in self.factors if f.name != name))

    def restore_all(self) -> "SceneSpec":
        return SceneSpec(self.nominal, ())

    def with_factor(self, name: str, value: str) -> "SceneSpec":
        rest = tuple(f for f in self.factors if f.name != name)
        return SceneSpec(self.nominal, rest + (Factor(name, value),))

    def key(self) -> str:
        """Canonical, human-readable key, e.g. 'pick_mug|lighting=dark;material(mug)=glossy'."""
        tail = ";".join(f"{f.name}={f.value}" for f in self.factors)
        return f"{self.nominal}|{tail}"

    # -- json ----------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"nominal": self.nominal, "factors": [f.to_dict() for f in self.factors]}

    @classmethod
    def from_dict(cls, d: Mapping) -> "SceneSpec":
        return cls(nominal=d["nominal"], factors=tuple(Factor.from_dict(f) for f in d.get("factors", [])))


# --------------------------------------------------------------------------- #
# training spec
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrainingSpec:
    """Contract extracted once from the policy's training/data config (3.2.2).

    Every field is optional: the risk list (Section 9) notes the training
    config may not expose all of these, and a missing field just means the
    matching contract check falls back to weaker evidence.

    TODO(verify): the real source keys for each field depend on the chosen
    policy (e.g. GR00T modality config); extraction is build-order step 1.
    """
    policy_name: str
    image_size: Optional[tuple[int, int]] = None            # (height, width)
    channel_order: Optional[str] = None                     # e.g. "RGB"
    norm_stats: Optional[Mapping[str, Mapping[str, Sequence[float]]]] = None
    camera_poses: Optional[Mapping[str, Sequence[float]]] = None   # name -> pose
    action_horizon: Optional[int] = None
    replan_steps: Optional[int] = None
    denoising_steps: Optional[int] = None
    control_hz: Optional[float] = None
    render_settings: Optional[Mapping[str, Any]] = None
    state_layout: Optional[Mapping[str, Sequence[int]]] = None     # name -> [start, end)
    action_layout: Optional[Mapping[str, Sequence[int]]] = None
    instruction_templates: Optional[Mapping[str, str]] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image_size is not None:
            object.__setattr__(self, "image_size", tuple(self.image_size))

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return json.loads(canonical_json(out))   # deep-copies nested mappings to plain json

    @classmethod
    def from_dict(cls, d: Mapping) -> "TrainingSpec":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown TrainingSpec fields: {sorted(unknown)}")
        return cls(**dict(d))


# --------------------------------------------------------------------------- #
# what was run
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CodeRef:
    """The code under test plus its eval config.

    The cache key's "code hash" (3.2.2) is ``key()``, which hashes the code
    hash together with the eval config, so a render-settings or chunk-length
    change does not collide with the unchanged config in the cache.
    """
    code_hash: str
    eval_config: Mapping[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.code_hash}+cfg:{short_hash(dict(self.eval_config))}"

    def to_dict(self) -> dict:
        return {"code_hash": self.code_hash, "eval_config": json.loads(canonical_json(dict(self.eval_config)))}

    @classmethod
    def from_dict(cls, d: Mapping) -> "CodeRef":
        return cls(code_hash=d["code_hash"], eval_config=d.get("eval_config", {}))


@dataclass(frozen=True)
class ActorRef:
    """Who produces the actions, and where they go.

    checkpoint: policy checkpoint id, or the reference controller's version.
    The policy always runs through the wrapper; only the reference controller
    may run RAW (G2 / C1) or through the WRAPPER (G3).
    """
    kind: ActorKind
    checkpoint: str
    pipeline: ActionPipeline = ActionPipeline.WRAPPER

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ActorKind(self.kind))
        object.__setattr__(self, "pipeline", ActionPipeline(self.pipeline))
        if not self.checkpoint:
            raise ValueError("actor needs a checkpoint / version id")
        if self.kind is ActorKind.POLICY and self.pipeline is ActionPipeline.RAW:
            raise ValueError("the policy always runs through the wrapper; RAW is for the reference controller")

    def key(self) -> str:
        return f"{self.kind.value}:{self.checkpoint}:{self.pipeline.value}"

    def to_dict(self) -> dict:
        return {"kind": self.kind.value, "checkpoint": self.checkpoint, "pipeline": self.pipeline.value}

    @classmethod
    def from_dict(cls, d: Mapping) -> "ActorRef":
        return cls(kind=ActorKind(d["kind"]), checkpoint=d["checkpoint"],
                   pipeline=ActionPipeline(d.get("pipeline", "wrapper")))


@dataclass(frozen=True)
class EpisodeResult:
    """One episode. ``logs`` holds the rollout instrumentation (3.2.2).

    The instrumentation schema (per-frame state, policy-input tensors,
    obs-step/applied-step pairs, gripper widths, subtask progress, EE pose at
    grasp) is defined with the real backend in build-order step 2; here it is
    an opaque JSON mapping, or a reference to where large logs are stored.
    """
    seed: int
    success: bool
    logs: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"seed": self.seed, "success": bool(self.success), "logs": json.loads(canonical_json(dict(self.logs)))}

    @classmethod
    def from_dict(cls, d: Mapping) -> "EpisodeResult":
        return cls(seed=int(d["seed"]), success=bool(d["success"]), logs=d.get("logs", {}))


@dataclass(frozen=True)
class ReplayResult:
    """One replay (e.g. Stage 0, C2, G3): what was run and how it went.

    ``outcome`` and the interval are filled in by the caller using
    ``ladder.stats``; they are optional so a raw run can be stored first.
    """
    replay_id: str
    scene: SceneSpec
    code_ref: CodeRef
    actor: ActorRef
    mode: Mode
    episodes: tuple[EpisodeResult, ...]
    outcome: Optional[Outcome] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", Mode(self.mode))
        object.__setattr__(self, "episodes", tuple(self.episodes))
        if self.outcome is not None:
            object.__setattr__(self, "outcome", Outcome(self.outcome))
        seeds = [e.seed for e in self.episodes]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds in replay {self.replay_id}")

    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def successes(self) -> int:
        return sum(1 for e in self.episodes if e.success)

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else float("nan")

    def by_seed(self) -> dict[int, bool]:
        return {e.seed: e.success for e in self.episodes}

    def to_dict(self) -> dict:
        return {
            "replay_id": self.replay_id,
            "scene": self.scene.to_dict(),
            "code_ref": self.code_ref.to_dict(),
            "actor": self.actor.to_dict(),
            "mode": self.mode.value,
            "episodes": [e.to_dict() for e in self.episodes],
            "outcome": self.outcome.value if self.outcome else None,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "ReplayResult":
        return cls(
            replay_id=d["replay_id"],
            scene=SceneSpec.from_dict(d["scene"]),
            code_ref=CodeRef.from_dict(d["code_ref"]),
            actor=ActorRef.from_dict(d["actor"]),
            mode=Mode(d["mode"]),
            episodes=tuple(EpisodeResult.from_dict(e) for e in d["episodes"]),
            outcome=Outcome(d["outcome"]) if d.get("outcome") else None,
            ci_low=d.get("ci_low"),
            ci_high=d.get("ci_high"),
        )


# --------------------------------------------------------------------------- #
# evidence, incident, verdict
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Evidence:
    """The single record every check emits (3.2.3)."""
    id: str
    source: str
    signal: str
    strength: float
    location: Optional[str] = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("evidence needs a stable id")
        if not (0.0 <= float(self.strength) <= 1.0):
            raise ValueError(f"evidence strength must be in [0, 1], got {self.strength}")

    def to_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "signal": self.signal,
                "strength": float(self.strength), "location": self.location, "detail": self.detail}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Evidence":
        return cls(id=d["id"], source=d["source"], signal=d["signal"], strength=float(d["strength"]),
                   location=d.get("location"), detail=d.get("detail", ""))


@dataclass(frozen=True)
class Incident:
    """A failing rollout handed to the router.

    Contains no clean commit and nothing from the bug ledger (3.2.1 #5, 3.3).
    ``reference_rate`` is the expected success for this condition from the
    failure map or the last validated run (Stage 0 compares against it).
    ``nominal_scene`` is the failure-map nominal that Stage 2 runs (3.2.5);
    when None it is ``SceneSpec(scene.nominal)``.
    """
    incident_id: str
    scene: SceneSpec
    training_spec: TrainingSpec
    code_ref: CodeRef
    policy: ActorRef
    failing_seeds: tuple[int, ...]
    observed_mode: Mode = Mode.INCIDENT
    observed_success_rate: Optional[float] = None
    reference_rate: Optional[float] = None
    log_refs: Mapping[str, str] = field(default_factory=dict)
    nominal_scene: Optional[SceneSpec] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "failing_seeds", tuple(int(s) for s in self.failing_seeds))
        object.__setattr__(self, "observed_mode", Mode(self.observed_mode))
        if self.policy.kind is not ActorKind.POLICY:
            raise ValueError("incident.policy must be a policy actor")

    def to_dict(self) -> dict:
        return {
            "incident_id": self.incident_id,
            "scene": self.scene.to_dict(),
            "training_spec": self.training_spec.to_dict(),
            "code_ref": self.code_ref.to_dict(),
            "policy": self.policy.to_dict(),
            "failing_seeds": list(self.failing_seeds),
            "observed_mode": self.observed_mode.value,
            "observed_success_rate": self.observed_success_rate,
            "reference_rate": self.reference_rate,
            "log_refs": dict(self.log_refs),
            "nominal_scene": self.nominal_scene.to_dict() if self.nominal_scene else None,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Incident":
        return cls(
            incident_id=d["incident_id"],
            scene=SceneSpec.from_dict(d["scene"]),
            training_spec=TrainingSpec.from_dict(d["training_spec"]),
            code_ref=CodeRef.from_dict(d["code_ref"]),
            policy=ActorRef.from_dict(d["policy"]),
            failing_seeds=tuple(d["failing_seeds"]),
            observed_mode=Mode(d.get("observed_mode", "incident")),
            observed_success_rate=d.get("observed_success_rate"),
            reference_rate=d.get("reference_rate"),
            log_refs=d.get("log_refs", {}),
            nominal_scene=SceneSpec.from_dict(d["nominal_scene"]) if d.get("nominal_scene") else None,
        )


@dataclass(frozen=True)
class Verdict:
    """Router output (3.2.8). The label -> destination mapping is router logic
    (a later step); this record only stores the result.

    by_elimination:      code verdict without a located cause (low confidence).
    likely_second_cause: the winning replay only partly restored success.
    multi_cause_flag:    Stage 2 came back "degraded".
    """
    label: VerdictLabel
    destination: Destination
    row: int
    evidence_ids: tuple[str, ...] = ()
    factor: Optional[str] = None
    location: Optional[str] = None
    by_elimination: bool = False
    likely_second_cause: bool = False
    multi_cause_flag: bool = False
    replays: tuple[ReplayResult, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", VerdictLabel(self.label))
        object.__setattr__(self, "destination", Destination(self.destination))
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        object.__setattr__(self, "replays", tuple(self.replays))
        if not (1 <= self.row <= 10):
            raise ValueError(f"decision-table row must be 1..10, got {self.row}")

    def to_dict(self) -> dict:
        return {
            "label": self.label.value,
            "destination": self.destination.value,
            "row": self.row,
            "evidence_ids": list(self.evidence_ids),
            "factor": self.factor,
            "location": self.location,
            "by_elimination": self.by_elimination,
            "likely_second_cause": self.likely_second_cause,
            "multi_cause_flag": self.multi_cause_flag,
            "replays": [r.to_dict() for r in self.replays],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> "Verdict":
        return cls(
            label=VerdictLabel(d["label"]),
            destination=Destination(d["destination"]),
            row=int(d["row"]),
            evidence_ids=tuple(d.get("evidence_ids", ())),
            factor=d.get("factor"),
            location=d.get("location"),
            by_elimination=bool(d.get("by_elimination", False)),
            likely_second_cause=bool(d.get("likely_second_cause", False)),
            multi_cause_flag=bool(d.get("multi_cause_flag", False)),
            replays=tuple(ReplayResult.from_dict(r) for r in d.get("replays", ())),
            notes=d.get("notes", ""),
        )
