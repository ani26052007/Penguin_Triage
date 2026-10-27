"""The RAW reference controller: Isaac Lab Mimic ``generate_dataset.py`` on the scene (G2, C1).

``MimicReference.run(scene, actor, seeds, *, variations)`` is ``IsaacBackend``'s reference runner. It runs
Mimic from an annotated source dataset with ``--generation_num_trials len(seeds)`` on the task's Arena
environment (``--task``, the task config's env args such as ``--embodiment gr1_pink``), with the scene's
variation overrides appended (Arena's registration callback consumes ``<asset>.<variation>...=value``
arguments, ``A:isaaclab_arena/environments/isaaclab_interop.py:145-146``; whether ``generate_dataset.py``
passes them through is VERIFY(V-mimic-override)). It returns one ``EpisodeResult`` per requested seed.

Rules (IMPL_PLAN "Decisions from the Phase 3 review", point 4):

* **RAW only.** A WRAPPER request (G3) raises ``ReplayUnavailable`` (decision 3); a policy actor is a
  ``ValueError``.
* **Never the actuated launcher.** The command is Arena's stock registration callback with the Arena
  environment name; no ``--external_environment_class_path``, no ``vla_isaac.arena_ext``, no actuator
  override variables, and no working copy on ``PYTHONPATH``. Gains are code under test; the reference
  must not see them.
* **No downloads.** The annotated source dataset is a local path from the backend config's ``datasets``
  map (the task config names its id). Absent, the run is refused with ``MimicConfigError``
  (VERIFY(V-mimic-source)); it is never fetched.
* **Trials, not paired seeds.** Trial i of the run is the result for ``seeds[i]``, in trial order, with
  ``logs["paired"] = False``: a Mimic trial has nothing to do with the policy episode of the same seed.
  The outcomes are read from Mimic's log (``MimicOutput``, VERIFY(V-mimic-out)): one cumulative
  "successes/attempts" line per finished attempt gives each attempt's outcome in order. The pilot
  environment sets ``generation_guarantee: True`` (``A:isaaclab_arena/tasks/common/mimic_default_params.py``),
  so Mimic may make more attempts than ``generation_num_trials`` (it counts successes,
  VERIFY(V-mimic-guarantee)); the first ``len(seeds)`` attempts are used, a fixed-size sample, and the
  total is kept in the logs. Fewer attempts than seeds, or an attempt count the log cannot order, is a
  ``MimicOutputError``.

Qualification (ISSUES #5): ``qualify(results, rule)`` puts a Wilson interval (``ladder.stats``) on the
trial success rate; the reference is unqualified when the interval's **lower bound** is below the
configured threshold, and ``Qualification.evidence()`` is then the ``reference_unqualified`` evidence
(row 2) to pass to ``route(..., extra_evidence=...)``.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ladder.backend import ReplayUnavailable, wrapper_reference
from ladder.stats import wilson_interval
from router import vocab as V
from router.schemas import ActionPipeline, ActorKind, ActorRef, EpisodeResult, Evidence, SceneSpec, short_hash

from vla_isaac.config import BackendConfig, ConfigError, TaskConfig
from vla_isaac.runner.subprocess_runner import ArenaRunError, ArenaRunner, RunInvocation, SubprocessRunner
from vla_isaac.scene.factors import hydra_override

META_FILE = "vla_meta.json"
EVIDENCE_SOURCE = "reference_qualification"


class MimicConfigError(ConfigError):
    """The reference cannot be run as configured (no local source dataset, wrong actor)."""


class MimicOutputError(RuntimeError):
    """Mimic's log does not give one ordered outcome per requested trial."""


# --------------------------------------------------------------------------- #
# config (task config ``reference:``)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MimicOutput:
    """How Mimic's log reports progress. ``progress`` is a regex with named groups ``successes`` and
    ``attempts`` matching a cumulative count line; one such line after each attempt gives the outcomes
    in order. VERIFY(V-mimic-out): the pattern is unverified (Isaac Lab is not checked out)."""
    progress: str

    def __post_init__(self) -> None:
        rx = re.compile(self.progress)
        if {"successes", "attempts"} - set(rx.groupindex):
            raise ConfigError("reference.mimic.output.progress needs named groups 'successes' and 'attempts'")


@dataclass(frozen=True)
class MimicSection:
    script: str                              # generate_dataset.py, relative to the Arena root
    task: str                                # Arena environment name (--task)
    env_args: Mapping[str, str]              # environment flags, e.g. {object: ..., embodiment: gr1_pink}
    external_callback: str                   # Arena's stock registration callback
    source_dataset: str                      # id in the backend config's datasets map: the annotated HDF5
    device: str = "cpu"
    num_envs: int = 1                        # 1: attempts finish one at a time, so the log orders them
    enable_cameras: bool = True
    extra_args: tuple[str, ...] = ()
    output: MimicOutput = field(default_factory=lambda: MimicOutput(DEFAULT_PROGRESS))

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ConfigError("reference.mimic.num_envs must be >= 1")
        flags = " ".join(map(str, self.extra_args)) + " " + " ".join(self.env_args)
        if "external_environment_class_path" in flags or "vla_isaac" in self.external_callback:
            raise ConfigError("reference.mimic must use Arena's stock environment, never the actuated launcher")


@dataclass(frozen=True)
class QualificationRule:
    """Unqualified when the Wilson lower bound of the trial success rate is below ``threshold``."""
    threshold: float
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if not (0.0 < self.threshold <= 1.0 and 0.0 < self.confidence < 1.0):
            raise ConfigError("reference.qualification: need 0 < threshold <= 1 and 0 < confidence < 1")


@dataclass(frozen=True)
class ReferenceSection:
    actor_checkpoint: str                    # ActorRef.checkpoint the router uses for the reference
    mimic: MimicSection
    qualification: QualificationRule


# PLACEHOLDER (VERIFY(V-mimic-out)): a cumulative "<successes>/<attempts> (<pct>%) successful demos generated
# by mimic" line, as remembered from Isaac Lab Mimic's generation loop, not read from its code.
DEFAULT_PROGRESS = r"(?P<successes>\d+)/(?P<attempts>\d+) \([0-9.]+%\) successful demos generated by mimic"


def _strict(d: Any, cls, where: str, required: Sequence[str]) -> dict:
    if not isinstance(d, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown, missing = set(d) - known, set(required) - set(d)
    if unknown or missing:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
    return dict(d)


def reference_section_from_dict(d: Mapping) -> ReferenceSection:
    top = _strict(d, ReferenceSection, "task config [reference]", ("actor_checkpoint", "mimic", "qualification"))
    m = _strict(top["mimic"], MimicSection, "reference.mimic",
                ("script", "task", "env_args", "external_callback", "source_dataset"))
    if "output" in m:
        m["output"] = MimicOutput(**_strict(m["output"], MimicOutput, "reference.mimic.output", ("progress",)))
    m["env_args"] = {str(k): str(v) for k, v in m["env_args"].items()}
    m["extra_args"] = tuple(str(x) for x in m.get("extra_args", ()))
    q = _strict(top["qualification"], QualificationRule, "reference.qualification", ("threshold",))
    return ReferenceSection(str(top["actor_checkpoint"]), MimicSection(**m),
                            QualificationRule(float(q["threshold"]), float(q.get("confidence", 0.95))))


# --------------------------------------------------------------------------- #
# command and log
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MimicRunSpec:
    """What ``RunInvocation.spec`` carries for a Mimic run (the runners name runs by ``run_name``)."""
    run_name: str
    num_trials: int
    num_envs: int


def mimic_argv(python: str, arena_root: Path, m: MimicSection, *, num_trials: int, input_file: Path,
               output_file: Path, variations: Mapping[str, Any], app_flags: Sequence[str] = ()) -> list[str]:
    """``generate_dataset.py`` with the trial count, the files, the stock Arena environment and the scene's
    variation overrides last (``A:docs/.../sequential_static_manipulation/step_3_data_generation.rst:116-128``)."""
    argv = [python, str(Path(arena_root) / m.script), "--device", m.device,
            "--generation_num_trials", str(num_trials), "--num_envs", str(m.num_envs),
            "--input_file", str(input_file), "--output_file", str(output_file)]
    if m.enable_cameras:
        argv.append("--enable_cameras")
    argv += ["--mimic", "--external_callback", m.external_callback, "--task", m.task]
    for k, v in m.env_args.items():
        argv += [f"--{k}", v]
    argv += list(app_flags) + list(m.extra_args)
    argv += [hydra_override(p, v) for p, v in variations.items()]   # VERIFY(V-mimic-override)
    return argv


def trial_outcomes(log: str, output: MimicOutput) -> tuple[list[bool], tuple[int, int]]:
    """Per-attempt outcomes in order, and the final (successes, attempts), from the cumulative progress
    lines in ``log``. A line that adds several attempts with mixed outcomes cannot be ordered."""
    rx = re.compile(output.progress)
    outcomes: list[bool] = []
    s0 = a0 = 0
    for n, line in enumerate(log.splitlines(), 1):
        m = rx.search(line)
        if m is None:
            continue
        s, a = int(m["successes"]), int(m["attempts"])
        ds, da = s - s0, a - a0
        if da == 0 and ds == 0:
            continue                                   # the same count printed again
        if da < 0 or ds < 0 or ds > da:
            raise MimicOutputError(f"log line {n}: counts {s}/{a} after {s0}/{a0} are not cumulative")
        if 0 < ds < da:
            raise MimicOutputError(f"log line {n}: {da} attempts with mixed outcomes in one step ({s0}/{a0} -> "
                                   f"{s}/{a}); their order is unknown (run with num_envs: 1)")
        outcomes += [ds > 0] * da
        s0, a0 = s, a
    return outcomes, (s0, a0)


# --------------------------------------------------------------------------- #
# the reference runner
# --------------------------------------------------------------------------- #

class MimicReference:
    """``IsaacBackend``'s RAW reference runner (``ReferenceRunner``)."""

    def __init__(self, cfg: BackendConfig, task: TaskConfig, *, runner: Optional[ArenaRunner] = None) -> None:
        if task.reference is None:
            raise MimicConfigError(f"task config {task.source} has no reference section")
        self.cfg, self.task, self.section = cfg, task, task.reference
        self.runner = runner or SubprocessRunner()

    def source_dataset(self) -> Path:
        m = self.section.mimic
        path = self.cfg.datasets.get(m.source_dataset)   # VERIFY(V-mimic-source): a local copy, never downloaded
        if path is None:
            raise MimicConfigError(f"the Mimic source dataset {m.source_dataset!r} is not in the backend config's "
                                   f"datasets map (VERIFY(V-mimic-source)); it is never downloaded")
        if not Path(path).is_file():
            raise MimicConfigError(f"the Mimic source dataset {m.source_dataset!r} is not at {path}")
        return Path(path)

    def check_actor(self, actor: ActorRef) -> None:
        if wrapper_reference(actor):
            raise ReplayUnavailable("Mimic is a RAW reference; through the wrapper (G3) it is deferred (decision 3)")
        if actor.kind is not ActorKind.REFERENCE or actor.pipeline is not ActionPipeline.RAW:
            raise ValueError(f"Mimic runs only the RAW reference, not {actor.key()}")
        if actor.checkpoint != self.section.actor_checkpoint:
            raise MimicConfigError(f"reference actor {actor.checkpoint!r} is not this task's Mimic reference "
                                   f"({self.section.actor_checkpoint!r})")

    def plan(self, scene: SceneSpec, actor: ActorRef, seeds: Sequence[int], *,
             variations: Mapping[str, Any]) -> RunInvocation:
        """The Mimic run for ``seeds`` (pure apart from resolving the source dataset)."""
        self.check_actor(actor)
        m, cfg = self.section.mimic, self.cfg
        key = {"scene": scene.key(), "actor": actor.key(), "seeds": list(seeds), "variations": dict(variations)}
        out = cfg.output_root / "reference" / short_hash(key, 16)
        # VERIFY(V-mimic-seed): the pilot env fixes the datagen seed (1), so two runs may draw the same trials.
        argv = mimic_argv(cfg.arena_python, cfg.arena_root, m, num_trials=len(seeds), input_file=self.source_dataset(),
                          output_file=out / "generated.hdf5", variations=variations, app_flags=cfg.app_flags)
        spec = MimicRunSpec(run_name=f"mimic_{short_hash(key, 8)}", num_trials=len(seeds), num_envs=m.num_envs)
        return RunInvocation(argv=tuple(argv), cwd=cfg.arena_root, env={}, timeout_s=cfg.timeouts.run_s,
                             log_path=out / "mimic.log", output_dir=out, spec=spec)

    def run(self, scene: SceneSpec, actor: ActorRef, seeds: Sequence[int], *,
            variations: Mapping[str, Any]) -> list[EpisodeResult]:
        seeds = [int(s) for s in seeds]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"need distinct seeds, got {seeds}")
        inv = self.plan(scene, actor, seeds, variations=variations)
        if inv.output_dir.exists():
            shutil.rmtree(inv.output_dir)              # a fresh directory per run: no stale log
        inv.output_dir.mkdir(parents=True)
        self.runner.run(inv)
        if not inv.log_path.is_file():
            raise ArenaRunError(f"Mimic run {inv.spec.run_name} left no log at {inv.log_path}", log_path=inv.log_path)
        outcomes, (successes, attempts) = trial_outcomes(inv.log_path.read_text(encoding="utf-8", errors="replace"),
                                                         self.section.mimic.output)
        if len(outcomes) < len(seeds):
            raise MimicOutputError(f"Mimic run {inv.spec.run_name}: {len(outcomes)} attempts in the log, {len(seeds)} "
                                   f"trials asked for (VERIFY(V-mimic-out)); log {inv.log_path}")
        used = outcomes[:len(seeds)]
        meta = {"reference": "mimic", "actor": actor.key(), "scene": scene.key(), "seeds": seeds,
                "trials": used, "attempts_total": attempts, "successes_total": successes, "paired": False,
                "argv": list(inv.argv), "log": str(inv.log_path), "variations": dict(variations)}
        meta_path = inv.output_dir / META_FILE
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        paths = {"vla_meta": str(meta_path), "log": str(inv.log_path), "output_file": argv_value(inv.argv, "--output_file")}
        return [EpisodeResult(s, ok, {"reference": "mimic", "paired": False, "trial_index": i,
                                      "attempts_total": attempts, "variations": dict(variations), "paths": paths})
                for i, (s, ok) in enumerate(zip(seeds, used))]


def argv_value(argv: Sequence[str], flag: str) -> Optional[str]:
    argv = list(argv)
    return argv[argv.index(flag) + 1] if flag in argv else None


# --------------------------------------------------------------------------- #
# qualification (ISSUES #5)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Qualification:
    condition: str                           # SceneSpec.key() of the qualified scene
    k: int
    n: int
    lower: float
    upper: float
    rule: QualificationRule

    @property
    def qualified(self) -> bool:
        return self.lower >= self.rule.threshold

    def to_dict(self) -> dict:
        return {"condition": self.condition, "k": self.k, "n": self.n, "lower": self.lower, "upper": self.upper,
                "threshold": self.rule.threshold, "confidence": self.rule.confidence, "qualified": self.qualified}

    def evidence(self) -> Optional[Evidence]:
        """``reference_unqualified`` (row 2) when unqualified; None when qualified."""
        if self.qualified:
            return None
        return Evidence(f"ref.unqualified.{short_hash(self.condition, 8)}", EVIDENCE_SOURCE,
                        V.SIG_REFERENCE_UNQUALIFIED, 1.0, f"reference.{self.condition}",
                        f"the reference controller succeeded in {self.k} of {self.n} trials on {self.condition}; "
                        f"Wilson {self.rule.confidence:.0%} lower bound {self.lower:.3f} < threshold "
                        f"{self.rule.threshold:.3f}")


def qualify(results: Sequence[EpisodeResult], rule: QualificationRule, condition: str) -> Qualification:
    """The reference's qualification on one condition from its trial results."""
    if not results:
        raise ValueError("no reference trials to qualify")
    k = sum(bool(r.success) for r in results)
    p = wilson_interval(k, len(results), rule.confidence)
    return Qualification(condition, k, len(results), p.lo, p.hi, rule)
