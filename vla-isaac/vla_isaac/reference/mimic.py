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
* **Trials, not paired seeds.** Every result has ``logs["paired"] = False``: a Mimic trial has nothing
  to do with the policy episode of the same seed. The outcomes are read from Mimic's log (``MimicOutput``,
  VERIFY(V-mimic-out)): one cumulative "successes/attempts" line per finished attempt gives each
  attempt's outcome in order. The pilot environment sets ``generation_guarantee: True``
  (``A:isaaclab_arena/tasks/common/mimic_default_params.py``), so Mimic may make more attempts than
  ``generation_num_trials`` (it counts successes, VERIFY(V-mimic-guarantee)); the first attempts are
  used, a fixed-size sample, and the total is kept in the logs. Too few attempts, or an attempt count
  the log cannot order, is a ``MimicOutputError``.
* **The datagen seed** (``reference.mimic.seed``, VERIFY(V-mimic-seed), a day-one GPU check). The pilot
  environment fixes Mimic's datagen seed, and whether a per-run seed reaches it is unread. Two modes:

  - ``per_run``: every run passes its own seed (``seed.flag``), derived from the scene, actor and
    requested seeds, so two runs draw independent trials. Trial i of the run is the result for
    ``seeds[i]``. ``logs["deterministic"] = False``.
  - ``fixed`` (the default until V-mimic-seed is settled): every run replays the same trial sequence,
    so the reference is deterministic. Router seed s **is** trial s of that sequence: a run makes
    ``max(seeds) + 1`` trials and returns those indices, so a later request for seeds 10..19 gets trials
    10..19, never a repeat of 0..9 presented as new. ``logs["deterministic"] = True``.

Qualification (ISSUES #5): ``qualify(results, rule, condition)``. The reference is unqualified when the
Wilson (``ladder.stats``) **lower bound** of its trial success rate is below the configured threshold,
and ``Qualification.evidence()`` is then the ``reference_unqualified`` evidence (row 2) to pass to
``route(..., extra_evidence=...)``. ``rule.trials`` (50 in the pilot) is how many trials a qualification
runs (``qualification_seeds``): with 20 only 20/20 clears 0.8 at 95 %. A deterministic reference gives a
**single pass/fail**: one fixed trial sequence, the same rule applied once, ``deterministic = True``, and
no interval reported (repeating the run repeats the trials; there is nothing to pool).
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


SEED_MODES = ("fixed", "per_run")


@dataclass(frozen=True)
class MimicSeed:
    """How Mimic's datagen seed behaves across runs (VERIFY(V-mimic-seed)). ``fixed``: every run replays
    one trial sequence (router seed s = trial s, at most ``max_trials``). ``per_run``: each run passes its
    own seed with ``flag`` (the flag itself is unverified)."""
    mode: str = "fixed"
    flag: Optional[str] = None
    max_trials: int = 200

    def __post_init__(self) -> None:
        if self.mode not in SEED_MODES:
            raise ConfigError(f"reference.mimic.seed.mode must be one of {SEED_MODES}, got {self.mode!r}")
        if self.mode == "per_run" and not self.flag:
            raise ConfigError("reference.mimic.seed.mode per_run needs seed.flag (the CLI flag of the datagen seed)")
        if self.max_trials < 1:
            raise ConfigError("reference.mimic.seed.max_trials must be >= 1")

    @property
    def deterministic(self) -> bool:
        return self.mode == "fixed"


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
    seed: MimicSeed = field(default_factory=MimicSeed)

    def __post_init__(self) -> None:
        if self.num_envs < 1:
            raise ConfigError("reference.mimic.num_envs must be >= 1")
        flags = " ".join(map(str, self.extra_args)) + " " + " ".join(self.env_args)
        if "external_environment_class_path" in flags or "vla_isaac" in self.external_callback:
            raise ConfigError("reference.mimic must use Arena's stock environment, never the actuated launcher")


@dataclass(frozen=True)
class QualificationRule:
    """Unqualified when the Wilson lower bound of the trial success rate is below ``threshold``.
    ``trials``: how many trials a qualification runs."""
    threshold: float
    confidence: float = 0.95
    trials: int = 50

    def __post_init__(self) -> None:
        if not (0.0 < self.threshold <= 1.0 and 0.0 < self.confidence < 1.0):
            raise ConfigError("reference.qualification: need 0 < threshold <= 1 and 0 < confidence < 1")
        if self.trials < 1:
            raise ConfigError("reference.qualification.trials must be >= 1")


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
    if "seed" in m:
        sd = _strict(m["seed"], MimicSeed, "reference.mimic.seed", ("mode",))
        m["seed"] = MimicSeed(str(sd["mode"]), None if sd.get("flag") is None else str(sd["flag"]),
                              int(sd.get("max_trials", 200)))
    m["env_args"] = {str(k): str(v) for k, v in m["env_args"].items()}
    m["extra_args"] = tuple(str(x) for x in m.get("extra_args", ()))
    q = _strict(top["qualification"], QualificationRule, "reference.qualification", ("threshold", "trials"))
    return ReferenceSection(str(top["actor_checkpoint"]), MimicSection(**m),
                            QualificationRule(float(q["threshold"]), float(q.get("confidence", 0.95)), int(q["trials"])))


def qualification_seeds(rule: QualificationRule) -> list[int]:
    """The seeds of a qualification run: ``0 .. trials-1`` (in ``fixed`` mode, trials 0 .. trials-1)."""
    return list(range(rule.trials))


# --------------------------------------------------------------------------- #
# command and log
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MimicRunSpec:
    """What ``RunInvocation.spec`` carries for a Mimic run (the runners name runs by ``run_name``).
    ``trial_indices[i]`` is the trial whose outcome is the result for the i-th requested seed."""
    run_name: str
    num_trials: int
    num_envs: int
    trial_indices: tuple[int, ...] = ()
    run_seed: Optional[int] = None
    deterministic: bool = True


def mimic_argv(python: str, arena_root: Path, m: MimicSection, *, num_trials: int, input_file: Path,
               output_file: Path, variations: Mapping[str, Any], app_flags: Sequence[str] = (),
               run_seed: Optional[int] = None) -> list[str]:
    """``generate_dataset.py`` with the trial count, the files, the stock Arena environment, the run's
    datagen seed (``per_run`` mode only) and the scene's variation overrides last
    (``A:docs/.../sequential_static_manipulation/step_3_data_generation.rst:116-128``)."""
    argv = [python, str(Path(arena_root) / m.script), "--device", m.device,
            "--generation_num_trials", str(num_trials), "--num_envs", str(m.num_envs),
            "--input_file", str(input_file), "--output_file", str(output_file)]
    if m.enable_cameras:
        argv.append("--enable_cameras")
    argv += ["--mimic", "--external_callback", m.external_callback, "--task", m.task]
    for k, v in m.env_args.items():
        argv += [f"--{k}", v]
    argv += list(app_flags) + list(m.extra_args)
    if run_seed is not None:
        argv += [str(m.seed.flag), str(run_seed)]                   # VERIFY(V-mimic-seed): the flag reaches the datagen RNG
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
        if m.seed.deterministic:
            # VERIFY(V-mimic-seed): every run replays one trial sequence, so seed s is trial s of it.
            bad = [s for s in seeds if not 0 <= s < m.seed.max_trials]
            if bad:
                raise ValueError(f"with a fixed Mimic datagen seed, seed s is trial s: need 0 <= s < "
                                 f"{m.seed.max_trials} (reference.mimic.seed.max_trials), got {bad}")
            indices, run_seed = tuple(seeds), None
        else:
            indices = tuple(range(len(seeds)))
            run_seed = int(short_hash({"scene": scene.key(), "actor": actor.key(), "seeds": list(seeds)}, 8), 16)
        num_trials = max(indices) + 1
        argv = mimic_argv(cfg.arena_python, cfg.arena_root, m, num_trials=num_trials, input_file=self.source_dataset(),
                          output_file=out / "generated.hdf5", variations=variations, app_flags=cfg.app_flags,
                          run_seed=run_seed)
        spec = MimicRunSpec(run_name=f"mimic_{short_hash(key, 8)}", num_trials=num_trials, num_envs=m.num_envs,
                            trial_indices=indices, run_seed=run_seed, deterministic=m.seed.deterministic)
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
        spec = inv.spec
        outcomes, (successes, attempts) = trial_outcomes(inv.log_path.read_text(encoding="utf-8", errors="replace"),
                                                         self.section.mimic.output)
        if len(outcomes) < spec.num_trials:
            raise MimicOutputError(f"Mimic run {spec.run_name}: {len(outcomes)} attempts in the log, {spec.num_trials} "
                                   f"trials asked for (VERIFY(V-mimic-out)); log {inv.log_path}")
        used = [outcomes[i] for i in spec.trial_indices]
        meta = {"reference": "mimic", "actor": actor.key(), "scene": scene.key(), "seeds": seeds,
                "trials": used, "trial_indices": list(spec.trial_indices), "attempts_total": attempts,
                "successes_total": successes, "paired": False, "deterministic": spec.deterministic,
                "run_seed": spec.run_seed, "argv": list(inv.argv), "log": str(inv.log_path),
                "variations": dict(variations)}
        meta_path = inv.output_dir / META_FILE
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        paths = {"vla_meta": str(meta_path), "log": str(inv.log_path), "output_file": argv_value(inv.argv, "--output_file")}
        return [EpisodeResult(s, ok, {"reference": "mimic", "paired": False, "trial_index": i,
                                      "deterministic": spec.deterministic, "run_seed": spec.run_seed,
                                      "attempts_total": attempts, "variations": dict(variations), "paths": paths})
                for s, ok, i in zip(seeds, used, spec.trial_indices)]


def argv_value(argv: Sequence[str], flag: str) -> Optional[str]:
    argv = list(argv)
    return argv[argv.index(flag) + 1] if flag in argv else None


# --------------------------------------------------------------------------- #
# qualification (ISSUES #5)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Qualification:
    """``deterministic``: the reference replays one fixed trial sequence (``fixed`` datagen seed). It is
    then a single pass/fail: ``qualified`` applies the same rule once to that sequence, and no interval
    is reported (``interval()`` is None, ``to_dict`` has ``lower`` / ``upper`` None), because repeating
    the run repeats the trials."""
    condition: str                           # SceneSpec.key() of the qualified scene
    k: int
    n: int
    lower: float                             # the rule's Wilson bound over the n trials
    upper: float
    rule: QualificationRule
    deterministic: bool = False

    @property
    def qualified(self) -> bool:
        return self.lower >= self.rule.threshold

    @property
    def rate(self) -> float:
        return self.k / self.n

    def interval(self) -> Optional[tuple[float, float]]:
        return None if self.deterministic else (self.lower, self.upper)

    def to_dict(self) -> dict:
        lo, hi = self.interval() or (None, None)
        return {"condition": self.condition, "k": self.k, "n": self.n, "rate": self.rate, "lower": lo, "upper": hi,
                "threshold": self.rule.threshold, "confidence": self.rule.confidence, "qualified": self.qualified,
                "deterministic": self.deterministic}

    def evidence(self) -> Optional[Evidence]:
        """``reference_unqualified`` (row 2) when unqualified; None when qualified."""
        if self.qualified:
            return None
        if self.deterministic:
            why = (f"deterministic reference (one fixed trial sequence, single pass/fail): fails the "
                   f"{self.rule.confidence:.0%} Wilson rule at threshold {self.rule.threshold:.3f}")
        else:
            why = (f"Wilson {self.rule.confidence:.0%} lower bound {self.lower:.3f} < threshold "
                   f"{self.rule.threshold:.3f}")
        return Evidence(f"ref.unqualified.{short_hash(self.condition, 8)}", EVIDENCE_SOURCE,
                        V.SIG_REFERENCE_UNQUALIFIED, 1.0, f"reference.{self.condition}",
                        f"the reference controller succeeded in {self.k} of {self.n} trials on {self.condition}; {why}")


def qualify(results: Sequence[EpisodeResult], rule: QualificationRule, condition: str) -> Qualification:
    """The reference's qualification on one condition from its trial results.

    Results that carry ``logs["deterministic"] = True`` (a ``fixed`` Mimic seed) must all do so and must be
    distinct trials (``trial_index``): the same trial twice is a repeated run, not more data. Results
    without the flag (a seed varied per run, or another backend) are pooled as independent trials."""
    if not results:
        raise ValueError("no reference trials to qualify")
    flags = {bool(r.logs.get("deterministic", False)) for r in results}
    if len(flags) > 1:
        raise ValueError("reference trials mix a deterministic and a per-run-seeded reference")
    deterministic = flags == {True}
    if deterministic:
        idx = [r.logs.get("trial_index", r.seed) for r in results]
        if len(set(idx)) != len(idx):
            raise ValueError(f"a deterministic reference repeats its trials; trials {sorted(i for i in set(idx) if idx.count(i) > 1)} "
                             f"appear more than once (repeated runs are not new data)")
    k = sum(bool(r.success) for r in results)
    p = wilson_interval(k, len(results), rule.confidence)
    return Qualification(condition, k, len(results), p.lo, p.hi, rule, deterministic)
