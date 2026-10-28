"""Phase 8: the injected-bug ledger (``bugs/``), the injected repos, Rung 0 on them, and the Stage 1 evidence
each bug produces without a GPU.

* Ledger: fields, plan 6.5 coverage, no target on a Rung 0 forbidden path, VERIFY ids for what needs a GPU.
* Injection: each bug changes exactly its targets; combined and conflicting cases; refusals.
* Silent: every ``silent: true`` bug passes the repo's own tests and Rung 0; ``silent: false`` fails it.
* Detection, by the ledger's ``detection``:
  - ``synthetic``: the injected repo's own client is driven by ``tests/fixtures/pw_driver.py`` (FIXTURE
    server and frames, real instrument) and the router check named in ``evidence`` fires on its traces,
    against nominal stats calibrated on the clean repo; the clean repo with another seed gives nothing;
  - ``real-config``: the config contract on the injected repo's own ``config/eval.yaml``;
  - ``needs-gpu``: nothing GPU-free fires (asserted, so the ledger's "elimination" stays true); a real run
    settles it (``verify``).
* Leak audit: ``inject.audit_repo`` is clean on every injected repo, and catches each kind of leak planted.

Runs on the real policy-wrapper checkout at the ledger's ``base_commit``; skipped otherwise.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ladder.rung0 import run_rung0
from ladder.workspace import WorkspaceManager
from router import vocab as V
from router.schemas import Mode, SceneSpec, TrainingSpec
from router.stage1.action_contract import M_GRIPPER_SIGN, M_GRIPPER_TRACK, M_JOINT_TRACK, M_RANGE, \
    ActionContractCheck, action_space
from router.stage1.cfg_contract import ConfigContractCheck
from router.stage1.nominal import NominalStats
from router.stage1.obs_contract import ObservationContractCheck
from router.stage1.staleness import M_CLEAN_DELAY, M_MODE_GAP, StalenessProbe

from vla_isaac.specs.norm_stats import load_flat_statistics
from vla_isaac.workspace import config_mapping, eval_config_loader, rung0_config

from tests.bugs_support import (BUGS, CAMERA, CKPT, CKPT_ID, CODE, LEDGER, PW, TASK, incident, inject, needs_base,
                                obs_nominal, source, store)  # noqa: F401  (store is a fixture)
from tests.helpers import POLICY, REF_RAW, make_backend

ROOT = Path(__file__).resolve().parents[1]
VERIFY_MD = ROOT.parent / "docs" / "VERIFY_ON_GPU.md"
PLAN_CASES = {"code-global", "code-partial", "code-condition-specific", "infra", "rung0", "combined"}
SCENE_CASES = {"vision", "interacting-vision", "rendering-mismatch", "sim-asset", "action-gap", "no-fault"}
BY_DETECTION = {d: sorted(b.id for b in LEDGER.bugs.values() if b.raw["detection"] == d) for d in inject.DETECTIONS}


# --------------------------------------------------------------------------- #
# the ledger
# --------------------------------------------------------------------------- #

def test_the_ledger_has_the_phase8_bugs_with_every_field():
    assert sorted(LEDGER.bugs) == [f"BUG-{i:02d}" for i in range(1, 15)]
    for b in LEDGER.bugs.values():
        assert set(inject.BUG_FIELDS) <= set(b.raw), b.id
        assert b.raw["tag"] in inject.TAGS and b.raw["difficulty"] in inject.DIFFICULTIES
    assert [b.id for b in LEDGER.bugs.values() if not b.silent] == ["BUG-13"]
    assert LEDGER.bugs["BUG-14"].raw["labels"] == ["injected", "infra"] and LEDGER.bugs["BUG-14"].raw["verdict"] == "infra"


def test_every_plan_case_is_served_by_a_bug_a_combined_case_or_the_scene_spec():
    served = {c for b in LEDGER.bugs.values() for c in b.raw["plan_cases"]}
    served |= {c for case in LEDGER.cases.values() for c in case["plan_cases"]}
    assert served == PLAN_CASES
    assert set(LEDGER.raw["scene_cases"]) == SCENE_CASES
    for case in LEDGER.cases.values():
        assert case["condition"] and all(LEDGER.bugs[b].silent for b in case["bugs"])


def test_no_bug_targets_a_rung0_forbidden_path_or_the_instrumentation():
    forbidden = rung0_config(TASK).forbidden_prefixes
    assert "policy_wrapper/instrument.py" in forbidden
    for b in LEDGER.bugs.values():
        for t in (*b.targets, *inject.patch_files(b.patch)):
            assert not any(t.startswith(p) for p in forbidden), (b.id, t)
            assert "instrument" not in t, (b.id, t)


def test_the_tooling_lives_outside_policy_wrapper_and_outside_any_repo_it_builds(store):
    assert PW.resolve() not in BUGS.resolve().parents and BUGS.resolve() not in PW.resolve().parents
    repo = store.repo().repo
    assert BUGS.resolve() not in repo.resolve().parents and repo.resolve() not in BUGS.resolve().parents
    with pytest.raises(inject.InjectError, match="inside the base checkout or the injection tooling"):
        inject.inject(PW, BUGS / "out", ledger=LEDGER)
    with pytest.raises(inject.InjectError, match="inside the base checkout or the injection tooling"):
        inject.inject(PW, PW / "out", ledger=LEDGER)


def test_launcher_variables_read_like_ordinary_configuration():
    names = {k for b in LEDGER.bugs.values() for k in (b.raw.get("launcher_env") or {})}
    assert names == {"GR00T_CLIENT_PREFETCH"}
    for n in names:
        assert not re.search(r"stale|inject|bug|fault|ledger", n, re.IGNORECASE), n


def test_what_needs_a_gpu_names_a_verify_id_listed_in_verify_on_gpu():
    text = VERIFY_MD.read_text()
    needs = [b for b in LEDGER.bugs.values() if "needs-gpu" in b.raw["detection"]]
    assert sorted(b.id for b in needs) == ["BUG-03", "BUG-04", "BUG-12"]
    for b in LEDGER.bugs.values():
        if "verify" in b.raw:
            assert f"| {b.raw['verify']} |" in text, (b.id, b.raw["verify"])


def test_a_needs_gpu_entry_without_a_verify_id_is_refused(tmp_path):
    d = yaml.safe_load((BUGS / "ledger.yaml").read_text())
    entry = next(b for b in d["bugs"] if b["id"] == "BUG-03")
    del entry["verify"]
    for b in d["bugs"]:
        b["patch"] = str(BUGS / b["patch"])
    (tmp_path / "ledger.yaml").write_text(yaml.safe_dump(d))
    with pytest.raises(inject.InjectError, match="BUG-03.*VERIFY id"):
        inject.load_ledger(tmp_path / "ledger.yaml")


# --------------------------------------------------------------------------- #
# injection
# --------------------------------------------------------------------------- #

def changed_files(diff: str) -> set[str]:
    return set(re.findall(r"^diff --git a/(\S+) b/", diff, re.MULTILINE))


@needs_base
def test_the_clean_build_is_the_base_tree_without_its_history(store):
    clean = store.repo()
    assert clean.diff == "" and clean.bugs == () and clean.launcher_env == {}
    tree = subprocess.run(["git", "-C", str(clean.repo), "rev-parse", "HEAD^{tree}"], capture_output=True, text=True)
    base = subprocess.run(["git", "-C", str(PW), "rev-parse", "HEAD^{tree}"], capture_output=True, text=True)
    assert tree.stdout == base.stdout
    assert subprocess.run(["git", "-C", str(PW), "rev-list", "--count", "HEAD"], capture_output=True,
                          text=True).stdout.strip() != "1"


@needs_base
@pytest.mark.parametrize("bug_id", sorted(LEDGER.bugs))
def test_each_bug_changes_exactly_its_targets(store, bug_id):
    b = LEDGER.bugs[bug_id]
    res = store.repo(bug_id)
    assert changed_files(res.diff) == set(b.targets)
    assert res.launcher_env == (b.raw.get("launcher_env") or {})


@needs_base
@pytest.mark.parametrize("bug_ids", [("BUG-05", "BUG-06", "BUG-10", "BUG-12"), ("BUG-01", "BUG-05", "BUG-07"),
                                     ("BUG-02", "BUG-11", "BUG-13"), ("BUG-03", "BUG-04", "BUG-09")])
def test_bugs_combine_into_one_repo(store, bug_ids):
    res = store.repo(*bug_ids)
    assert changed_files(res.diff) == {t for b in bug_ids for t in LEDGER.bugs[b].targets}
    assert inject.audit_repo(res.repo, PW, LEDGER) == []


@needs_base
@pytest.mark.parametrize("case_id", sorted(LEDGER.cases))
def test_a_combined_case_adds_its_bugs_and_returns_its_scene_condition(store, case_id):
    case = LEDGER.cases[case_id]
    res = store.repo(case=case_id)
    assert res.bugs == tuple(case["bugs"]) and res.condition == case["condition"]
    assert inject.audit_repo(res.repo, PW, LEDGER) == []


@needs_base
def test_bugs_that_edit_the_same_lines_are_refused_and_nothing_is_left(tmp_path):
    with pytest.raises(inject.InjectError, match="BUG-14: the patch does not apply"):
        inject.inject(PW, tmp_path / "x", ["BUG-09", "BUG-14"], ledger=LEDGER)
    assert not (tmp_path / "x").exists()


@needs_base
def test_bad_requests_are_refused(tmp_path):
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "f").write_text("x")
    for kw, match in (({"out": tmp_path / "full"}, "not empty"), ({"bug_ids": ["BUG-99"]}, "no bug"),
                      ({"bug_ids": ["BUG-01", "BUG-01"]}, "named twice"), ({"case": "CASE-99"}, "no case"),
                      ({"bug_ids": ["BUG-07", "BUG-08"]}, "generate the same file")):
        args = {"out": tmp_path / "o", "ledger": LEDGER, "checkpoint_stats": CKPT / "statistics.json", **kw}
        with pytest.raises(inject.InjectError, match=match):
            inject.inject(PW, **args)
    with pytest.raises(inject.InjectError, match="--checkpoint-stats"):
        inject.inject(PW, tmp_path / "s", ["BUG-07"], ledger=LEDGER)


# --------------------------------------------------------------------------- #
# silent bugs pass the repo's tests and Rung 0; silent=false does not
# --------------------------------------------------------------------------- #

class _Env:
    """``run_rung0`` env over hagent's LocalEnv, with the injected change as ``git_diff``."""

    def __init__(self, work: Path, diff: str):
        from hagent.env import LocalEnv
        self.env, self.diff = LocalEnv(work, timeout_s=600), diff

    def execute(self, cmd, timeout_s=None):
        return self.env.execute(cmd, timeout_s)

    def git_diff(self):
        return self.diff


RUNG0 = rung0_config(TASK, lint_cmd=f"env -u PYTHONPATH {sys.executable} -m compileall -q policy_wrapper",
                     test_cmd=f"env -u PYTHONPATH {sys.executable} -m pytest -q -p no:cacheprovider", timeout_s=600)


@needs_base
@pytest.mark.parametrize("bug_id", sorted(LEDGER.bugs))
def test_silent_bugs_pass_rung0_and_the_others_fail_it_at_the_tests(store, tmp_path, bug_id):
    import shutil

    res = store.repo(bug_id)
    work = tmp_path / "repo"
    shutil.copytree(res.repo, work, symlinks=True)            # Rung 0 writes caches; the cached repo stays as built
    out = run_rung0(_Env(work, res.diff), RUNG0)
    checks = {c.name: c.status for c in out.checks}
    assert checks["diff"] == "pass" and checks["lint"] == "pass", out.feedback
    if LEDGER.bugs[bug_id].silent:
        assert out.passed and checks["tests"] == "pass", out.feedback
    else:
        assert not out.passed and checks["tests"] == "fail"


# --------------------------------------------------------------------------- #
# detection: real eval config
# --------------------------------------------------------------------------- #

STATS = load_flat_statistics(CKPT / "statistics.json", "gr1", TASK.eval_config.norm_stats_groups)
INSTRUCTION = "Place the sauce bottle on the top shelf of the fridge, and close the fridge door."
PILOT_SPEC = TrainingSpec("gr00t_n1d6_gr1_ranch_bottle_into_fridge", image_size=(512, 512), norm_stats=STATS,
                          action_horizon=16, replan_steps=16, denoising_steps=4, control_hz=50, render_settings={},
                          instruction_templates={"default": INSTRUCTION})


def code_ref(store, tmp_path, *bug_ids):
    ws = WorkspaceManager(tmp_path / "ws", {"pw": store.repo(*bug_ids).repo},
                          config_loader=eval_config_loader(TASK, {CKPT_ID: CKPT}))
    return ws.base_ref("pw")


def cfg_evidence(code) -> list:
    from router.schemas import Incident

    inc = Incident("pw", SceneSpec(TASK.nominal_scene), PILOT_SPEC, code, POLICY, (0,), Mode.CLEAN)
    return ConfigContractCheck(config_mapping(TASK)).run(inc)


@needs_base
def test_the_clean_eval_config_gives_no_config_evidence(store, tmp_path):
    assert cfg_evidence(code_ref(store, tmp_path)) == []


@needs_base
@pytest.mark.parametrize("bug_id", BY_DETECTION["real-config"])
def test_real_config_bugs_give_their_one_config_violation(store, tmp_path, bug_id):
    want = LEDGER.bugs[bug_id].raw["evidence"]
    (ev,) = cfg_evidence(code_ref(store, tmp_path, bug_id))
    assert (ev.source, ev.signal, ev.id, ev.location) == (want["check"], V.SIG_CONTRACT_VIOLATION, want["id"],
                                                          want["location"])


@needs_base
@pytest.mark.parametrize("bug_id", [b for b in sorted(LEDGER.bugs) if b not in BY_DETECTION["real-config"]
                                    and b != "BUG-13"])
def test_the_other_bugs_leave_the_mapped_config_keys_alone(store, tmp_path, bug_id):
    assert cfg_evidence(code_ref(store, tmp_path, bug_id)) == []


# --------------------------------------------------------------------------- #
# detection: driven runs
# --------------------------------------------------------------------------- #

def observation_evidence(store, bug_ids, scene) -> list:
    nominal = obs_nominal(store.run(scene=scene, seed=0).traces(Mode.CLEAN, CODE.key()))
    traces = store.run(*bug_ids, scene=scene, seed=1).traces(Mode.CLEAN, CODE.key())
    src, refs = source({Mode.CLEAN: traces})
    return ObservationContractCheck(nominal, src, [CAMERA]).run(incident(refs))


@needs_base
@pytest.mark.parametrize("scene", ["bright", "dark"])
def test_the_clean_repo_gives_no_observation_evidence_against_its_own_nominal(store, scene):
    assert observation_evidence(store, (), scene) == []


@needs_base
@pytest.mark.parametrize("bug_id, scene", [("BUG-01", "bright"), ("BUG-01", "dark"), ("BUG-02", "bright"),
                                           ("BUG-02", "dark"), ("BUG-11", "dark")])
def test_observation_bugs_fire_the_observation_contract(store, bug_id, scene):
    want = LEDGER.bugs[bug_id].raw["evidence"]
    ev = observation_evidence(store, (bug_id,), scene)
    assert ev and all(e.source == want["check"] and e.signal == V.SIG_CONTRACT_VIOLATION for e in ev)
    if bug_id == "BUG-11":
        assert {e.location for e in ev} == {"obs.normalization", "obs.channel_means"}
    else:
        (e,) = ev
        assert e.id == f"{want['id']}.{CAMERA}" and e.location == want["location"]
        assert ("BGR" if bug_id == "BUG-01" else "rotation by 180 degrees") in e.detail


@needs_base
def test_the_low_light_path_is_silent_in_bright_frames(store):
    """Condition-specific: the same repo gives nothing in the bright cell."""
    assert observation_evidence(store, ("BUG-11",), "bright") == []


DELAY_NOMINAL = NominalStats.from_samples({M_CLEAN_DELAY: [0.0, 0.0, 0.0], M_MODE_GAP: [0.0, 0.0, 0.0]},
                                          default_min_std=0.1)
LOAD = dict(num_envs=2, episodes=2, steps=64, server_delay_ms=60.0, step_ms=10.0)


def staleness_evidence(store, tmp_path, bug_ids) -> list:
    """Clean mode: 1 env, no load; incident mode: 2 envs, a slow server, a stepping env. Each mode's
    environment is what ``IsaacBackend.launch_env`` gives a policy actor with the case's launcher env."""
    launcher = dict(store.repo(*bug_ids).launcher_env)
    tmp_path.mkdir(parents=True, exist_ok=True)
    backend = make_backend(tmp_path, incident_env=launcher).backend
    clean = store.run(*bug_ids, env=backend.launch_env(POLICY, Mode.CLEAN)).traces(Mode.CLEAN, CODE.key())
    loaded = store.run(*bug_ids, env=backend.launch_env(POLICY, Mode.INCIDENT), **LOAD).traces(Mode.INCIDENT, CODE.key())
    src, refs = source({Mode.CLEAN: clean, Mode.INCIDENT: loaded})
    return StalenessProbe(DELAY_NOMINAL, src).run(incident(refs, Mode.INCIDENT))


@needs_base
def test_the_clean_repo_has_no_extra_delay_in_either_mode(store, tmp_path):
    assert staleness_evidence(store, tmp_path, ()) == []


@needs_base
def test_the_latency_buffer_is_a_constant_delay_present_in_clean_mode(store, tmp_path):
    want = LEDGER.bugs["BUG-09"].raw["evidence"]
    (ev,) = staleness_evidence(store, tmp_path, ("BUG-09",))
    assert (ev.signal, ev.id, ev.location) == (want["signal"], want["id"], want["location"])
    later = float(re.search(r"applied ([0-9.]+) steps later", ev.detail).group(1))
    assert 1.5 < later <= 2.0                                   # a 3-deep buffer: two steps, less the first ones


@needs_base
def test_the_prefetch_is_staleness_under_load_that_is_absent_in_clean_mode(store, tmp_path):
    want = LEDGER.bugs["BUG-14"].raw["evidence"]
    (ev,) = staleness_evidence(store, tmp_path, ("BUG-14",))
    assert (ev.signal, ev.id, ev.location) == (want["signal"], want["id"], want["location"])
    clean = store.run("BUG-14").traces(Mode.CLEAN, CODE.key())
    assert {p.extra_delay for t in clean for p in t.step_pairs} == {0}


@needs_base
def test_the_prefetch_stays_off_without_its_variable_even_under_load(store):
    loaded = store.run("BUG-14", **LOAD).traces(Mode.INCIDENT, CODE.key())
    assert {p.extra_delay for t in loaded for p in t.step_pairs} == {0}


PREFETCH_ERRORS = """
import sys
sys.path[:0] = [sys.argv[1] + "/tests/fake_gr00t", sys.argv[1]]
from policy_wrapper.policy.gr00t_remote_closedloop_policy import _ChunkPrefetch

def boom():
    raise ConnectionError("server gone")

p = _ChunkPrefetch()
p.submit(boom, None)
p.cancel()                                  # the discarded request's error is swallowed
assert not p.pending
p.submit(boom, None)
try:
    p.take()                                # a taken chunk's error propagates
except ConnectionError:
    print("take raised")
p.close()
"""


@needs_base
def test_the_prefetch_drops_a_discarded_requests_error_and_propagates_a_taken_ones(store):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-c", PREFETCH_ERRORS, str(store.repo("BUG-14").repo)], capture_output=True,
                       text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "take raised"


# -- BUG-12: gains, a FIXTURE joint model ----------------------------------------------------- #

NOMINAL_GAINS = {"stiffness": 3000.0, "damping": 100.0}      # GR1 arm actuators (Arena's GR1 config)
CONTROL_HZ, SUBSTEPS = 50, 20


def track(cmd: list[list[float]], cols: list[int], gains: dict) -> list[list[float]]:
    """FIXTURE: the tracked joints for commanded ``cmd``: a unit-inertia PD joint per column in ``cols``
    (semi-implicit Euler, ``SUBSTEPS`` per control step), the other columns exactly on command."""
    k, d, h = gains["stiffness"], gains["damping"], 1.0 / (CONTROL_HZ * SUBSTEPS)
    pos, vel, out = list(cmd[0]), [0.0] * len(cmd[0]), []
    for row in cmd:
        for _ in range(SUBSTEPS):
            for j in cols:
                vel[j] += h * (k * (row[j] - pos[j]) - d * vel[j])
                pos[j] += h * vel[j]
        out.append([pos[j] if j in cols else row[j] for j in range(len(row))])
    return out


def arm_columns(repo: Path) -> list[int]:
    ev = yaml.safe_load((repo / "config" / "eval.yaml").read_text())["client"]
    action = yaml.safe_load((repo / ev["action_joints_config_path"]).read_text())["joints"]
    groups = yaml.safe_load((repo / ev["policy_joints_config_path"]).read_text())["joints"]
    return sorted(action[n] for g in ("left_arm", "right_arm") for n in groups[g] if n in action)


def tracking_traces(store, bug_ids, seed, gains_of) -> list:
    from router.prereq.trace import EpisodeTrace

    repo = store.repo(*bug_ids).repo
    cols = arm_columns(repo)
    run = store.run(*bug_ids, seed=seed)
    return [EpisodeTrace(seed=i, mode=Mode.CLEAN, joint_cmd=tuple(map(tuple, cmd)),
                         joint_pos=tuple(map(tuple, track(cmd, cols, gains_of))), code_key=CODE.key())
            for i, (_, cmd) in enumerate(sorted(run.actions().items()))]


def gains_for(store, tmp_path, bug_ids, actor) -> dict:
    """What ``IsaacBackend`` would actuate for ``actor`` with the injected repo's eval config."""
    ev = yaml.safe_load((store.repo(*bug_ids).repo / "config" / "eval.yaml").read_text())
    tmp_path.mkdir(parents=True, exist_ok=True)
    rig = make_backend(tmp_path, eval_config=ev)
    overrides = rig.backend.actuator_overrides(actor, rig.code().eval_config)
    groups = {g: {**NOMINAL_GAINS, **overrides.get(g, {})} for g in ("left-arm", "right-arm")}
    assert groups["left-arm"] == groups["right-arm"]
    return groups["left-arm"]


def joint_evidence(store, tmp_path, bug_ids) -> list:
    from router.stage1.action_contract import action_metrics

    cal = tracking_traces(store, (), 0, NOMINAL_GAINS)
    samples = action_metrics(cal, action_space(PILOT_SPEC))
    assert set(samples) == {M_JOINT_TRACK}
    floor = max(0.05 * abs(sum(samples[M_JOINT_TRACK]) / len(samples[M_JOINT_TRACK])), 1e-3)
    nominal = NominalStats.from_samples({**samples, M_RANGE: [0.0, 0.0], M_GRIPPER_SIGN: [1.0, 1.0],
                                         M_GRIPPER_TRACK: [0.0, 0.0]}, min_std={M_JOINT_TRACK: floor})
    traces = tracking_traces(store, bug_ids, 1, gains_for(store, tmp_path, bug_ids, POLICY))
    src, refs = source({Mode.CLEAN: traces})
    return ActionContractCheck(nominal, src).run(incident(refs))


@needs_base
def test_nominal_gains_give_no_tracking_evidence(store, tmp_path):
    assert joint_evidence(store, tmp_path, ()) == []


@needs_base
def test_the_gain_decay_fires_joint_tracking_for_the_policy_and_never_reaches_the_reference(store, tmp_path):
    want = LEDGER.bugs["BUG-12"].raw["evidence"]
    assert gains_for(store, tmp_path / "p", ("BUG-12",), POLICY) == {"stiffness": 300.0, "damping": 100.0}
    assert gains_for(store, tmp_path / "r", ("BUG-12",), REF_RAW) == NOMINAL_GAINS      # G2 is unaffected
    (ev,) = joint_evidence(store, tmp_path / "e", ("BUG-12",))
    assert (ev.source, ev.id, ev.location) == (want["check"], want["id"], want["location"])


# -- needs-gpu: nothing GPU-free fires ---------------------------------------------------------- #

@needs_base
@pytest.mark.parametrize("bug_id", BY_DETECTION["needs-gpu"])
def test_needs_gpu_bugs_fire_no_gpu_free_check(store, tmp_path, bug_id):
    """The ledger's "elimination" for these is true here: no observation, staleness, config or
    (FIXTURE-tracked) joint evidence; only a real run's behaviour shows them (``verify``)."""
    assert LEDGER.bugs[bug_id].raw["evidence"]["check"] == "elimination"
    assert observation_evidence(store, (bug_id,), "bright") == []
    assert staleness_evidence(store, tmp_path / "s", (bug_id,)) == []
    assert cfg_evidence(code_ref(store, tmp_path / "c", bug_id)) == []
    assert joint_evidence(store, tmp_path / "j", (bug_id,)) == []


@needs_base
def test_the_gr1_hands_have_no_gripper_for_the_sign_check_and_their_closure_clips(store):
    """Why the hand sign is elimination-only: no single-index gripper in GR1's action layout, and a
    negated hand command reads as fully open, exactly like a hand that was never told to close."""
    from vla_isaac.traces.hand_closure import closure

    layout = {"left_arm": [0, 7], "right_arm": [7, 14], "left_hand": [14, 20], "right_hand": [20, 26]}
    spec = TrainingSpec("gr1", action_layout=layout)
    assert action_space(spec).gripper_index is None and action_space(spec, gripper_name="right_hand").gripper_index is None
    span = dict(TASK.traces.hands["right"])
    names = list(span)
    closed = [hi for lo, hi in span.values()]
    negated = [-hi for lo, hi in span.values()]
    opened = [lo for lo, hi in span.values()]
    assert closure([closed], names, span) == [1.0]
    assert closure([negated], names, span) == closure([opened], names, span) == [0.0]


# --------------------------------------------------------------------------- #
# leak audit
# --------------------------------------------------------------------------- #

def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


@needs_base
@pytest.mark.parametrize("bug_id", [None, *sorted(LEDGER.bugs)])
def test_every_injected_repo_passes_the_leak_audit(store, bug_id):
    res = store.repo(*(() if bug_id is None else (bug_id,)))
    assert inject.audit_repo(res.repo, PW, LEDGER) == []


@needs_base
@pytest.mark.parametrize("bug_id", ["BUG-07", "BUG-14"])
def test_an_injected_repo_is_one_commit_with_only_its_tree_and_nothing_that_names_the_injection(store, bug_id):
    repo = store.repo(bug_id).repo
    assert len(git(repo, "log", "--all", "--format=%H").split()) == 1
    assert len(git(repo, "reflog").splitlines()) == 1
    reachable = {line.split()[0] for line in git(repo, "rev-list", "--objects", "--all").splitlines()}
    stored = set(git(repo, "cat-file", "--batch-all-objects", "--batch-check=%(objectname)").split())
    assert stored == reachable
    counts = dict(line.split(": ") for line in git(repo, "count-objects", "-v").splitlines())
    assert int(counts["count"]) + int(counts["in-pack"]) == len(stored) and counts["garbage"] == "0"
    every = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch-all-objects", "--batch"], capture_output=True,
                           check=True).stdout.decode(errors="replace").lower()
    files = "\n".join(p.read_text(errors="replace") for p in repo.rglob("*")
                      if p.is_file() and ".git" not in p.relative_to(repo).parts).lower()
    for word in ("bug", "inject", "ledger", "todo_bug", "vla-isaac", "vla_isaac", *map(str.lower, LEDGER.secrets())):
        assert word not in every and word not in files, word
    mtimes = {int(p.lstat().st_mtime) for p in [repo, *repo.rglob("*")] if ".git" not in p.relative_to(repo).parts}
    assert mtimes == {int(git(PW, "log", "-1", "--format=%ct").strip())}
    assert git(repo, "log", "-1", "--format=%an %ae %aI %cn %ce %cI") == git(PW, "log", "-1", "--format=%an %ae %aI %cn %ce %cI")
    assert not (repo / ".git" / "hooks").exists() or not list((repo / ".git" / "hooks").iterdir())


@needs_base
@pytest.mark.parametrize("plant", ["commit", "word", "secret", "mtime", "base_blob", "path"])
def test_the_audit_catches_each_planted_leak(store, tmp_path, plant):
    import shutil

    work = tmp_path / "repo"
    shutil.copytree(store.repo("BUG-05").repo, work, symlinks=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "x", "GIT_AUTHOR_EMAIL": "x@x", "GIT_COMMITTER_NAME": "x",
           "GIT_COMMITTER_EMAIL": "x@x"}
    if plant == "commit":
        (work / "README.md").write_text((work / "README.md").read_text() + "\n")
        subprocess.run(["git", "-C", str(work), "commit", "-qam", "more"], check=True, env=env)
        match = "commits reachable"
    elif plant == "word":
        p = work / "config" / "eval.yaml"
        p.write_text(p.read_text() + "# see ledger\n")
        match = "contains 'ledger'"
    elif plant == "secret":
        p = work / "README.md"
        p.write_text(p.read_text() + "chunk_replan_mismatch\n")
        match = "chunk_replan_mismatch"
    elif plant == "mtime":
        os.utime(work / "README.md", (1, 1))
        match = "mtime"
    elif plant == "base_blob":
        blob = git(PW, "rev-parse", "HEAD:config/eval.yaml").strip()
        data = subprocess.run(["git", "-C", str(PW), "cat-file", "blob", blob], capture_output=True, check=True).stdout
        subprocess.run(["git", "-C", str(work), "hash-object", "-w", "--stdin"], input=data, check=True,
                       capture_output=True)
        match = "not reachable|base's version of config/eval.yaml"
    else:
        (work / "notes_injected.txt").write_text("x")
        match = "path notes_injected.txt"
    problems = inject.audit_repo(work, PW, LEDGER)
    assert any(re.search(match, p) for p in problems), problems
