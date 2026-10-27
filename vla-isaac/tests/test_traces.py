"""Phase 3: ``IsaacTraceSource`` / ``IsaacBehaviorSource`` over ``IsaacBackend`` runs with traces.

The runs are the fake runner's, with ``tests/trace_fixtures.TraceWriter`` writing the real policy
wrapper's instrument files (scripted chunk schedule) and a synthetic HDF5 dataset for every record.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from router.prereq.trace import StepPair, TraceUnavailable
from router.schemas import Incident, Mode, TrainingSpec
from router.stage1.nominal import NominalStats
from router.stage1.staleness import M_CLEAN_DELAY, M_MODE_GAP, StalenessProbe, estimate_extra_delay
from router import vocab as V

from vla_isaac.config import load_task_config
from vla_isaac.runner.subprocess_runner import FakeArenaRunner
from vla_isaac.seeds import join_seed
from vla_isaac.traces import instrument_reader
from vla_isaac.traces.sources import IsaacBehaviorSource, IsaacTraceSource

from tests.helpers import NOMINAL, PILOT_TASK, POLICY, make_backend
from tests.trace_fixtures import CLOSE_AT, LENGTH, OBJECT0, PW, TraceWriter, policy_wrapper

LAYOUT = load_task_config(PILOT_TASK).traces
KINDS = ("success", "door", "grasp", "reach")              # episode k has kind KINDS[k % 4]
pytestmark = pytest.mark.skipif(not (PW / "policy_wrapper" / "instrument.py").is_file(),
                                reason=f"no policy-wrapper checkout at {PW}")


def kind(k: int) -> str:
    return KINDS[k % 4]


def make_rig(tmp_path, *, delay: int = 0, **kw):
    writer = TraceWriter(LAYOUT, failure_kind=kind, delay=delay)
    runner = FakeArenaRunner(success=lambda spec, k: kind(k) == "success", on_run=writer)
    return make_backend(tmp_path, runner=runner, **kw), writer


def incident(code, mode: Mode, ref, observed=Mode.CLEAN) -> Incident:
    return Incident("inc", NOMINAL, TrainingSpec("gr00t"), code, POLICY, failing_seeds=(1, 2, 3),
                    observed_mode=observed, log_refs={mode.value: str(ref)})


def traced(tmp_path, seeds=range(8), mode=Mode.CLEAN, **kw):
    rig, writer = make_rig(tmp_path, **kw)
    code = rig.code()
    results = rig.backend.run(NOMINAL, code, POLICY, list(seeds), mode, record_trajectories=True)
    refs = {r.logs["paths"]["run_root"] for r in results}
    assert len(refs) == 1
    return rig, code, results, incident(code, mode, refs.pop())


@pytest.fixture
def clean(tmp_path):
    return traced(tmp_path)


# ---- the run writes what the sources read -----------------------------------------------------------

def test_a_traced_run_switches_the_wrapper_instrument_on_and_records_where(clean):
    rig, code, results, _ = clean
    meta = json.loads(Path(results[0].logs["paths"]["vla_meta"]).read_text())
    inv = rig.runner.calls[0]
    assert inv.env[instrument_reader.ENV_DIR] == meta["instrument_dir"]
    assert inv.env[instrument_reader.ENV_FRAME_EVERY] == "10" and inv.env[instrument_reader.ENV_PROFILE_BUCKETS] == "16"
    assert inv.env[instrument_reader.ENV_OBS_KEYS] == "policy.left_eef_pos,policy.right_eef_pos"
    assert meta["record_trajectories"] and meta["seed_stride"] == 10000 and meta["code_key"] == code.key()


def test_an_untraced_run_sets_no_instrument_variables(tmp_path):
    rig, _ = make_rig(tmp_path)
    rig.backend.run(NOMINAL, rig.code(), POLICY, [0, 1], Mode.CLEAN)
    assert not [k for k in rig.runner.calls[0].env if k.startswith("VLA_INSTRUMENT")]


def test_the_instrument_variable_names_are_the_wrappers():
    instrument, _ = policy_wrapper()
    for name in ("ENV_DIR", "ENV_FRAME_EVERY", "ENV_PROFILE_BUCKETS", "ENV_OBS_KEYS", "FORMAT"):
        assert getattr(instrument_reader, name) == getattr(instrument, name)


# ---- EpisodeTrace -------------------------------------------------------------------------------------

def test_synthetic_hdf5_jsonl_and_instrument_files_give_episode_traces(clean):
    _, code, results, inc = clean
    traces = IsaacTraceSource(LAYOUT).load(inc, Mode.CLEAN)
    assert [t.seed for t in traces] == [join_seed(0, k, 10000) for k in range(8)] == [r.seed for r in results]
    t = traces[0]
    assert t.mode is Mode.CLEAN and t.code_key == code.key()
    assert [f.step for f in t.frames] == list(range(0, LENGTH, 16))       # one sample per request
    f = t.frames[0]
    assert (f.camera, f.policy_input.shape, len(f.policy_input.row_profile)) == ("ego_view", (32, 32, 3), 16)
    assert f.render == f.policy_input                                     # the fixture sends the frame unchanged
    assert t.step_pairs[:2] == (StepPair(0, 0, 0), StepPair(0, 1, 1)) and len(t.step_pairs) == LENGTH
    assert {p.extra_delay for p in t.step_pairs} == {0}
    assert t.raw_actions[1] == (2.0, 3.0) and len(t.raw_actions) == LENGTH   # chunk row 1, as the server sent it
    assert len(t.joint_cmd) == len(t.joint_pos) == LENGTH and len(t.joint_cmd[0]) == 36
    assert t.gripper_width[0] == pytest.approx(1.0) and t.gripper_width[-1] == pytest.approx(0.0)
    assert t.gripper_cmd_width[CLOSE_AT - 1] == pytest.approx(t.gripper_width[CLOSE_AT])   # cmd one step ahead
    assert t.sim.nan_fraction == 0.0 and t.sim.object_initial_pos == pytest.approx(tuple(OBJECT0))
    assert t.sim.object_rest_pos == pytest.approx(tuple(OBJECT0))          # at the objects_settled event


def test_tracked_joints_are_reordered_to_the_action_joints(clean):
    t = IsaacTraceSource(LAYOUT).load(clean[3], Mode.CLEAN)[0]
    i = LAYOUT.action_joints.index("R_index_proximal_joint")
    lo, hi = LAYOUT.hands["right"]["R_index_proximal_joint"]
    assert t.joint_pos[-1][i] == pytest.approx(hi) and t.joint_pos[0][i] == pytest.approx(lo)
    assert t.joint_pos[-1][LAYOUT.action_joints.index("left_shoulder_pitch_joint")] == pytest.approx(0.5)


# ---- BehaviorTrace ------------------------------------------------------------------------------------

def test_behavior_traces_have_the_grasp_moment_and_the_phases_with_the_door_as_post_place(clean):
    _, code, results, inc = clean
    traces = IsaacBehaviorSource(LAYOUT).load(inc, Mode.CLEAN)
    assert [t.success for t in traces] == [r.success for r in results] == [kind(k) == "success" for k in range(8)]
    by_kind = {kind(k): t for k, t in enumerate(traces)}
    ok = by_kind["success"]
    assert ok.grasp.step == CLOSE_AT and ok.grasp.ee_pos == pytest.approx(ok.grasp.target_pos)
    assert [(s.name, s.completed) for s in ok.subtasks] == [(p, True) for p in
                                                           ("reach", "grasp", "lift", "place", "post_place")]
    assert by_kind["door"].failure_phase() == "post_place" and by_kind["door"].grasp is not None
    assert by_kind["grasp"].failure_phase() == "grasp" and by_kind["grasp"].grasp is None
    assert by_kind["reach"].failure_phase() == "reach"
    assert all(t.code_key == code.key() for t in traces)


def test_without_progress_tracking_only_the_offline_subtasks_are_reported(clean):
    _, _, results, inc = clean
    path = Path(results[0].logs["paths"]["episode_results"])
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    for r in recs:
        r.pop("progress")
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    t = IsaacBehaviorSource(LAYOUT).load(inc, Mode.CLEAN)[1]               # a door failure
    assert [s.name for s in t.subtasks] == ["reach", "grasp"] and t.failure_phase() is None


def test_incident_mode_joins_the_three_sources_on_env_and_episode(tmp_path):
    _, code, results, inc = traced(tmp_path, seeds=range(6), mode=Mode.INCIDENT)
    traces = IsaacBehaviorSource(LAYOUT).load(inc, Mode.INCIDENT)
    assert [t.success for t in traces] == [r.success for r in results]
    assert [t.failure_phase() for t in traces] == [None, "post_place", "grasp", "reach", None, "post_place"]
    assert {len(t.step_pairs) for t in IsaacTraceSource(LAYOUT).load(inc, Mode.INCIDENT)} == {LENGTH}


# ---- refusals -----------------------------------------------------------------------------------------

def _sources():
    return IsaacTraceSource(LAYOUT), IsaacBehaviorSource(LAYOUT)


def test_traces_of_other_code_are_refused(clean):
    rig, code, _, inc = clean
    other = incident(type(code)(code.code_hash, {**code.eval_config, "server": {"denoising_steps": 8}}), Mode.CLEAN,
                     inc.log_refs["clean"])
    for src in _sources():
        assert src.load(inc, Mode.CLEAN)
        with pytest.raises(TraceUnavailable, match="not for the code under test"):
            src.load(other, Mode.CLEAN)


def test_a_run_without_a_code_key_is_refused(clean):
    _, _, results, inc = clean
    meta_path = Path(results[0].logs["paths"]["vla_meta"])
    meta = json.loads(meta_path.read_text())
    meta.pop("code_key")
    meta_path.write_text(json.dumps(meta))
    for src in _sources():
        with pytest.raises(TraceUnavailable, match="carry no code key"):
            src.load(inc, Mode.CLEAN)


@pytest.mark.parametrize("damage,match", [
    (lambda meta, here: (here / "vla_meta.json").unlink(), "no vla_meta.json"),
    (lambda meta, here: shutil.rmtree(meta["instrument_dir"]), "no instrument file"),
    (lambda meta, here: next(Path(meta["run_dir"]).glob("dataset_*")).unlink(), "no trajectory dataset"),
    (lambda meta, here: next(Path(meta["instrument_dir"]).glob("*.json")).write_text("{"), "cannot be read"),
])
def test_missing_or_broken_logs_are_unavailable_never_clean(clean, damage, match):
    _, _, results, inc = clean
    meta_path = Path(results[0].logs["paths"]["vla_meta"])
    damage(json.loads(meta_path.read_text()), meta_path.parent)
    for src in _sources():
        with pytest.raises(TraceUnavailable, match=match):
            src.load(inc, Mode.CLEAN)


def test_files_named_outside_the_run_directory_are_refused(clean, tmp_path):
    _, _, results, inc = clean
    meta_path = Path(results[0].logs["paths"]["vla_meta"])
    meta = json.loads(meta_path.read_text())
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(meta["instrument_dir"], elsewhere)
    meta["instrument_dir"] = str(elsewhere)
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(TraceUnavailable, match="outside the run directory"):
        IsaacTraceSource(LAYOUT).load(inc, Mode.CLEAN)


def test_an_untraced_run_and_a_missing_reference_are_unavailable_and_the_wrong_mode_is_an_error(tmp_path, clean):
    rig, _ = make_rig(tmp_path / "plain")
    res = rig.backend.run(NOMINAL, rig.code(), POLICY, [0, 1], Mode.CLEAN)
    plain = incident(rig.code(), Mode.CLEAN, res[0].logs["paths"]["run_root"])
    _, _, _, inc = clean
    for src in _sources():
        with pytest.raises(TraceUnavailable, match="did not record traces"):
            src.load(plain, Mode.CLEAN)
        with pytest.raises(TraceUnavailable, match="no log_refs"):
            src.load(inc, Mode.INCIDENT)
        wrong = Incident(inc.incident_id, inc.scene, inc.training_spec, inc.code_ref, inc.policy, inc.failing_seeds,
                         log_refs={"incident": inc.log_refs["clean"]})
        with pytest.raises(ValueError, match="clean'-mode run"):
            src.load(wrong, Mode.INCIDENT)


# ---- the staleness probe on the instrument's step pairs ----------------------------------------------

NOMINAL_DELAY = NominalStats.from_samples({M_CLEAN_DELAY: [0.0, 0.0, 0.0], M_MODE_GAP: [0.0, 0.0, 0.0]},
                                          default_min_std=0.1)


@pytest.mark.parametrize("delay", [0, 3])
def test_the_staleness_probe_recovers_a_constant_extra_delay_from_a_delay_buffer(tmp_path, delay):
    _, _, _, inc = traced(tmp_path, seeds=range(4), delay=delay)
    source = IsaacTraceSource(LAYOUT)
    clean = source.load(inc, Mode.CLEAN)
    assert {p.extra_delay for t in clean for p in t.step_pairs} == {delay}
    assert estimate_extra_delay(clean, NOMINAL_DELAY) == pytest.approx(float(delay))
    evidence = StalenessProbe(NOMINAL_DELAY, source).run(inc)
    if delay:
        [e] = evidence
        assert e.signal == V.SIG_STALENESS_CONSTANT and "3.00 steps later" in e.detail
    else:
        assert evidence == []
