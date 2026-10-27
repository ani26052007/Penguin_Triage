"""Phase 5: the failure-map grid on MockBackend (known rates), with the reference qualification table,
resume after a crash, and sharding across two fake hosts."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from ladder.backend import MockBackend, MockRule
from ladder.stats import wilson_interval
from router.schemas import ActorKind, CodeRef, Factor, Mode, SceneSpec

from vla_isaac.config import ConfigError, load_task_config
from vla_isaac.experiments.grid import (CELLS_DIR, FLAG_CANDIDATE, FLAG_NOMINAL, FLAG_REF_UNQUALIFIED, GridError,
                                        GridRunner, GridSpec, GridTable, ReferenceSpec, collect, main)
from vla_isaac.scene.factors import FactorTable

from tests.helpers import PILOT_TASK, REF_RAW, REF_WRAP, REPO, make_backend

CODE = CodeRef("c0")
TASK = load_task_config(PILOT_TASK)
RULE = TASK.reference.qualification                       # the Phase 4 rule: 0.8 at 95 %, 50 trials
REFERENCE = ReferenceSpec(REF_RAW, RULE)
NOMINAL = "gr1_fridge_ranch"


DARK = SceneSpec(NOMINAL, (Factor("lighting.intensity", "100.0"),))
BRIGHT = SceneSpec(NOMINAL, (Factor("lighting.intensity", "3000.0"),))
HDR = SceneSpec(NOMINAL, (Factor("lighting.hdr", "home_office_robolab"),))
DARK_HDR = SceneSpec(NOMINAL, (Factor("lighting.intensity", "100.0"), Factor("lighting.hdr", "home_office_robolab")))


def grid(**kw) -> GridSpec:
    d = {"name": "test_grid", "nominal": NOMINAL, "episodes": 20, "policy_checkpoint": "ckpt",
         "factors": {"lighting.intensity": [None, "100.0", "3000.0"], "lighting.hdr": [None, "home_office_robolab"]},
         **kw}
    return GridSpec.from_dict(d)


POLICY = grid().policy
# Known rates. The policy: nominal 1.0, dark 0.0 (the failure), dark+HDR 0.0, everything else 0.9.
# The reference: 1.0 everywhere except dark+HDR, where it is 0.5 (something besides appearance changed).
RULES = (MockRule(1.0, scene=SceneSpec(NOMINAL), actor=POLICY), MockRule(0.0, scene=DARK, actor=POLICY),
         MockRule(0.0, scene=DARK_HDR, actor=POLICY), MockRule(0.9, actor=POLICY),
         MockRule(0.5, scene=DARK_HDR, actor=REF_RAW), MockRule(1.0, actor=REF_RAW))


def mock(**kw) -> MockBackend:
    return MockBackend(RULES, 0.0, salt="grid", **kw)


class Recording(MockBackend):
    """MockBackend that records its calls and can crash on the n-th policy cell."""

    def __init__(self, *a, crash_at=None, **kw):
        super().__init__(*a, **kw)
        self.log, self.crash_at = [], crash_at

    def run(self, scene, code_ref, actor, seeds, mode):
        if actor.kind is ActorKind.POLICY:
            if self.crash_at is not None and sum(a.kind is ActorKind.POLICY for _, a, _, _ in self.log) == self.crash_at:
                raise RuntimeError("simulated crash mid-grid")
        self.log.append((scene.key(), actor, list(seeds), Mode(mode)))
        return super().run(scene, code_ref, actor, seeds, mode)


def recording(**kw) -> Recording:
    return Recording(RULES, 0.0, salt="grid", **kw)


def run_all(tmp_path, spec=None, backend=None, **kw) -> GridTable:
    spec = spec or grid()
    GridRunner(backend or mock(), spec, CODE, tmp_path, reference=REFERENCE, **kw).run()
    return GridTable(spec, collect(spec, [tmp_path]))


# ---- the grid ------------------------------------------------------------------------------------

def test_cells_are_every_combination_with_the_nominal_cell_first():
    cells = grid().cells()
    assert [c.scene.key() for c in cells][0] == f"{NOMINAL}|" and cells[0].nominal
    assert len(cells) == 6 and len({c.cell_id for c in cells}) == 6
    assert {c.scene.key() for c in cells} == {s.key() for s in (SceneSpec(NOMINAL), DARK, BRIGHT, HDR, DARK_HDR,
                                                                  BRIGHT.with_factor("lighting.hdr", "home_office_robolab"))}
    no_null = grid(factors={"lighting.intensity": ["100.0", "3000.0"]})
    assert [c.scene.key() for c in no_null.cells()] == [f"{NOMINAL}|", DARK.key(), BRIGHT.key()]   # nominal added


def test_one_policy_run_and_one_reference_run_per_cell(tmp_path):
    be = recording()
    run_all(tmp_path, backend=be)
    policy = [c for c in be.log if c[1].kind is ActorKind.POLICY]
    ref = [c for c in be.log if c[1].kind is ActorKind.REFERENCE]
    assert len(policy) == len(ref) == 6 and len({c[0] for c in policy}) == 6
    assert all(c[2] == list(range(20)) and c[3] is Mode.CLEAN for c in policy)          # one build: episodes 0..19
    assert all(c[1] == REF_RAW and c[2] == list(range(50)) and c[3] is Mode.CLEAN for c in ref)   # 50 trials


def test_the_table_has_known_rates_wilson_intervals_and_the_reference_next_to_the_policy(tmp_path):
    rows = {r["scene"]: r for r in run_all(tmp_path).rows()}
    nominal, dark, dark_hdr = rows[f"{NOMINAL}|"], rows[DARK.key()], rows[DARK_HDR.key()]
    assert (nominal["policy"]["k"], nominal["policy"]["n"]) == (20, 20)
    assert dark["policy"]["k"] == 0 and dark_hdr["policy"]["k"] == 0
    for r in rows.values():
        p = wilson_interval(r["policy"]["k"], r["policy"]["n"], 0.95)                   # from ladder.stats
        assert (r["policy"]["lower"], r["policy"]["upper"]) == (p.lo, p.hi)
        q = wilson_interval(r["reference"]["k"], r["reference"]["n"], RULE.confidence)
        assert (r["reference"]["lower"], r["reference"]["upper"]) == (q.lo, q.hi) and r["reference"]["n"] == 50
    direct = mock()                                       # the same episodes, counted without the grid
    for s in (BRIGHT, HDR):
        assert rows[s.key()]["policy"]["k"] == sum(e.success for e in direct.run(s, CODE, POLICY, range(20), Mode.CLEAN))
    assert dark["reference"]["k"] == 50 and dark["reference"]["qualified"]
    assert 15 <= dark_hdr["reference"]["k"] <= 35 and not dark_hdr["reference"]["qualified"]


def test_cells_with_an_unqualified_reference_are_flagged_and_never_candidates(tmp_path):
    table = run_all(tmp_path)
    rows = {r["scene"]: r for r in table.rows()}
    assert rows[f"{NOMINAL}|"]["flags"] == [FLAG_NOMINAL]
    assert rows[DARK.key()]["flags"] == [FLAG_CANDIDATE] and not rows[DARK.key()][FLAG_REF_UNQUALIFIED]
    assert rows[DARK_HDR.key()]["flags"] == [FLAG_REF_UNQUALIFIED] and rows[DARK_HDR.key()][FLAG_REF_UNQUALIFIED]
    assert [s for s, r in rows.items() if r[FLAG_REF_UNQUALIFIED]] == [DARK_HDR.key()]
    csv = table.csv().splitlines()
    head = csv[0].split(",")
    assert head[:4] == ["cell_id", "scene", "factor:lighting.intensity", "factor:lighting.hdr"]
    assert "reference_lower" in head and FLAG_REF_UNQUALIFIED in head and "policy_upper" in head
    flagged = [line for line in csv[1:] if line.split(",")[head.index(FLAG_REF_UNQUALIFIED)] == "true"]
    assert len(flagged) == 1 and DARK_HDR.key() in flagged[0]


def test_without_a_reference_there_is_no_qualification(tmp_path):
    spec = grid(reference=False)
    be = recording()
    GridRunner(be, spec, CODE, tmp_path).run()
    table = GridTable(spec, collect(spec, [tmp_path]))
    assert all(r["reference"] is None and not r[FLAG_REF_UNQUALIFIED] for r in table.rows())
    assert all(c[1].kind is ActorKind.POLICY for c in be.log)
    with pytest.raises(ConfigError, match="no reference was given"):
        GridRunner(mock(), grid(), CODE, tmp_path / "x")


def test_the_grid_qualifies_only_the_raw_reference():
    with pytest.raises(ConfigError, match="RAW reference only"):
        ReferenceSpec(REF_WRAP, RULE)
    with pytest.raises(ConfigError, match="RAW reference only"):
        ReferenceSpec(POLICY, RULE)


class DeterministicReference(MockBackend):
    """The reference behaves like Mimic with a fixed datagen seed: deterministic trials."""

    def run(self, scene, code_ref, actor, seeds, mode):
        out = super().run(scene, code_ref, actor, seeds, mode)
        if actor.kind is ActorKind.REFERENCE:
            out = [replace(e, logs={**e.logs, "deterministic": True, "trial_index": e.seed}) for e in out]
        return out


def test_a_deterministic_reference_is_a_single_pass_fail_in_the_table(tmp_path):
    table = run_all(tmp_path, backend=DeterministicReference(RULES, 0.0, salt="grid"))
    rows = {r["scene"]: r for r in table.rows()}
    assert all(r["reference"]["deterministic"] and r["reference"]["lower"] is None for r in rows.values())
    assert rows[DARK_HDR.key()][FLAG_REF_UNQUALIFIED] and not rows[DARK.key()][FLAG_REF_UNQUALIFIED]
    head = table.csv().splitlines()[0].split(",")
    assert all(line.split(",")[head.index("reference_deterministic")] == "true" for line in table.csv().splitlines()[1:])


def test_the_table_is_written_as_json_and_csv(tmp_path):
    table = run_all(tmp_path)
    js, cs = table.write(tmp_path / "out")
    data = json.loads(js.read_text())
    assert data["grid"] == "test_grid" and len(data["cells"]) == 6 and data["candidate"] == {"nominal_min": 0.8,
                                                                                            "shifted_max": 0.4}
    assert cs.read_text() == table.csv() and len(cs.read_text().splitlines()) == 7


# ---- resume ------------------------------------------------------------------------------------------

def test_resume_after_a_crash_mid_grid_runs_only_the_missing_cells_and_gives_the_same_table(tmp_path):
    reference = run_all(tmp_path / "clean").to_dict()
    crashing = recording(crash_at=3)
    with pytest.raises(RuntimeError, match="simulated crash"):
        GridRunner(crashing, grid(), CODE, tmp_path / "out", reference=REFERENCE).run()
    done = sorted((tmp_path / "out" / CELLS_DIR).glob("*.json"))
    assert len(done) == 3
    (tmp_path / "out" / CELLS_DIR / ".half_written.json.tmp").write_text("{")         # a crash mid-write
    with pytest.raises(GridError, match="3 cell"):
        collect(grid(), [tmp_path / "out"])
    second = recording()
    GridRunner(second, grid(), CODE, tmp_path / "out", reference=REFERENCE).run()
    first_cells = {c[0] for c in crashing.log}
    assert len([c for c in second.log if c[1].kind is ActorKind.POLICY]) == 3
    assert not first_cells & {c[0] for c in second.log if c[1].kind is ActorKind.POLICY}
    assert GridTable(grid(), collect(grid(), [tmp_path / "out"])).to_dict() == reference
    third = recording()
    GridRunner(third, grid(), CODE, tmp_path / "out", reference=REFERENCE).run()
    assert third.log == []                                                              # all done: nothing reruns


def test_a_cell_run_with_another_configuration_is_refused_not_reused(tmp_path):
    run_all(tmp_path)
    for changed in (grid(episodes=10), grid(confidence=0.9)):
        with pytest.raises(GridError, match="another configuration"):
            GridRunner(mock(), changed, CODE, tmp_path, reference=REFERENCE).run()
    with pytest.raises(GridError, match="another configuration"):
        GridRunner(mock(), grid(), CodeRef("c1"), tmp_path, reference=REFERENCE).run()
    with pytest.raises(GridError, match="another configuration"):
        GridRunner(mock(), grid(), CODE, tmp_path, reference=ReferenceSpec(REF_RAW, replace(RULE, trials=20))).run()


# ---- sharding across two fake hosts ------------------------------------------------------------------

def test_two_hosts_produce_the_same_table_as_one_host(tmp_path):
    spec = grid(hosts=["l4-a", "l4-b"])
    one = run_all(tmp_path / "one", spec=spec)
    a, b = recording(), recording()                                     # two machines: separate backends
    ra = GridRunner(a, spec, CODE, tmp_path / "a", reference=REFERENCE).run(host="l4-a")
    rb = GridRunner(b, spec, CODE, tmp_path / "b", reference=REFERENCE).run(host="l4-b")
    assert [r.index for r in ra] == [0, 2, 4] and [r.index for r in rb] == [1, 3, 5]
    assert {c[0] for c in a.log}.isdisjoint({c[0] for c in b.log})
    merged = GridTable(spec, collect(spec, [tmp_path / "a", tmp_path / "b"]))
    assert merged.to_dict() == one.to_dict() and merged.csv() == one.csv()
    with pytest.raises(GridError, match="not run yet"):
        collect(spec, [tmp_path / "a"])
    with pytest.raises(GridError, match="not one of the grid's hosts"):
        GridRunner(a, spec, CODE, tmp_path / "a", reference=REFERENCE).run(host="l4-c")


def test_merge_refuses_two_different_results_for_one_cell_and_foreign_cells(tmp_path):
    spec = grid(hosts=["l4-a", "l4-b"])
    run_all(tmp_path / "one", spec=spec)
    other = MockBackend(RULES, 0.0, salt="another-salt")             # same cell, different outcomes
    GridRunner(other, spec, CODE, tmp_path / "two", reference=REFERENCE).run()
    with pytest.raises(GridError, match="two different results"):
        collect(spec, [tmp_path / "one", tmp_path / "two"])
    renamed = grid(name="other_grid", hosts=["l4-a", "l4-b"])
    with pytest.raises(GridError, match="belongs to grid"):
        collect(renamed, [tmp_path / "one"])


def test_the_merge_cli_writes_the_table(tmp_path):
    spec_path = tmp_path / "grid.yaml"
    import yaml
    spec_path.write_text(yaml.safe_dump({"name": "test_grid", "nominal": NOMINAL, "episodes": 20,
                                         "policy_checkpoint": "ckpt", "hosts": ["l4-a", "l4-b"],
                                         "factors": {"lighting.intensity": [None, "100.0", "3000.0"],
                                                     "lighting.hdr": [None, "home_office_robolab"]}}))
    spec = GridSpec.load(spec_path)
    for host in spec.hosts:
        GridRunner(mock(), spec, CODE, tmp_path / host, reference=REFERENCE).run(host=host)
    assert main(["merge", "--grid", str(spec_path), "--out", str(tmp_path / "merged"),
                 "--from", str(tmp_path / "l4-a"), str(tmp_path / "l4-b")]) == 0
    assert len(json.loads((tmp_path / "merged" / "grid_table.json").read_text())["cells"]) == 6


# ---- one build per cell on IsaacBackend (fake runner) ---------------------------------------------------

def test_on_isaac_backend_each_cell_is_one_build(tmp_path):
    rig = make_backend(tmp_path)
    spec = grid(episodes=3, reference=False, factors={"lighting.intensity": [None, "100.0"]})
    GridRunner(rig.backend, spec, rig.code(), tmp_path / "out").run()
    assert len(rig.runner.calls) == 2                                     # two cells, two builds
    assert all(inv.spec.seed == 0 and inv.spec.num_episodes == 3 and inv.spec.num_envs == 1
               for inv in rig.runner.calls)


# ---- the grid YAML ------------------------------------------------------------------------------------

def test_the_pilot_grid_loads_and_names_only_known_factors():
    spec = GridSpec.load(REPO / "config" / "grid_gr1_fridge.yaml")
    spec.check_factors(list(FactorTable.load(TASK.factor_table).specs))
    assert spec.mode is Mode.CLEAN and spec.reference and spec.code_base == "pw"
    assert spec.cells()[0].nominal and len(spec.cells()) == 8
    with pytest.raises(ConfigError, match="not in the factor table"):
        grid(factors={"lighting.glare": [None, "1"]}).check_factors(list(FactorTable.load(TASK.factor_table).specs))


@pytest.mark.parametrize("bad, match", [
    ({"seeds": 3}, "unknown keys"),
    ({"factors": {"lighting.intensity": []}}, "distinct levels"),
    ({"factors": {"lighting.intensity": ["1", "1"]}}, "distinct levels"),
    ({"episodes": 0}, "episodes"),
    ({"hosts": ["a", "a"]}, "duplicate hosts"),
    ({"candidate": {"nominal": 0.8}}, "unknown keys"),
])
def test_the_grid_yaml_is_strict(bad, match):
    with pytest.raises(ConfigError, match=match):
        grid(**bad)
