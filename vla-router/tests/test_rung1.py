"""Rung 1 (plan.md Section 4): branch-keyed two-scene staged validation, the compact result, the CLI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hagent.config import CustomCommand
from hagent.env import LocalEnv
from ladder.backend import MockRule
from ladder.rung1 import (Rung1Outcome, git_diff_hash, main, run_rung1, rung1_target, staged_plan_for)
from ladder.stats import StageDecision
from router.schemas import CodeRef
from router.stage2 import Branch
from scripts.confusion_matrix import LIGHT
from tests.ladder_cases import CASES, PATCHED, FnBackend, rung1_config
from tests.router_cases import QuotaBackend

CASE = CASES["combined_code_plus_light"]          # incident scene: lighting shifted; nominal: pick_mug
INCIDENT = CASE.incident()
C0 = INCIDENT.code_ref
CFG = rung1_config()


def quota(nom_before, nom_after, lit_before, lit_after) -> QuotaBackend:
    """Noise-free rates for (nominal | lit scene) x (before = c0 | after = c1)."""
    return QuotaBackend((MockRule(lit_after, code=PATCHED, factors_present=(LIGHT,)),
                         MockRule(nom_after, code=PATCHED),
                         MockRule(lit_before, code=C0, factors_present=(LIGHT,))), nom_before)


def go(backend, branch=Branch.G, cfg=CFG, incident=INCIDENT):
    return run_rung1(backend, incident, PATCHED, branch, cfg)


# --------------------------------------------------------------------------- #
# the four outcomes
# --------------------------------------------------------------------------- #

def test_pass_when_target_restored_and_other_scene_restored():
    r = go(quota(0.4, 0.9, 0.1, 0.9))
    assert r.outcome is Rung1Outcome.PASS and r.stage == "A" and r.n == 10
    assert r.target.which == "nominal" and r.other.which == "incident" and r.other.state == "restored"
    assert r.to_router and not r.back_to_repair and not r.to_rung2


def test_pass_with_residual_when_other_scene_improved_but_not_restored():
    r = go(quota(0.4, 0.9, 0.1, 0.4))
    assert r.outcome is Rung1Outcome.PASS_WITH_RESIDUAL
    assert r.other.improved and r.other.state == "residual"
    assert r.to_router and not r.back_to_repair          # never back to the repair agent


def test_pass_with_residual_when_other_scene_is_unchanged():
    """The gap in the user's table: unchanged is no regression, so it is residual, not fail."""
    r = go(quota(0.4, 0.9, 0.1, 0.1))
    assert r.outcome is Rung1Outcome.PASS_WITH_RESIDUAL
    assert not r.other.improved and r.other.state == "residual"


def test_fail_when_target_is_still_at_the_broken_baseline():
    r = go(quota(0.1, 0.1, 0.1, 0.4))
    assert r.outcome is Rung1Outcome.FAIL and r.stage == "A" and r.target_decision is StageDecision.FAIL
    assert r.back_to_repair and any("broken baseline" in n for n in r.notes)


def test_fail_on_regression_even_when_the_target_is_restored():
    r = go(quota(0.4, 0.9, 0.9, 0.2), branch=Branch.G)        # lit scene was fine, patch breaks it
    assert r.target_decision is StageDecision.PASS
    assert r.outcome is Rung1Outcome.FAIL and r.other.state == "regression"
    assert any("regression" in n for n in r.notes)


def test_ambiguous_after_stage_b_goes_to_rung2():
    r = go(quota(0.1, 0.4, 0.1, 0.1))
    assert r.outcome is Rung1Outcome.AMBIGUOUS and r.stage == "B" and r.n == 50
    assert r.to_rung2 and r.target_decision is StageDecision.AMBIGUOUS


def test_regression_beats_an_ambiguous_target():
    r = go(quota(0.1, 0.4, 0.9, 0.1))
    assert r.outcome is Rung1Outcome.FAIL and r.stage == "A"


# --------------------------------------------------------------------------- #
# the scene is keyed on the BRANCH
# --------------------------------------------------------------------------- #

def test_target_scene_follows_the_branch_not_the_flags():
    be = quota(0.4, 0.9, 0.1, 0.4)
    assert rung1_target(Branch.G) == rung1_target(Branch.G_DEGRADED) == "nominal"
    assert rung1_target(Branch.C) == "incident"
    assert go(be, Branch.G).target.which == "nominal"
    assert go(be, Branch.G_DEGRADED).target.which == "nominal"
    r = go(be, Branch.C)
    assert r.target.which == "incident" and r.other.which == "nominal"
    # the same patch and data: branch G passes at Stage A (nominal 0.4 -> 0.9), branch C does not,
    # because its target (the incident scene, 0.1 -> 0.4) is only partly recovered -> ambiguous at Stage B
    assert go(be, Branch.G).outcome is Rung1Outcome.PASS_WITH_RESIDUAL
    assert r.outcome is Rung1Outcome.AMBIGUOUS and r.stage == "B"


@pytest.mark.parametrize("branch", [Branch.UNKNOWN, None])
def test_no_target_for_an_unknown_branch(branch):
    with pytest.raises(ValueError, match="no Rung 1 target"):
        go(quota(0.4, 0.9, 0.1, 0.4), branch)


def test_identical_code_is_rejected():
    with pytest.raises(ValueError, match="identical"):
        run_rung1(quota(0.4, 0.9, 0.1, 0.4), INCIDENT, C0, Branch.G, CFG)


def test_single_scene_incident_has_no_other_check():
    inc = CASES["code_partial_offset"].incident()               # incident scene == nominal scene
    r = run_rung1(QuotaBackend((MockRule(0.9, code=PATCHED),), 0.35), inc, PATCHED, Branch.G, CFG)
    assert r.other is None and r.outcome is Rung1Outcome.PASS


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #

def test_stage_b_reuses_stage_a_seeds_and_runs_each_episode_once():
    # A is inconclusive (6/10), B passes (>= 0.7 over 50): successes only on seeds < 6 and 10..49 with s%10 < 8
    def fn(scene, code, s):
        if code != PATCHED.key():
            return s % 10 < 1
        return s < 6 if s < 10 else s % 10 < 8
    be = FnBackend(fn)
    r = go(be, Branch.G)
    assert r.stage == "B" and r.n == 50 and r.outcome in (Rung1Outcome.PASS, Rung1Outcome.PASS_WITH_RESIDUAL)
    assert max(be.runs.values()) == 1                            # every (scene, code, seed) simulated once


def test_stage_a_and_b_use_the_bonferroni_split():
    plan = staged_plan_for(50)
    assert (plan.n_a, plan.n_b) == (10, 50) and plan.alpha_a + plan.alpha_b == pytest.approx(0.05)
    assert staged_plan_for(10).n_a == 5 and staged_plan_for(20).n_a == 10
    with pytest.raises(ValueError):
        staged_plan_for(1)


def test_seeds_are_the_incidents_failing_seeds_first():
    be = FnBackend(lambda scene, code, s: True)
    go(be)
    assert {k[3] for k in be.runs} == set(range(10))             # failing seeds 0..9, stage A only


# --------------------------------------------------------------------------- #
# compact output
# --------------------------------------------------------------------------- #

def test_compact_result_has_rate_interval_decision_and_error_mode():
    r = go(quota(0.4, 0.9, 0.1, 0.4))
    text = r.compact()
    assert text.splitlines()[0].startswith("RUNG1 PASS_WITH_RESIDUAL")
    assert "9/10=0.90" in text and "CI[" in text and "target decision pass" in text
    assert "top error mode:" in text and len(text.splitlines()) <= 6
    assert json.loads(json.dumps(r.to_dict()))["outcome"] == "pass_with_residual"


def test_top_error_mode_reads_the_failure_phase():
    from router.schemas import EpisodeResult
    from ladder.rung1 import top_error_mode
    eps = [EpisodeResult(0, False, {"failure_phase": "grasp"}), EpisodeResult(1, False, {"failure_phase": "grasp"}),
           EpisodeResult(2, False, {"error_mode": "reach"}), EpisodeResult(3, True)]
    assert top_error_mode(eps) == "grasp (2/3 failed)"
    assert top_error_mode([EpisodeResult(0, True)]).startswith("none")
    assert top_error_mode([EpisodeResult(0, False)]).startswith("unknown")


# --------------------------------------------------------------------------- #
# CLI, and registration as a hagent custom command
# --------------------------------------------------------------------------- #

def write_config(path: Path, after_rate: float, before_rate: float = 0.1, salt: str = "cli") -> Path:
    cfg = {
        "incident": INCIDENT.to_dict(), "branch": "G_degraded", "after": PATCHED.to_dict(),
        "thresholds": {"failure_rate": 0.15, "nominal_band": [0.70, 0.95], "agreement": 1.0, "min_agreement": 0.9},
        "backend": {"kind": "mock", "default_rate": before_rate, "salt": salt, "rules": [
            {"rate": after_rate, "code": PATCHED.to_dict()}]}}
    path.write_text(json.dumps(cfg))
    return path


def test_cli_prints_the_compact_result_and_exits_by_outcome(tmp_path, capsys):
    ok = write_config(tmp_path / "ok.json", 0.97)
    assert main(["--seeds", "30", "--config", str(ok)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("RUNG1 PASS") and "top error mode" in out
    bad = write_config(tmp_path / "bad.json", 0.1)
    assert main(["--seeds", "30", "--config", str(bad)]) == 1
    assert "RUNG1 FAIL" in capsys.readouterr().out
    assert main(["--seeds", "30", "--config", str(ok), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] in ("pass", "pass_with_residual")


def test_cli_errors_cleanly(tmp_path, capsys):
    assert main(["--config", str(tmp_path / "missing.json")]) == 2
    assert "run_rung1: error" in capsys.readouterr().err
    cfg = json.loads(write_config(tmp_path / "c.json", 0.9).read_text())
    cfg["backend"] = {"kind": "isaac"}
    p = tmp_path / "isaac.json"
    p.write_text(json.dumps(cfg))
    assert main(["--config", str(p)]) == 2                           # IsaacBackend is not built yet
    with pytest.raises(SystemExit):
        main([])                                                      # no config at all


def test_cli_is_registered_as_a_hagent_custom_command(tmp_path):
    script = Path(__file__).parents[1] / "ladder" / "bin" / "run_rung1"
    cfg = write_config(tmp_path / "cfg.json", 0.97)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    env = LocalEnv(repo)
    try:
        env.install_commands([CustomCommand("run_rung1", str(script), "Run staged validation. Usage: run_rung1 --seeds N")])
        obs = env.execute(f"RUNG1_CONFIG={cfg} run_rung1 --seeds 20")
    finally:
        env.close()
    assert obs.returncode == 0, obs.output
    assert obs.output.startswith("RUNG1 PASS") and "top error mode" in obs.output


def test_default_after_code_hash_matches_the_repairers_recipe(tmp_path):
    """No ``after`` in the config: the CLI hashes the working-tree diff like hagent's git_diff_hash."""
    from ladder.repair import patched_code_ref
    repo = tmp_path / "r"
    repo.mkdir()
    for a in (["init", "-q"], ["config", "user.name", "t"], ["config", "user.email", "t@t"]):
        subprocess.run(["git", *a], cwd=repo, check=True)
    (repo / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "i"], cwd=repo, check=True)
    (repo / "a.py").write_text("x = 2\n")
    env = LocalEnv(repo)
    patch = env.git_diff()
    assert patched_code_ref(CodeRef("c0"), patch).code_hash == f"c0+{git_diff_hash(str(repo))}"
