#!/usr/bin/env python3
"""Automatic repair of an injected policy-wrapper bug: router evidence -> report -> hagent -> scripted check.

From the 2026-10-29 session; ran only there, on the box, with Nemotron models on Nebius Token Factory
(NEBIUS_API_KEY in the environment; never written by this script). Generalised from the session's
auto_repair.py (BUG-02), auto_bug07.py (BUG-07) and auto_bug03.py (BUG-03): the repo, bug id, number of
attempts and output directory are arguments. It runs outside the incident loop: nothing here is the router
or ladder driving a run, and no simulator is started.

Per attempt: copy the bugged repo, build the task, run `hagent solve` in hybrid mode (local backend), score
the result, and write results.json and attemptN.patch (and attemptN_prompt.txt) before the next attempt.

Evidence modes (--bug sets a default, --evidence overrides):
  obs   router ObservationContractCheck on instrument directories (--obs-nominal, --obs-check) ->
        router.report.write_report with the small model (RepoIndex from the repo; no file named by a human)
  cfg   router ConfigContractCheck on the repo (--checkpoint) -> write_report
  none  no router evidence: a symptom-only task from --task-file (no location, no file named)

Scoring (--bug sets a default, --score overrides):
  absent:FILE:TEXT   TEXT no longer occurs in FILE and FILE is the only file changed (a line check: a different
                     valid fix would be scored not fixed)
  cfg                the router's own config check on the patched repo gives 0 evidence items

The first BUG-07 scoring in the session compared the patched repo with the clean one byte for byte and was
wrong (it scored 0/5); the `cfg` score is the one reported.

usage: auto_repair.py REPO --bug BUG-02 --out DIR [-n 5] [--hagent-config hybrid-agent/config.toml] ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # vla-isaac
sys.path.insert(0, str(HERE))

PRESETS = {
    "BUG-02": {"evidence": "obs", "score": "absent:policy_wrapper/policy/gr00t_core.py:[:, ::-1, ::-1]",
               "condition": "success 0/30 in the failing configuration; stock client 3/4"},
    "BUG-07": {"evidence": "cfg", "score": "cfg",
               "condition": "success 0/20 in the failing configuration; stock client 3/4"},
    "BUG-03": {"evidence": "none", "score": "absent:policy_wrapper/utils/joints_conversion.py:-values if joint_group.endswith",
               "condition": "success 0/20 in the failing configuration; stock client 3/4"},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Automatic repair attempts with hagent on an injected repo.")
    p.add_argument("repo", type=Path, help="the bugged policy-wrapper repo (a git repo; copied per attempt)")
    p.add_argument("--bug", required=True, help="bug id, used for the preset and in the output")
    p.add_argument("-n", "--attempts", type=int, default=5)
    p.add_argument("--out", type=Path, required=True, help="results.json, attemptN.patch and prompts go here")
    p.add_argument("--hagent-config", type=Path, default=ROOT.parent / "hybrid-agent" / "config.toml")
    p.add_argument("--evidence", choices=("obs", "cfg", "none"))
    p.add_argument("--score", help="absent:FILE:TEXT or cfg")
    p.add_argument("--obs-nominal", type=Path, help="obs: clean instrument directory")
    p.add_argument("--obs-check", type=Path, help="obs: bugged instrument directory")
    p.add_argument("--checkpoint", type=Path, help="cfg: checkpoint directory with statistics.json")
    p.add_argument("--checkpoint-id", default=None)
    p.add_argument("--task-config", type=Path, default=None)
    p.add_argument("--task-file", type=Path, help="none: the symptom-only task text")
    p.add_argument("--work-dir", type=Path, default=None, help="where per-attempt copies are made (default: temp)")
    a = p.parse_args(argv)
    preset = PRESETS.get(a.bug, {})
    a.evidence = a.evidence or preset.get("evidence")
    a.score = a.score or preset.get("score")
    a.condition = preset.get("condition", "failing in the bugged configuration")
    for need, flag in ((a.evidence, "--evidence"), (a.score, "--score")):
        if not need:
            p.error(f"{flag} is required for {a.bug} (no preset)")
    if a.attempts < 1:
        p.error("-n must be at least 1")
    if a.evidence == "obs" and not (a.obs_nominal and a.obs_check):
        p.error("--evidence obs needs --obs-nominal and --obs-check")
    if a.evidence == "cfg" and not a.checkpoint:
        p.error("--evidence cfg needs --checkpoint")
    if a.evidence == "none" and not a.task_file:
        p.error("--evidence none needs --task-file")
    if a.score != "cfg" and not a.score.startswith("absent:"):
        p.error("--score must be 'cfg' or 'absent:FILE:TEXT'")
    if a.score.startswith("absent:") and a.score.count(":") < 2:
        p.error("--score absent: needs FILE:TEXT")
    if a.score == "cfg" and not a.checkpoint:
        p.error("--score cfg needs --checkpoint")
    return a


def changed_files(work: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(work), "diff", "HEAD", "--name-only"], capture_output=True, text=True)
    return out.stdout.split()


def score_absent(work: Path, spec: str, changed: list[str]) -> bool:
    _, file, text = spec.split(":", 2)
    path = work / file
    return path.exists() and text not in path.read_text() and changed == [file]


def router_evidence(a):
    """Router evidence for the bugged repo, or () in none mode."""
    from router.schemas import Destination, Verdict, VerdictLabel

    if a.evidence == "obs":
        import router_obs_check as oc

        r = oc._router()
        nominal = oc.obs_nominal(r, oc.traces(r, a.obs_nominal))
        ev = oc.check(r, nominal, oc.traces(r, a.obs_check))
    else:
        ev = cfg_evidence(a, a.repo)
    if not ev:
        raise SystemExit("router produced no evidence; nothing to build a report from")
    return ev, Verdict(VerdictLabel.CODE, Destination.HAGENT, 4, tuple(e.id for e in ev), location=ev[0].location)


def cfg_evidence(a, repo: Path) -> list:
    import router_cfg_check as cc
    from vla_isaac.config import load_task_config

    task = load_task_config(a.task_config or cc.DEFAULT_TASK)
    return cc.check_repo(repo, task, a.checkpoint, a.checkpoint_id or cc.DEFAULT_CKPT_ID)


def main(argv=None) -> int:
    a = parse_args(argv)
    from hagent.config import load_settings
    from hagent.models import OpenAICompatModel
    from hagent.runner import solve

    settings = load_settings(str(a.hagent_config))
    small, big = OpenAICompatModel(settings.small, "small"), OpenAICompatModel(settings.big, "big")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.evidence != "none":
        from router.report import RepoIndex, ReportContext, write_report

        ev, verdict = router_evidence(a)
    base_work = a.work_dir or Path(tempfile.mkdtemp(prefix="auto-repair-"))
    results = []
    for i in range(1, a.attempts + 1):
        work = base_work / f"{a.bug}-attempt{i}"
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(a.repo, work)
        extra = {}
        if a.evidence == "none":
            prompt, cost = a.task_file.read_text(), 0.0
        else:
            ctx = ReportContext(f"{a.bug}-auto{i}", "gr1_fridge_ranch", "run policy_runner (not available to the agent)",
                                a.condition, str(work))
            rep = write_report(verdict, ev, small, repo=RepoIndex.from_dir(work), context=ctx)
            prompt, cost = rep.hagent_prompt or rep.task, rep.cost
            extra = {"report_source": rep.source, "localization": [str(item) for item in rep.localization]}
        (a.out / f"attempt{i}_prompt.txt").write_text(prompt)
        t0 = time.time()
        res = solve(prompt, str(work), settings, "hybrid", small=small, big=big)
        changed = changed_files(work)
        if a.score == "cfg":
            left = cfg_evidence(a, work)
            fixed, extra["config_evidence_left"] = not left, [e.id for e in left]
        else:
            fixed = score_absent(work, a.score, changed)
        (a.out / f"attempt{i}.patch").write_text(res.patch or "")
        r = dict(attempt=i, **extra, exit=res.exit_status, files=changed, fixed=fixed,
                 cost=round(res.total_cost + cost, 4), secs=int(time.time() - t0))
        results.append(r)
        print(json.dumps(r), flush=True)
        (a.out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"{a.bug} AUTO-FIXED {sum(r['fixed'] for r in results)}/{len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
