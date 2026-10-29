import os, sys, shutil, subprocess, json, time
from pathlib import Path
H = os.path.expanduser("~")
src = open(f"{H}/router_check.py").read().split("clean, bugged =")[0]
exec(src)                                    # traces, obs_nominal, check, CODE ... (router code unchanged)
from router.schemas import Verdict, VerdictLabel, Destination
from router.report import write_report, RepoIndex, ReportContext
from hagent.config import load_settings
from hagent.models import OpenAICompatModel
from hagent.runner import solve

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BASE = "/tmp/pw-bug02-orig"
ev = check(obs_nominal(traces(f"{H}/router_ev/clean")), traces(f"{H}/router_ev/bugged"))
assert ev, "router produced no evidence"
verdict = Verdict(VerdictLabel.CODE, Destination.HAGENT, 4, tuple(e.id for e in ev), location=ev[0].location)
settings = load_settings(f"{H}/comp_and_research/Penguin_Triage/hybrid-agent/config.toml")
small = OpenAICompatModel(settings.small, "small"); big = OpenAICompatModel(settings.big, "big")
out = Path(f"{H}/auto_repair"); out.mkdir(exist_ok=True)
FLIP = "[:, ::-1, ::-1]"
results = []
for i in range(1, N + 1):
    work = f"/tmp/pw-auto{i}"; shutil.rmtree(work, ignore_errors=True); shutil.copytree(BASE, work)
    ctx = ReportContext(f"auto{i}", "gr1_fridge_ranch", "run policy_runner (not available to the agent)",
                        "success 0/30 in the failing configuration; stock client 3/4", work)
    rep = write_report(verdict, ev, small, repo=RepoIndex.from_dir(work), context=ctx)
    prompt = rep.hagent_prompt or rep.task
    (out / f"attempt{i}_prompt.txt").write_text(prompt)
    t0 = time.time()
    res = solve(prompt, work, settings, "hybrid", small=small, big=big)
    core = Path(work, "policy_wrapper/policy/gr00t_core.py").read_text()
    changed = subprocess.run(["git", "-C", work, "diff", "HEAD", "--name-only"], capture_output=True, text=True).stdout.split()
    fixed = FLIP not in core and changed == ["policy_wrapper/policy/gr00t_core.py"]
    (out / f"attempt{i}.patch").write_text(res.patch or "")
    r = dict(attempt=i, report_source=rep.source, localization=[(l.path if hasattr(l, "path") else str(l)) for l in rep.localization],
             exit=res.exit_status, flip_removed=FLIP not in core, files_changed=changed, fixed=fixed,
             cost=round(res.total_cost + rep.cost, 4), secs=int(time.time() - t0))
    results.append(r); print(json.dumps(r), flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
print(f"AUTO-FIXED {sum(r['fixed'] for r in results)}/{len(results)}")
