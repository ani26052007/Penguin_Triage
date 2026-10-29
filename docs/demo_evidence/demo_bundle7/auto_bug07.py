import os, sys, shutil, subprocess, json, time
from pathlib import Path
H = os.path.expanduser("~")
src = open(f"{H}/cfg_check.py").read().split("for name, repo in")[0]
exec(src)
from router.schemas import Verdict, VerdictLabel, Destination
from router.report import write_report, RepoIndex, ReportContext
from hagent.config import load_settings
from hagent.models import OpenAICompatModel
from hagent.runner import solve
ev = run("/tmp/pw-bug07"); assert ev, "no evidence"
verdict = Verdict(VerdictLabel.CODE, Destination.HAGENT, 4, tuple(e.id for e in ev), location=ev[0].location)
settings = load_settings(f"{R}/hybrid-agent/config.toml")
small = OpenAICompatModel(settings.small, "small"); big = OpenAICompatModel(settings.big, "big")
out = Path(f"{H}/auto_bug07"); out.mkdir(exist_ok=True); res = []
orig = Path("/tmp/pw-bug07/config/eval.yaml").read_text()
clean = Path(f"{R}/policy-wrapper/config/eval.yaml").read_text()
for i in range(1, 6):
    w = f"/tmp/pw-b07auto{i}"; shutil.rmtree(w, ignore_errors=True); shutil.copytree("/tmp/pw-bug07", w)
    ctx = ReportContext(f"b07auto{i}", "gr1_fridge_ranch", "run policy_runner (not available to the agent)",
                        "success 0/20 in the failing configuration; stock client 3/4", w)
    rep = write_report(verdict, ev, small, repo=RepoIndex.from_dir(w), context=ctx)
    t0 = time.time(); r = solve(rep.hagent_prompt or rep.task, w, settings, "hybrid", small=small, big=big)
    ch = subprocess.run(["git","-C",w,"diff","HEAD","--name-only"],capture_output=True,text=True).stdout.split()
    same = subprocess.run(["diff","-rq","--exclude=.git","--exclude=__pycache__",f"{R}/policy-wrapper",w],capture_output=True).returncode == 0
    d = dict(attempt=i, source=rep.source, exit=r.exit_status, equals_clean=same, files=ch,
             cost=round(r.total_cost+rep.cost,4), secs=int(time.time()-t0))
    res.append(d); print(json.dumps(d), flush=True)
    (out/f"attempt{i}.patch").write_text(r.patch or ""); (out/"results.json").write_text(json.dumps(res, indent=1))
print(f"BUG07 AUTO-FIXED {sum(x['equals_clean'] for x in res)}/{len(res)}")
