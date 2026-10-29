import os, shutil, subprocess, json, time
from pathlib import Path
from hagent.config import load_settings
from hagent.models import OpenAICompatModel
from hagent.runner import solve
H = os.path.expanduser("~")
settings = load_settings(f"{H}/comp_and_research/Penguin_Triage/hybrid-agent/config.toml")
small = OpenAICompatModel(settings.small, "small"); big = OpenAICompatModel(settings.big, "big")
TASK = ("Incident report (measured): a GR00T policy driven by this repo's client (policy_wrapper) fails a "
        "put-the-bottle-in-the-fridge task: 0 of 20 episodes succeed. With the same checkpoint and seeds, "
        "Arena's stock client succeeds about 3 of 4. The robot's arms move toward the bottle but the hands "
        "never close on it, so it cannot grasp. Router evidence: config and observation contracts are clean "
        "(images, instruction, replan and denoising settings match training); no located cause. "
        "Treat any cause as a hypothesis to verify. Task: find the bug in this repo and fix it with the "
        "smallest possible change. Do not change config values, tests or the server. Run "
        "`python -m compileall -q policy_wrapper` before you finish.")
out = Path(f"{H}/auto_bug03"); out.mkdir(exist_ok=True); res_all = []
for i in range(1, 6):
    w = f"/tmp/pw-b03auto{i}"; shutil.rmtree(w, ignore_errors=True); shutil.copytree("/tmp/pw-bug03", w)
    t0 = time.time(); r = solve(TASK, w, settings, "hybrid", small=small, big=big)
    f = Path(w, "policy_wrapper/utils/joints_conversion.py").read_text()
    ch = subprocess.run(["git","-C",w,"diff","HEAD","--name-only"],capture_output=True,text=True).stdout.split()
    fixed = "-values if joint_group.endswith" not in f and ch == ["policy_wrapper/utils/joints_conversion.py"]
    (out/f"attempt{i}.patch").write_text(r.patch or "")
    d = dict(attempt=i, exit=r.exit_status, fixed=fixed, files=ch, cost=round(r.total_cost,4), secs=int(time.time()-t0))
    res_all.append(d); print(json.dumps(d), flush=True)
    (out/"results.json").write_text(json.dumps(res_all, indent=1))
print(f"BUG03 AUTO-FIXED {sum(x['fixed'] for x in res_all)}/{len(res_all)}")
