# rescore07.py -- from the 2026-10-29/30 session. Rescoring of the BUG-07 automatic-repair attempts
# with the router's own config check (the first scoring, byte-identical to clean, was wrong).
# Merged version of two steps that ran separately on the VM (apply patches, then check).
# Needs cfg_check.py (same folder) and its hard-coded paths (vla-isaac checkout, checkpoint, policy-wrapper).
# Usage: python rescore07.py [attempts_dir=~/auto_bug07] [bugged_repo=/tmp/pw-bug07]
import os, sys, json, shutil, subprocess
from pathlib import Path

here = Path(__file__).resolve().parent
src = (here / "cfg_check.py").read_text().split("for name, repo in")[0]
exec(src)   # defines run(repo), CKPT, ... (router code unchanged)

attempts = Path(sys.argv[1] if len(sys.argv) > 1 else "~/auto_bug07").expanduser()
bugged = sys.argv[2] if len(sys.argv) > 2 else "/tmp/pw-bug07"
ck = json.load(open(CKPT / "statistics.json"))["gr1"]

def same_stats(path):
    try:
        d = json.load(open(path))["gr1"]
    except Exception as e:
        return f"unreadable: {str(e)[:60]}"
    return "values equal checkpoint" if d == ck else "values DIFFER from checkpoint"

for i in range(1, 6):
    w = f"/tmp/rs07_{i}"
    shutil.rmtree(w, ignore_errors=True)
    shutil.copytree(bugged, w)
    patch = attempts / f"attempt{i}.patch"
    if not patch.exists() or patch.stat().st_size == 0:
        print(f"== A{i}: empty patch (unchanged bugged repo)")
    else:
        r = subprocess.run(["git", "-C", w, "apply", str(patch)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"== A{i}: patch did not apply: {r.stderr[:80]}")
            continue
        print(f"== A{i}: patch applied")
    print(f"   override file: {same_stats(w + '/config/statistics.json')}")
    try:
        ev = run(w)
        print(f"   config check: {len(ev)} item(s)", [e.id for e in ev])
    except Exception as e:
        print("   config check ERROR:", str(e)[:230])
