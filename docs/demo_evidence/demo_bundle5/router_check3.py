import os
src = open(os.path.expanduser("~/router_check.py")).read().split("clean, bugged =")[0]
exec(src)
h = os.path.expanduser("~/router_ev/")
nom = obs_nominal(traces(h + "clean"))
for n, d in (("held-out CLEAN (seed 12)", "clean2"), ("BUG-02 rotation", "bugged"), ("BUG-01 channel swap", "bug01")):
    ev = check(nom, traces(h + d))
    print(f"{n}: {len(ev)} evidence item(s)")
    for e in ev:
        print("  ", e.id, f"{e.strength:.2f}", e.detail[:120])
