import os
src = open(os.path.expanduser("~/router_check.py")).read().split("clean, bugged =")[0]
exec(src)
h = os.path.expanduser("~/router_ev/")
c1, c2, b = traces(h + "clean"), traces(h + "clean2"), traces(h + "bugged")
nom = obs_nominal(c1)
for n, t in (("independent CLEAN run (seed 12)", c2), ("BUG-02 (seed 11)", b)):
    ev = check(nom, t)
    print(f"{n}: {len(ev)} evidence item(s)")
    for e in ev:
        print("  ", e.id, f"{e.strength:.2f}", e.detail[:110])
