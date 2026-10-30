import os
H = os.path.expanduser("~")
src = open(f"{H}/router_check.py").read().split("clean, bugged =")[0]
exec(src)   # traces, obs_nominal, check, CODE, SPEC, POLICY ... (router code unchanged)
from router.stage1.staleness import StalenessProbe, M_CLEAN_DELAY, M_MODE_GAP, delay_summary
from router.stage1.nominal import NominalStats
E = f"{H}/router_ev/"
runs = {"clean(11)": "clean", "clean(12)": "clean2", "clean(13)": "clean3", "clean(14)": "clean4",
        "BUG-02": "bugged", "BUG-01": "bug01", "BUG-09": "bug09"}
tr = {}
for k, d in runs.items():
    try:
        tr[k] = traces(E + d)
    except Exception as e:
        print(f"{k}: no traces ({str(e)[:70]})")
print("== observation check, nominal from clean(11)")
nom = obs_nominal(tr["clean(11)"])
fp = 0; n = 0
for k, t in tr.items():
    if k == "clean(11)":
        continue
    ev = check(nom, t)
    if k.startswith("clean"):
        n += 1; fp += bool(ev)
    print(f"  {k}: {len(ev)} item(s) {[e.id for e in ev]}")
print(f"  false positives on held-out clean runs: {fp}/{n}")
print("== extra action delay (steps), from the logged step pairs")
DN = NominalStats.from_samples({M_CLEAN_DELAY: [0.0, 0.0, 0.0], M_MODE_GAP: [0.0, 0.0, 0.0]}, default_min_std=0.1)
for k, t in tr.items():
    try:
        s = delay_summary(t)
        ev = StalenessProbe(DN, InMemoryTraceSource({"traces/clean": list(t)})).run(
            Incident("s", SceneSpec("gr1_fridge_ranch"), SPEC, CODE, POLICY, (0,), Mode.CLEAN, log_refs={"clean": "traces/clean"}))
        print(f"  {k}: median episode-mean delay {s.mean:.2f}, max {s.max}, probe evidence {[e.id for e in ev]}")
    except Exception as e:
        print(f"  {k}: staleness not run ({str(e)[:90]})")
