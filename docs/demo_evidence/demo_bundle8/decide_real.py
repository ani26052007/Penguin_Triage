import os
H = os.path.expanduser("~")
n1, n2 = {}, {}
exec(open(f"{H}/router_check.py").read().split("clean, bugged =")[0], n1)
exec(open(f"{H}/cfg_check.py").read().split("for name, repo in")[0], n2)
from router import vocab as V
from router.decide import decide
from router.schemas import ReplayResult, EpisodeResult, Outcome, Mode
E = f"{H}/router_ev/"
nom = n1["obs_nominal"](n1["traces"](E + "clean"))
ev = {"BUG-02": n1["check"](nom, n1["traces"](E + "bugged")),
      "BUG-07": n2["run"]("/tmp/pw-bug07"),
      "BUG-03": []}
unchecked = {V.SIG_STALENESS_VARIABLE, V.SIG_RENDER_PHYSICS_BROKEN, V.SIG_RENDER_SETTINGS_DIFFER,
             V.SIG_IMPLIED_TARGET_OFFSET, V.SIG_IMPLIED_TARGET_OK, V.SIG_IMPLIED_TARGET_WRONG,
             *(V.failure_phase(p) for p in V.PHASES)}
def rep(rid, ok, n):
    eps = tuple(EpisodeResult(i, i < ok) for i in range(n))
    return ReplayResult(rid, n1["SceneSpec"]("gr1_fridge_ranch"), n1["CODE"], n1["POLICY"], Mode.CLEAN, eps, Outcome.NO_EFFECT)
for bug, e in ev.items():
    replays = [rep(V.R_STAGE0, 0, 20), rep(V.R_STAGE2_NOMINAL, 0, 20)]
    print(f"== {bug}: {len(e)} real evidence item(s) {[x.id for x in e]}")
    try:
        v = decide(e, replays, unchecked=unchecked, unavailable={V.R_G3})
        print(f"   label={v.label.value} row={v.row} location={v.location} elimination={v.by_elimination}")
        print("   notes:", v.notes[:300])
    except Exception as ex:
        print("   decide ERROR:", type(ex).__name__, str(ex)[:300])
