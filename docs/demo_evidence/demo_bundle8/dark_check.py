import os
H = os.path.expanduser("~")
n = {}
exec(open(f"{H}/router_check.py").read().split("clean, bugged =")[0], n)
E = f"{H}/router_ev/"
nom = n["obs_nominal"](n["traces"](E + "clean"))
ev = n["check"](nom, n["traces"](E + "dark10"))
print(f"dark10: {len(ev)} item(s)")
for e in ev: print("  ", e.id, f"{e.strength:.2f}", e.detail[:140])
