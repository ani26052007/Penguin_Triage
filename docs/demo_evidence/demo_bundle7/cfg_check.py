import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.expanduser("~/comp_and_research/Penguin_Triage/vla-isaac"))
from ladder.workspace import WorkspaceManager
from router.schemas import Incident, SceneSpec, TrainingSpec, Mode, ActorRef, ActorKind
from router.stage1.cfg_contract import ConfigContractCheck
from vla_isaac.config import load_task_config
from vla_isaac.specs.norm_stats import load_flat_statistics
from vla_isaac.workspace import config_mapping, eval_config_loader
H = os.path.expanduser("~"); R = f"{H}/comp_and_research/Penguin_Triage"
TASK = load_task_config(Path(f"{R}/vla-isaac/config/pilot_gr1_fridge.yaml"))
CKPT = Path(f"{H}/data/models/pilot/ckpt")
CKPT_ID = "nvidia/GN1.6-Tuned-Arena-GR1-PlaceItemCloseDoor-Task/ranch_bottle_into_fridge"
STATS = load_flat_statistics(CKPT / "statistics.json", "gr1", TASK.eval_config.norm_stats_groups)
SPEC = TrainingSpec("gr00t_n1d6_gr1_ranch_bottle_into_fridge", image_size=(512, 512), norm_stats=STATS,
    action_horizon=16, replan_steps=16, denoising_steps=4, control_hz=50, render_settings={},
    instruction_templates={"default": "Place the sauce bottle on the top shelf of the fridge, and close the fridge door."})
POLICY = ActorRef(ActorKind.POLICY, "ckpt")
def run(repo):
    with tempfile.TemporaryDirectory() as td:
        ws = WorkspaceManager(Path(td) / "ws", {"pw": repo}, config_loader=eval_config_loader(TASK, {CKPT_ID: CKPT}))
        inc = Incident("pw", SceneSpec(TASK.nominal_scene), SPEC, ws.base_ref("pw"), POLICY, (0,), Mode.CLEAN)
        return ConfigContractCheck(config_mapping(TASK)).run(inc)
for name, repo in (("CLEAN", f"{R}/policy-wrapper"), ("BUG-07", "/tmp/pw-bug07"), ("BUG-05", "/tmp/pw-bug05"), ("BUG-06", "/tmp/pw-bug06")):
    ev = run(repo); print(f"== {name}: {len(ev)} evidence item(s)")
    for e in ev: print("  ", e.id, f"{e.strength:.2f}", e.location, "|", e.detail[:110])
