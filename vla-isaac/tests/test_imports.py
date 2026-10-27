"""Hard rule: every module imports without Isaac Sim, Isaac Lab, Arena, GR00T, torch, h5py, zmq or
msgpack. Checked in a fresh interpreter whose import system refuses those packages."""

from __future__ import annotations

import pkgutil
import subprocess
import sys
from pathlib import Path

import vla_isaac

BLOCKED = ("torch", "numpy", "h5py", "zmq", "msgpack", "cv2", "isaacsim", "isaaclab", "isaaclab_arena",
           "isaaclab_arena_environments", "gr00t", "omni", "pxr")
PROGRAM = """
import importlib, importlib.abc, sys
BLOCKED = set(sys.argv[1].split(","))
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"blocked for the GPU-free import test: {name}")
sys.meta_path.insert(0, Block())
for m in sys.argv[2:]:
    importlib.import_module(m)
print("ok", len(sys.argv) - 2)
"""


def test_every_module_imports_without_heavy_dependencies():
    mods = [m.name for m in pkgutil.walk_packages(vla_isaac.__path__, "vla_isaac.")]
    assert "vla_isaac.backend.isaac_backend" in mods and "vla_isaac.arena_ext.launch_experiment" in mods
    p = subprocess.run([sys.executable, "-c", PROGRAM, ",".join(BLOCKED), *mods], capture_output=True, text=True,
                       cwd=Path(vla_isaac.__file__).parents[1])
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == f"ok {len(mods)}"
